import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import numpy as np

import pickle
import math
import joblib
import copy

import optuna

from utils.simclr import simclr_loss_func
from utils.barlow_twin import barlow_loss_func 
from datasets.pretrain_time_dataloader import PretrainingTimeDataset,pretrain_time_dataloader

from self_supervised_models.backbones import MLP
from callbacks.callbacks import SelfSupervisedCallback
#from self_supervised_models.transformer_encoder import TransformerEncoder
from self_supervised_models.temporal_vit_time import TransformerEncoder

def temporal_scarf(percentage,batch_data):
    is_channel_corrupt = torch.rand((1,)) < 0.5
    if percentage < 100 :
        n,t,c = batch_data.shape
        replacement_data = (torch.randint(1,n-1,size=(n,))+torch.arange(n))%n
        if is_channel_corrupt:
            channels_to_corrupt = torch.randperm(c)[:int(percentage*c/100)]
            batch_data[:,:,channels_to_corrupt] = batch_data[replacement_data][:,:,channels_to_corrupt]
        else:
            time_stamps_to_corrupt = torch.randperm(t)[:int(percentage*t/100)]
            batch_data[:,time_stamps_to_corrupt,:] = batch_data[replacement_data][:,time_stamps_to_corrupt,:]
    return batch_data
        



class Sentinel2TemporalContrastiveLearning(pl.LightningModule):
    def __init__(self,sentinel_input_dims,d_model,n_head,num_layer,mlp_dim,dropout,loss,temperature,lr,scarf,is_mixup,projector_layer,is_seasonal,**kwargs):
        super(Sentinel2TemporalContrastiveLearning,self).__init__()
        self.sentinel_transformer_encoder = TransformerEncoder(sentinel_input_dims,d_model,n_head,num_layer,mlp_dim,dropout,projector_layer,is_seasonal=is_seasonal)
        self.loss = loss
        self.temperature = temperature
        self.lr = lr
        self.downstream_accuracy = 0
        self.is_mixup = is_mixup
        self.is_seasonal = is_seasonal
        self.scarf = scarf
        self.config = {'d_model':d_model,
                       'n_head':n_head,
                       'num_layer':num_layer,
                       'mlp_dim':mlp_dim,
                       'dropout':dropout,
                       'projector_layer':projector_layer}
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("temporal_transfomer")
        parser.add_argument("--d_model",type=int,nargs="+",default=[128])
        parser.add_argument("--n_head",type=int,nargs='+',default=[4])
        parser.add_argument("--num_layer",type=int,nargs="+",default=[4])
        parser.add_argument("--mlp_dim",type=int,nargs="+",default=[256])
        parser.add_argument("--lr",type=float,nargs="+",default=[1e-3,1e-3])
        parser.add_argument("--dropout",type=float,nargs="+",default=[0.0,0.0])
        parser.add_argument("--is_mixup",action='store_true')
        parser.add_argument("--is_seasonal",action='store_true')
        parser.add_argument("--projector_layer",type=int,default=2)
        parser.add_argument("--scarf",type=int,default=20)
        return parent_parser

    @staticmethod
    def return_hyper_parameter_args():
        return ["d_model","n_head","num_layer","mlp_dim","lr","dropout"]
    
    def training_step(self,batch,batch_idx):
        x1,_ = batch
        if self.is_mixup:
            random_num = torch.randn((1),device=x1.device).uniform_(0,1)
            n = x1.shape[0]
            x1_reverse = x1[[i for i in range(n-1,-1,-1)]]
            x1_mix = x1*random_num + x1_reverse*(1-random_num)
            x1 = torch.cat([x1,x1_reverse],dim=0)
        x2 = temporal_scarf(self.scarf,x1)
        y1_embedding,y1_projector = self.sentinel_transformer_encoder(x1)
        y2_embedding,y2_projector = self.sentinel_transformer_encoder(x2)
        loss = self.loss(y1_projector,y2_projector,self.temperature)
            
        if self.is_seasonal:
            y1_season = self.sentinel_transformer_encoder.return_chunk_embeddings(x1)
            y2_season = self.sentinel_transformer_encoder.return_chunk_embeddings(x2)
            seasonal_loss_y1 = (F.cross_entropy(y1_season[0],torch.ones(x1.shape[0],device=x1.device,dtype=torch.int64)*0) +
                               F.cross_entropy(y1_season[1],torch.ones(x1.shape[0],device=x1.device,dtype=torch.int64)*1) +
                               F.cross_entropy(y1_season[2],torch.ones(x1.shape[0],device=x1.device,dtype=torch.int64)*2) +
                               F.cross_entropy(y1_season[3],torch.ones(x1.shape[0],device=x1.device,dtype=torch.int64)*3))

            seasonal_loss_y2 = (F.cross_entropy(y2_season[0],torch.ones(x2.shape[0],device=x1.device,dtype=torch.int64)*0) +
                               F.cross_entropy(y2_season[1],torch.ones(x2.shape[0],device=x1.device,dtype=torch.int64)*1) +
                               F.cross_entropy(y2_season[2],torch.ones(x2.shape[0],device=x1.device,dtype=torch.int64)*2) +
                               F.cross_entropy(y2_season[3],torch.ones(x2.shape[0],device=x1.device,dtype=torch.int64)*3))

            seasonal_loss = seasonal_loss_y1 + seasonal_loss_y2
        else :
            seasonal_loss = 0
        self.log_dict({'simclr_loss':loss,
                        'seasonal_loss':seasonal_loss,
                        'total_loss': loss+seasonal_loss},prog_bar=True)

        return loss+seasonal_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,int(0.05*self.trainer.max_epochs),self.trainer.max_epochs,5e-5)
        return [optimizer],[scheduler]
