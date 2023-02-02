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
import os
import optuna

from utils.simclr import simclr_loss_func
from utils.barlow_twin import barlow_loss_func 
from datasets.pretrain_time_dataloader import PretrainingTimeDataset,pretrain_time_dataloader
from datasets.bert_style_dataloader import BertDataset,bert_dataloader

from self_supervised_models.backbones import MLP
from callbacks.callbacks import SelfSupervisedCallback
from self_supervised_models.temporal_vit_time import TransformerEncoder

os.environ['CUDA_LAUNCH_BLOCKING']="1"

class BertStyleTemporalTransformer(pl.LightningModule):
    def __init__(self,planet_input_dims,sentinel_input_dims,d_model,n_head,num_layer,mlp_dim,dropout,projector_layer,loss,lr,**kwargs):
        super(BertStyleTemporalTransformer,self).__init__()
        self.sentinel_transformer_encoder = TransformerEncoder(sentinel_input_dims,d_model,n_head,num_layer,mlp_dim,dropout,projector_layer)
        self.projector = (nn.Linear(d_model,planet_input_dims)
                            if projector_layer == 0
                            else MLP(d_model,projector_layer,planet_input_dims))
        self.mlp_dim = mlp_dim
        self.planet_input_dims = planet_input_dims
        self.loss = loss
        self.lr = lr
        self.config = {'d_model':d_model,
                       'n_head':n_head,
                       'num_layer':num_layer,
                       'mlp_dim':mlp_dim,
                       'dropout':dropout,
                       'projector_layer':projector_layer}

    def forward(self,x):
        return self.sentinel_transformer_encoder.return_embeddings(x)

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
        return parent_parser

    @staticmethod
    def return_hyper_parameter_args():
        return ["d_model","n_head","num_layer","mlp_dim","lr","dropout"]


    def training_step(self,batch,batch_idx):
        sentinel_embeddings,index,planet_val = batch
        sentinel_embeddings = self.sentinel_transformer_encoder.return_embeddings(sentinel_embeddings)
        N,t,c = sentinel_embeddings.shape
        planet_predictions = self.projector(torch.reshape(sentinel_embeddings,(-1,c))).reshape(N,t,-1)
        planet_prediction_samples = planet_predictions[:,index,:][torch.arange(N,device=planet_predictions.device),torch.arange(N,device=planet_predictions.device),:,:]
        loss = F.mse_loss(planet_val,planet_prediction_samples)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,int(0.05*self.trainer.max_epochs),self.trainer.max_epochs,5e-5)
        return [optimizer],[scheduler]

if __name__ == "__main__":
    
    bert_style_temporal_transformer = BertStyleTemporalTransformer(36,12,32,4,4,64,1e-5,0,torch.nn.MSELoss(),1e-3)
    bert_dataset = BertDataset("../utils/h5_folder/pretraining_time2.h5",is_normalize=False)
    bert_dataloader = bert_dataloader(bert_dataset,128,8,True,True,True)
    wandb_logger = WandbLogger(project="planet_sentinel_multimodality_bert_style_self_supervised")
    trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=1000,
            logger=wandb_logger)

    trainer.fit(bert_style_temporal_transformer,bert_dataloader)
