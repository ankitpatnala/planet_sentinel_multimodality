import torch
from torch import nn
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

class TemporalContrastiveLearning(pl.LightningModule):
    def __init__(self,planet_input_dims,sentinel_input_dims,d_model,n_head,num_layer,mlp_dim,dropout,loss,temperature,lr,is_mixup,projector_layer,**kwargs):
        super(TemporalContrastiveLearning,self).__init__()
        self.planet_transformer_encoder = TransformerEncoder(planet_input_dims,d_model,n_head,num_layer,mlp_dim,dropout,projector_layer,mode_type='planet')
        self.sentinel_transformer_encoder = TransformerEncoder(sentinel_input_dims,d_model,n_head,num_layer,mlp_dim,dropout,projector_layer)
        self.loss = loss
        self.temperature = temperature
        self.lr = lr
        self.downstream_accuracy = 0
        self.is_mixup = is_mixup
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
        parser.add_argument("--projector_layer",type=int,default=2)
        return parent_parser

    @staticmethod
    def return_hyper_parameter_args():
        return ["d_model","n_head","num_layer","mlp_dim","lr","dropout"]
    
    def training_step(self,batch,batch_idx):
        x1,x2 = batch
        if self.is_mixup:
            random_num = torch.randn((1),device=x1.device).uniform_(0,1)
            n = x1.shape[0]
            x1_reverse = x1[[i for i in range(n-1,-1,-1)]]
            x2_reverse = x2[[i for i in range(n-1,-1,-1)]]
            x1_mix = x1*random_num + x1_reverse*(1-random_num)
            x2_mix = x2*random_num + x2_reverse*(1-random_num)
            x1 = torch.cat([x1,x1_reverse],dim=0)
            x2 = torch.cat([x2,x2_reverse],dim=0)
        y1_embedding,y1_projector = self.sentinel_transformer_encoder(x1)
        y2_embedding,y2_projector = self.planet_transformer_encoder(x2)
        loss = self.loss(y1_projector,y2_projector,self.temperature)
        #loss = self.loss(y1_projector,y2_projector)
        self.log_dict({'simclr_loss':loss},prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,int(0.05*self.trainer.max_epochs),self.trainer.max_epochs,5e-5)
        return [optimizer],[scheduler]

def run_temporal_contrastive(trial):
    d_model = trial.suggest_categorical("d_model",[32,64,128,256])
    n_head = trial.suggest_categorical("n_head",[4,8,16])
    n_layers = trial.suggest_categorical("n_layers",[2,4,6,8])
    mlp_dim = trial.suggest_categorical("mlp_dim",[64,128,256,512])
    lr = trial.suggest_uniform("lr",1e-5,1e-3)

    temporal_contrastive = TemporalContrastiveLearning(36,12,d_model,n_head,n_layers,mlp_dim,0.2,simclr_loss_func,0.07,lr,True)
    pretraining_time_dataset = PretrainingTimeDataset("../utils/h5_folder/pretraining_time.h5")
    pretraining_time_dataloader = pretrain_time_dataloader(pretraining_time_dataset,128,16,True,True)
    config = {'d_model':d_model,
              'n_head' : n_head,
              'n_layers':n_layers,
              'mlp_dim':mlp_dim,
              'lr' : lr}


    wandb_logger = WandbLogger(project="planet_sentinel_multimodality_self_supervised",
                               config=config,
                               version=f'temporal_contrastive_learing_d_model_{d_model}_n_head_{n_head}_n_layers_{n_layers}_mlp_dim_{mlp_dim}_lr_{lr}')

    callback = SelfSupervisedTransformerCallback()

    trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=1000,
            logger=wandb_logger,
            callbacks=[callback])

    trainer.fit(temporal_contrastive,pretraining_time_dataloader)
    return temporal_contrastive.downstream_accuracy

class HyperParameterCallback:
    def __init__(self,pickle_file):
        self.pickle_file = pickle_file

    def __call__(self,study,trial):
        joblib.dump(study,self.pickle_file)

def hyper_parameter_sweeping(pickle_file=None,ckpt_path=None):
    if pickle_file is None:
        hyper_parameter_callback = HyperParameterCallback("./temporal_contrastive_self_supervised_pos_2_batch128_normalize_mixupi_adamw.pkl")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: run_temporal_contrastive(
            trial),
            n_trials=25,
            callbacks=[hyper_parameter_callback])
    else :
        hyper_parameter_callback = HyperParameterCallback(pickle_file)
        study = joblib.load(pickle_file)
        study.optimize(lambda trial: train_lstm(
            trial,
            ckpt_path=ckpt_path),
            n_trials=25,
            callbacks=[hyper_parameter_callback])

        
if __name__ == "__main__":
    #run_temporal_contrastive()
    hyper_parameter_sweeping()


    

    

    


