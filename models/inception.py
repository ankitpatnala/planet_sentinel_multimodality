import breizhcrops as bzh
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torchmetrics
from pytorch_lightning.loggers import WandbLogger
import wandb
import optuna

from self_supervised_models.multimodal import MLP
from collections import OrderedDict
import re

import h5py
import sys
import joblib

sys.path.append("../../planet_sentinel_multi_modality")

from datasets import sentinel2_dataloader as s2_loader
from models.return_self_supervised_model import return_self_supervised_model_sentinel2,run_time_series_with_mlp

class InceptionTime(pl.LightningModule):
    def __init__(
            self,
            input_dim,
            num_classes,
            num_layers=6,
            hidden_dims=128,
            kernel_size=40,
            use_bias=False,
            loss=F.cross_entropy,
            optimizer=torch.optim.Adam,
            lr=0.001,
            self_supervised_ckpt=None,
            **kwargs):
        super(InceptionTime,self).__init__()
        self.save_hyperparameters()
        if self_supervised_ckpt is not None:
            self.self_supervised,input_dim = return_self_supervised_model_sentinel2(self_supervised_ckpt,**kwargs)
            for param in self.self_supervised.parameters():
                param.requires_grad=False
            self.self_supervised.eval()
            self.embedding = True
        else :
            self.embedding = False
        self.inception_time = bzh.models.InceptionTime(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    num_layers=num_layers,
                    hidden_dims=hidden_dims,
                    kernel_size=kernel_size,
                    use_bias=True,
                    device=torch.device('cuda')) #.to(torch.device('cuda'))
        self.loss = loss
        self.optim = optimizer
        self.lr = lr
        self.accuracy = torchmetrics.Accuracy()
        self.accuracy_score = 0.0 
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("inception")
        parser.add_argument("--hidden_dims",type=int,nargs="+",default=[128])
        parser.add_argument("--num_layers",type=int,nargs='+',default=[4])
        parser.add_argument("--lr",type=float,nargs="+",default=[1e-3,1e-3])
        parser.add_argument("--kernel_size",type=int,nargs="+",default=[40])
        return parent_parser

    @staticmethod
    def return_hyper_parameter_args():
        return ["hidden_dims","num_layers","lr","kernel_size"]

    def training_step(self,batch,batch_idx):
        x,y = batch
        if self.embedding:
            with torch.no_grad():
                x = run_time_series_with_mlp(self.self_supervised,x)
        y_pred = self.inception_time(x)
        loss = self.loss(y_pred,y-1)
        self.log_dict({'training_loss':loss},prog_bar=True)
        return loss

    def validation_step(self,batch,batch_idx):
        x,y = batch
        if self.embedding:
            with torch.no_grad():
                x = run_time_series_with_mlp(self.self_supervised,x)
        y_pred = self.inception_time(x)
        loss = self.loss(y_pred,y-1)
        acc  = self.accuracy(y_pred,y-1)
        return {'val_loss':loss,'val_acc':acc}

    def validation_epoch_end(self,outputs):
        loss = []
        acc = []
        for output in outputs:
            loss.append(output['val_loss'])
            acc.append(output['val_acc'])
        self.accuracy_score = torch.mean(torch.Tensor(acc))
        loss = torch.mean(torch.Tensor(loss))
        self.log_dict({"val_loss":loss,"val_acc":self.accuracy_score},prog_bar=True)

    def configure_optimizers(self):
        return self.optim(self.parameters(),lr=self.lr)

def train_transformer(trial,ckpt_path=None):
    lr = trial.suggest_float("lr",1e-5,1e-3,log=True)
    num_layers = trial.suggest_categorical("num_layers",[2,4,8])
    hidden_dims = trial.suggest_categorical("hidden_dims",[128,256,512,1024])
    kernel_size = trial.suggest_categorical("kernel_size",[40,80,120,136])
    inception_time = InceptionTime(
            input_dim=12,
            num_classes=9,
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            lr=lr,
            self_supervised_ckpt=ckpt_path) #.to(torch.device('cuda'))
    train_dataset = s2_loader.Sentinel2Dataset("../utils/h5_folder/train_sentinel_ts.hdf5")
    train_dataloader = s2_loader.sentinel2_dataloader(train_dataset,256,8,True,True)
    val_dataset = s2_loader.Sentinel2Dataset("../utils/h5_folder/val_sentinel_ts.hdf5")
    val_dataloader = s2_loader.sentinel2_dataloader(val_dataset,256,8,True,False)
    config = {'lr': lr, 
                'num_layers': num_layers,
                'hidden_dims' : hidden_dims,
                'kernel_size' : kernel_size}
    wandb_logger = WandbLogger(project="planet_sentinel_multimodality_downstream",
                               config=config,
                               version=f'inception_lr_{lr}_num_layers_{num_layers}_hidden_dims_{hidden_dims}_kernel_size_{kernel_size}')
    trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=50,
            logger=wandb_logger)
    trainer.fit(inception_time,train_dataloader,val_dataloader)
    wandb.finish()
    return inception_time.accuracy_score

class HyperParameterCallback:
    def __init__(self,pickle_file):
        self.pickle_file = pickle_file

    def __call__(self,study,trial):
        joblib.dump(study,self.pickle_file)


def hyper_parameter_sweeping(pickle_file=None,ckpt_path=None):
    if pickle_file is None:
        hyper_parameter_callback = HyperParameterCallback("./inception_study.pkl")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: train_transformer(
            trial,
            ckpt_path=ckpt_path),
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
    hyper_parameter_sweeping("./inception_study.pkl")

