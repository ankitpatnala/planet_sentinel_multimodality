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

def return_self_supervised_model_sentinel2(ckpt_path):
    sentinel_mlp = MLP(12,4,256)
    ckpt = torch.load(ckpt_path)
    new_ckpt = {}
    for key in ckpt['state_dict'].keys():
        if 'backbone_sentinel' in key:
            mlp_key = re.sub('backbone_sentinel.',"",key)
            new_ckpt[mlp_key] = ckpt['state_dict'][key]
    sentinel_mlp.load_state_dict(new_ckpt)
    return sentinel_mlp,256

class LSTM(pl.LightningModule):
    def __init__(
            self,
            input_dim,
            num_classes,
            hidden_dims=128,
            num_layers=4,
            loss=F.cross_entropy,
            optimizer=torch.optim.Adam,
            lr=0.001,
            self_supervised_ckpt=None):
        super(LSTM,self).__init__()
        self.save_hyperparameters()
        if self_supervised_ckpt is not None:
            self.self_supervised,input_dim = return_self_supervised_model_sentinel2(self_supervised_ckpt)
            for param in self.self_supervised.parameters():
                param.requires_grad=False
            self.self_supervised.eval()
            self.embedding = True
        else :
            self.embedding = False
        self.lstm = bzh.models.LSTM(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dims=hidden_dims,
                num_layers=num_layers)
        self.loss = loss
        self.optim = optimizer
        self.lr = lr
        self.accuracy = torchmetrics.Accuracy()
        self.accuracy_score = 0.0 


    def training_step(self,batch,batch_idx):
        x,y = batch
        if self.embedding:
            with torch.no_grad():
                x = self.self_supervised(x)
        y_pred = self.lstm(x)
        loss = self.loss(y_pred,y-1)
        self.log_dict({'training_loss':loss},prog_bar=True)
        return loss

    def validation_step(self,batch,batch_idx):
        x,y = batch
        if self.embedding:
            with torch.no_grad():
                x = self.self_supervised(x)
        y_pred = self.lstm(x)
        loss = self.loss(y_pred,y-1)
        acc  = self.accuracy(y_pred,y-1)
        return {'val_loss':loss,'val_acc':acc}

        #self.log_dict({'validation_loss':loss,'validation_acc':acc},prog_bar=True)

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
        return self.optim(self.lstm.parameters(),lr=self.lr)

def train_lstm(trial,ckpt_path=None):
    lr = trial.suggest_float("lr",1e-5,1e-3,log=True)
    hidden_dims = trial.suggest_categorical("hidden_dims",[32,64,128,256])
    num_layers = trial.suggest_categorical("num_layers",[2,3,4,5,6])
    lstm = LSTM(input_dim=12,num_classes=9,hidden_dims=hidden_dims,num_layers=num_layers,lr=lr,self_supervised_ckpt=ckpt_path)
    train_dataset = s2_loader.Sentinel2Dataset("../utils/h5_folder/train_sentinel_ts.hdf5")
    train_dataloader = s2_loader.sentinel2_dataloader(train_dataset,256,8,True,True)
    val_dataset = s2_loader.Sentinel2Dataset("../utils/h5_folder/val_sentinel_ts.hdf5")
    val_dataloader = s2_loader.sentinel2_dataloader(val_dataset,256,8,True,False)
    config = {'lr': lstm.lr, 
                'hidden_dims': lstm.lstm.lstm.hidden_size,
                'num_layers' : lstm.lstm.lstm.num_layers}
    wandb_logger = WandbLogger(project="planet_sentinel_multimodality_downstream",
                               config=config)
    #wandb_logger.experiment.config.update(
    #        {'lr': lstm.lr, 
    #    'hidden_dims': lstm.lstm.lstm.hidden_size,
    #    'num_layers' : lstm.lstm.lstm.num_layers})


    trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=50,
            logger=wandb_logger)
    trainer.fit(lstm,train_dataloader,val_dataloader)

    return lstm.accuracy_score

class HyperParameterCallback:
    def __init__(self,pickle_file):
        self.pickle_file = pickle_file

    def __call__(self,study,trial):
        joblib.dump(study,self.pickle_file)


def hyper_parameter_sweeping(pickle_file=None,ckpt_path=None):
    if pickle_file is None:
        hyper_parameter_callback = HyperParameterCallback("./study2.pkl")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: train_lstm(
            trial,
            ckpt_path=ckpt_path),
            n_trials=10,
            callbacks=[hyper_parameter_callback])
    else :
        hyper_parameter_callback = HyperParameterCallback(pickle_file)
        study = joblib.load(pickle_file)
        study.optimize(lambda trial: train_lstm(
            trial,
            ckpt_path="lightning_logs/version_42105/checkpoints/epoch=826-step=88489.ckpt"),
            n_trials=10,
            callbacks=[hyper_parameter_callback])


if __name__ == "__main__":
    hyper_parameter_sweeping()

    #train_lstm()






