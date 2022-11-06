import breizhcrops as bzh
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torchmetrics
from pytorch_lightning.loggers import WandbLogger
import wandb
import optuna

from self_supervised_models.backbones import MLP, ResMLP
from self_supervised_models.transformer_encoder import TransformerEncoder 

from collections import OrderedDict
import re

import h5py
import sys
import joblib

sys.path.append("../../planet_sentinel_multi_modality")

from datasets import sentinel2_dataloader as s2_loader

def return_self_supervised_model_sentinel2(ckpt_path,type='mlp',config=None):
    if type == "mlp":
        sentinel_mlp = MLP(12,4,256)
        model_name = 'backbone_sentinel'
        emb_dim = 256
    if type == "resmlp":
        sentinel_mlp = ResMLP(12,4,256)
        model_name = 'backbone_sentinel'
        emb_dim = 256
    if type == 'temporal_transformer':
        sentinel_mlp = TransformerEncoder(
                12,
                32,
                16,   #config['n_head'],
                4,   #config['num_layer'],
                128,   #config['mlp_dim'],
                0.2)
        model_name = 'sentinel_transformer_encoder'
        emb_dim = 128 #config['mlp_dim']
    ckpt = torch.load(ckpt_path)
    new_ckpt = OrderedDict()
    for key in ckpt['state_dict'].keys():
        if model_name in key:
            mlp_key = re.sub(f'{model_name}.',"",key)
            new_ckpt[mlp_key] = ckpt['state_dict'][key]
    sentinel_mlp.load_state_dict(new_ckpt)
    return sentinel_mlp,emb_dim

def run_time_series_with_mlp(model,x):
    b,t,n = x.shape
    if hasattr(model,'return_embeddings'):
        x =  model.return_embeddings(x)
        return x
    x = model(x.reshape(-1,n)).reshape(b,t,-1)
    return x

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
            dropout=0.0,
            self_supervised_ckpt=None,
            **kwargs):
        super(LSTM,self).__init__()
        self.save_hyperparameters()
        if self_supervised_ckpt is not None:
            self.self_supervised,input_dim = return_self_supervised_model_sentinel2(self_supervised_ckpt,type='temporal_transformer',**kwargs)
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
                num_layers=num_layers,
                dropout=dropout)
        self.loss = loss
        self.optim = optimizer
        self.lr = lr
        self.accuracy = torchmetrics.Accuracy()
        self.accuracy_score = 0.0 

    def training_step(self,batch,batch_idx):
        x,y = batch
        if self.embedding:
            with torch.no_grad():
                x = run_time_series_with_mlp(self.self_supervised,x)
        y_pred = self.lstm(x)
        loss = self.loss(y_pred,y-1)
        self.log_dict({'training_loss':loss},prog_bar=True)
        return loss

    def validation_step(self,batch,batch_idx):
        x,y = batch
        if self.embedding:
            with torch.no_grad():
                x = run_time_series_with_mlp(self.self_supervised,x)
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
    dropout = trial.suggest_uniform("dropout",0.0,0.6)
    lstm = LSTM(
            input_dim=12,
            num_classes=9,
            hidden_dims=hidden_dims,
            num_layers=num_layers,
            lr=lr,
            dropout=dropout,
            self_supervised_ckpt=ckpt_path)
    train_dataset = s2_loader.Sentinel2Dataset("../utils/h5_folder/train_sentinel_ts.hdf5")
    train_dataloader = s2_loader.sentinel2_dataloader(train_dataset,256,8,True,True)
    val_dataset = s2_loader.Sentinel2Dataset("../utils/h5_folder/val_sentinel_ts.hdf5")
    val_dataloader = s2_loader.sentinel2_dataloader(val_dataset,256,8,True,False)
    config = {'lr': lstm.lr, 
                'hidden_dims': lstm.lstm.lstm.hidden_size,
                'num_layers' : lstm.lstm.lstm.num_layers,
                'dropout': lstm.lstm.lstm.dropout}
    wandb_logger = WandbLogger(project="planet_sentinel_multimodality_downstream",
                               config=config,
                               version=f'lstm_lr_{lr}_hidden_dims_{hidden_dims}_num_layers_{num_layers}_dropout{dropout}')
    trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=50,
            logger=wandb_logger)
    trainer.fit(lstm,train_dataloader,val_dataloader)

    wandb.finish()

    return lstm.accuracy_score

class HyperParameterCallback:
    def __init__(self,pickle_file):
        self.pickle_file = pickle_file

    def __call__(self,study,trial):
        joblib.dump(study,self.pickle_file)


def hyper_parameter_sweeping(pickle_file=None,ckpt_path=None):
    if pickle_file is None:
        hyper_parameter_callback = HyperParameterCallback("./lstm_study_resmlp_self_supervised.pkl")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: train_lstm(
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
    #hyper_parameter_sweeping(ckpt_path="/p/project/deepacf/kiste/patnala1/planet_sentinel_multimodality/slurm_scripts/planet_sentinel_multimodality_downstream/multi_modal_self_supervised/checkpoints/epoch=999-step=107000.ckpt")
    #hyper_parameter_sweeping(ckpt_path="/p/project/deepacf/kiste/patnala1/planet_sentinel_multimodality/slurm_scripts/planet_sentinel_multimodality_self_supervised/multi_modal_self_supervised_backbone_resmlp/checkpoints/epoch=999-step=107000.ckpt")

    hyper_parameter_sweeping(ckpt_path="/p/project/deepacf/kiste/patnala1/planet_sentinel_multimodality/slurm_scripts/planet_sentinel_multimodality_self_supervised/temporal_contrastive_learing_d_model_64_n_head_4_n_layers_8_mlp_dim_128_lr_0.0008778568797244744/checkpoints/epoch=999-step=24000-v15.ckpt")
