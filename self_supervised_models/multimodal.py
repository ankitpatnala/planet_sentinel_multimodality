import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from utils.simclr import simclr_loss_func
from datasets.pretrain_dataloader import PretrainingDataset,pretrain_dataloader

from self_supervised_models.backbones import MLP,ResMLP 
from callbacks.callbacks import SelfSupervisedCallback

def scarf(percentage,batch_data):
    if torch.randn(1).uniform_(0,1) < 0.5 :
        if percentage < 100:
            n,c = batch_data.shape
            channels_to_corrupt = torch.randperm(c)[:int(percentage*c/100)]
            replacement_data = (torch.randint(1,n-1,size=(n,))+torch.arange(n))%n
            batch_data[:,channels_to_corrupt] = batch_data[replacement_data][:,channels_to_corrupt]
    return batch_data


class Multimodal(pl.LightningModule):
    def __init__(
            self,
            planet_input_dims,
            sentinel_input_dims,
            num_layers,
            hidden_dim,
            loss,
            temperature,
            lr,
            pretrain_type='mlp',
            scarf=100,
            projector_layer = 2,
            **kwargs):
        super(Multimodal,self).__init__()
        backbone_model = MLP if pretrain_type == 'mlp' else ResMLP
        self.backbone_sentinel = backbone_model(sentinel_input_dims,num_layers,hidden_dim)
        self.backbone_planet =  backbone_model(planet_input_dims,num_layers,hidden_dim)
        self.projector_sentinel = (nn.Linear(hidden_dim,hidden_dim)
                                    if projector_layer == 0  
                                    else MLP(hidden_dim,projector_layer,hidden_dim))
        self.projector_planet = (nn.Linear(hidden_dim,hidden_dim)
                                  if projector_layer == 0 
                                  else MLP(hidden_dim,projector_layer,hidden_dim))
        self.loss = loss
        self.temperature = temperature
        self.lr = lr
        self.scarf = scarf
        self.config = {'num_layers': num_layers,
                'hidden_dim': hidden_dim}
        self.downstream_accuracy = 0.0

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("mulitmodal")
        parser.add_argument("--num_layers",type=int,nargs="+",default=[4])
        parser.add_argument("--hidden_dim",type=int,nargs='+',default=[256])
        parser.add_argument("--lr",type=float,nargs="+",default=[1e-3,1e-3])
        parser.add_argument("--dropout",type=float,nargs="+",default=[0.0,0.0])
        parser.add_argument("--scarf",type=int,default=100)
        parser.add_argument("--projector_layer",type=int,default=2)
        return parent_parser

    @staticmethod
    def return_hyper_parameter_args():
        return ["num_layers","hidden_dim","lr","dropout"]
    
    def training_step(self,batch,batch_idx):
        x1,x2 = batch
        x1 = scarf(self.scarf,x1)
        x2 = scarf(self.scarf,x2)
        y1 = self.backbone_sentinel(x1)
        y2 = self.backbone_planet(x2)
        z1 = self.projector_sentinel(y1)
        z2 = self.projector_planet(y2)
        loss = self.loss(z1,z2,self.temperature)
        self.log_dict({'simclr_loss':loss},prog_bar=True)
        return loss

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(),lr=self.lr,weight_decay=1e-5)]


def run_multimodal(backbone='mlp'):
    if backbone == 'mlp':
        sentinel_mlp = MLP(12,4,256)
        planet_mlp = MLP(36,4,256)
    if backbone == 'resmlp':
        sentinel_mlp = ResMLP(12,4,256)
        planet_mlp = ResMLP(36,4,256)
    sentinel_projector = MLP(256,2,256)
    planet_projector = MLP(256,2,256)
    multimodal = Multimodal(
                        sentinel_mlp,
                        planet_mlp,
                        sentinel_projector,
                        planet_projector,
                        simclr_loss_func,
                        1.0,
                        0.001)
    pretraining_dataset = PretrainingDataset("../../planet_sentinel_multimodality/utils/h5_folder/pretraining_point.h5")
    pretraining_dataloader = pretrain_dataloader(pretraining_dataset,2048,32,True,True)
    wandb_logger = WandbLogger(project="planet_sentinel_multimodality_self_supervised",
                               version=f'multi_modal_self_supervised_backbone_{backbone}_callbacktest')
    self_supervised_callback = SelfSupervisedCallback()
    trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=1000,
            logger=wandb_logger,
            callbacks=[self_supervised_callback])
    trainer.fit(multimodal,pretraining_dataloader)

if __name__ == "__main__":
    run_multimodal('mlp')


    


