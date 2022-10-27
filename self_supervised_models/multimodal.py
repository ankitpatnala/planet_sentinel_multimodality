import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from utils.simclr import simclr_loss_func
from datasets.pretrain_dataloader import PretrainingDataset,pretrain_dataloader


from self_supervised_models.backbones import MLP,ResMLP 

class Multimodal(pl.LightningModule):
    def __init__(
            self,
            backbone_sentinel,
            backbone_planet,
            projector_sentinel,
            projector_planet,
            loss,
            temperature,
            learning_rate):
        super(Multimodal,self).__init__()
        self.backbone_sentinel = backbone_sentinel
        self.backbone_planet = backbone_planet
        self.projector_sentinel = projector_sentinel
        self.projector_planet = projector_planet
        self.loss = loss
        self.temperature = temperature
        self.lr = learning_rate


    def training_step(self,batch,batch_idx):
        x1,x2 = batch
        y1 = self.backbone_sentinel(x1)
        y2 = self.backbone_planet(x2)
        z1 = self.projector_sentinel(y1)
        z2 = self.projector_planet(y2)
        loss = self.loss(z1,z2,self.temperature)
        self.log_dict({'simclr_loss':loss},prog_bar=True)
        return loss

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(),lr=self.lr)]


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
                               version=f'multi_modal_self_supervised_backbone_{backbone}')
    trainer = pl.Trainer(accelerator='gpu',devices=1,max_epochs=1000,logger=wandb_logger)
    trainer.fit(multimodal,pretraining_dataloader)

if __name__ == "__main__":
    run_multimodal('resmlp')


    


