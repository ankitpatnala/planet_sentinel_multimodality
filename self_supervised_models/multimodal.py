import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from utils.simclr import simclr




class MLP(nn.Module):
    def __init__(self,num_layers,hidden_dim):


class Multimodal(pl.LightningModule):
    def __init__(
            self,
            backbone_sentinel,
            backbone_planet,
            projetor_sentinel,
            projectr_planet,
            loss,
            temperature,
            learning_rate):
        self.backbone_sentinel = backbone_sentinel
        self.backbone_planet = backbone_planet
        self.projector_sentinel = projector_sentinel
        self.projector_planet = projector_planet
        self.loss = loss
        self.temperature = temperature
        self.lr = learning_rate


    def training_step(self,batch,batch_idx):
        x1,x2 = batch
        y1 = self.backbone_sentinel(x)
        y2 = self.backbone_planet(x)
        z1 = self.projector_sentinel(y1)
        z2 = self.projector_planet(y2)
        loss = self.loss(z1,z2,self.temperature)
        self.log_dict({'simclr_loss':loss},prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.lr)


