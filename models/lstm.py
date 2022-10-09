import breizhcrops as bzh
import pytorch_lightning as pl
import torch.nn.functional as F
import torch

import h5py
import sys

sys.path.append("../../planet_sentinel_multi_modality")

from datasets import sentinel2_dataloader as s2_loader

class LSTM(pl.LightningModule):
    def __init__(
            self,
            input_dim,
            num_clases,
            hidden_dims=128,
            num_layers=4,
            loss=F.cross_entropy,
            optimizer=torch.optim.Adam,
            lr=0.001):
        super(LSTM,self).__init__()
        self.lstm = bzh.models.LSTM(input_dim=input_dim,num_clases=num_clases,hidden_dims=hidden_dims,num_layers=num_layers)
        self.loss = loss
        self.optim = optimizer
        self.lr = lr
        

    def training_step(self,batch,batch_idx):
        x,y = batch
        y_pred = self.lstm(x)
        return self.loss(y_pred,y)

    def validation_step(self,batch,batch_idx):
        x,y = batch
        y_pred = self.lstm(x)
        return self.loss(y_pred,y)

    def configuring_optimizers(self):
        return self.optimizer(self.lstm.parameter(),lr=self.lr)

if __name__ == "__main__":


