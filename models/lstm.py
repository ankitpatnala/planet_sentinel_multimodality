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
            num_classes,
            hidden_dims=128,
            num_layers=4,
            loss=F.cross_entropy,
            optimizer=torch.optim.Adam,
            lr=0.001):
        super(LSTM,self).__init__()
        self.lstm = bzh.models.LSTM(input_dim=input_dim,num_classes=num_classes,hidden_dims=hidden_dims,num_layers=num_layers)
        self.loss = loss
        self.optim = optimizer
        self.lr = lr
        

    def training_step(self,batch,batch_idx):
        x,y = batch
        y_pred = self.lstm(x)
        loss = self.loss(y_pred,y-1)
        self.log_dict({'training_loss':loss},prog_bar=True)
        return loss

    def validation_step(self,batch,batch_idx):
        x,y = batch
        y_pred = self.lstm(x)
        loss = self.loss(y_pred,y-1)
        self.log_dict({'validation_loss':loss},prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.optim(self.lstm.parameters(),lr=self.lr)

if __name__ == "__main__":
    lstm = LSTM(input_dim=12,num_classes=9)

    trainer = pl.Trainer(gpus=1,max_epochs=50)
    train_dataset = s2_loader.Sentinel2Dataset("../utils/h5_folder/train_sentinel_ts.hdf5")
    train_dataloader = s2_loader.sentinel2_dataloader(train_dataset,256,8,True,True)
    val_dataset = s2_loader.Sentinel2Dataset("../utils/h5_folder/val_sentinel_ts.hdf5")
    val_dataloader = s2_loader.sentinel2_dataloader(val_dataset,256,8,True,False)
    trainer.fit(lstm,train_dataloader,val_dataloader)


