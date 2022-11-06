import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from datasets import sentinel2_dataloader as s2_loader
from models.lstm import LSTM

import os

class SelfSupervisedCallback(Callback):
    def __init__(self):
        super(SelfSupervisedCallback,self).__init__()
        self.train_dataset = s2_loader.Sentinel2Dataset("../utils/h5_folder/train_sentinel_ts.hdf5")
        self.train_dataloader = s2_loader.sentinel2_dataloader(self.train_dataset,256,8,True,True)
        self.val_dataset = s2_loader.Sentinel2Dataset("../utils/h5_folder/val_sentinel_ts.hdf5")
        self.val_dataloader = s2_loader.sentinel2_dataloader(self.val_dataset,256,8,True,False)
        self.ckpt_path = ""

    def on_train_epoch_start(self,trainer,pl_module):
        if (pl_module.current_epoch-1)%50==0:
            lstm = LSTM(12,9,self_supervised_ckpt=self.ckpt_path)
            callback_trainer = pl.Trainer(
                    accelerator='gpu',
                    devices=1,
                    max_epochs=10)
            callback_trainer.fit(lstm,self.train_dataloader,self.val_dataloader)

    def on_train_epoch_end(self,trainer,pl_module):
        if pl_module.current_epoch%50 == 0:
            self.ckpt_path = f"{trainer.logger._project}/{trainer.logger.version}/checkpoints/epoch={trainer.current_epoch}-step={trainer.global_step}.ckpt"


class SelfSupervisedTransformerCallback(Callback):
    def __init__(self):
        super(SelfSupervisedTransformerCallback,self).__init__()
        self.train_dataset = s2_loader.Sentinel2Dataset("../utils/h5_folder/train_sentinel_ts.hdf5")
        self.train_dataloader = s2_loader.sentinel2_dataloader(self.train_dataset,256,8,True,True)
        self.val_dataset = s2_loader.Sentinel2Dataset("../utils/h5_folder/val_sentinel_ts.hdf5")
        self.val_dataloader = s2_loader.sentinel2_dataloader(self.val_dataset,256,8,True,False)
        self.ckpt_path = ""

    def on_train_epoch_start(self,trainer,pl_module):
        if (pl_module.current_epoch-1)%250==0 or pl_module.current_epoch==999:
            lstm = LSTM(12,9,self_supervised_ckpt=self.ckpt_path,config=pl_module.kwargs)
            callback_trainer = pl.Trainer(
                    accelerator='gpu',
                    devices=1,
                    max_epochs=10)
            callback_trainer.fit(lstm,self.train_dataloader,self.val_dataloader)
            pl_module.downstream_accuracy = lstm.accuracy_score

    def on_train_epoch_end(self,trainer,pl_module):
        if pl_module.current_epoch%250==0 or pl_module.current_epoch==998:
            self.ckpt_path = f"{trainer.logger._project}/{trainer.logger.version}/checkpoints/epoch={trainer.current_epoch}-step={trainer.global_step}.ckpt"

