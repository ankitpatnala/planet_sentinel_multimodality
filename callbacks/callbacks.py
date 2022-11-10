import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from datasets import sentinel2_dataloader as s2_loader
#from models.lstm import LSTM
#from models.inception import InceptionTime
#from models.transformer import Transformer
import optuna

import os
import joblib

#MODELS = {'lstm':LSTM,
#          'inception':InceptionTime,
#          'transformer':Transformer}
#
class SelfSupervisedCallbackOld(Callback):
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
                    max_epochs=10,
                    logger=trainer.logger)
            callback_trainer.fit(lstm,self.train_dataloader,self.val_dataloader)

    def on_train_epoch_end(self,trainer,pl_module):
        if pl_module.current_epoch%50 == 0:
            self.ckpt_path = f"{trainer.logger._project}/{trainer.logger.version}/checkpoints/epoch={trainer.current_epoch}-step={trainer.global_step}.ckpt"


class SelfSupervisedCallback(Callback):
    def __init__(self,baseline_model_type='lstm',pretrain_type=None,baseline_hyper_param_file=None,**kwargs):
        super(SelfSupervisedCallback,self).__init__()
        self.train_dataset = s2_loader.Sentinel2Dataset("../utils/h5_folder/train_sentinel_ts.hdf5")
        self.train_dataloader = s2_loader.sentinel2_dataloader(self.train_dataset,256,8,True,True,True)
        self.val_dataset = s2_loader.Sentinel2Dataset("../utils/h5_folder/val_sentinel_ts.hdf5")
        self.val_dataloader = s2_loader.sentinel2_dataloader(self.val_dataset,256,8,True,False,True)
        self.ckpt_path = ""
        self.baseline_model_type = baseline_model_type
        self.load_from_hyperparameter = (baseline_hyper_param_file is not None)
        self.params = {}
        if baseline_hyper_param_file is not None:
            study = joblib.load(baseline_hyper_param_file)
            self.params = study.best_params
        self.params['pretrain_type'] = pretrain_type
        print("kwargs",kwargs)

    def on_train_epoch_start(self,trainer,pl_module):
        if (pl_module.current_epoch-1)%250==0 or pl_module.current_epoch==999:
            if self.load_from_hyperparameter:
                self.params['self_supervised_ckpt']=self.ckpt_path
                print(self.params)
                model = MODELS[self.baseline_model_type](12,9,**self.params,config=pl_module.config)
             
            else:
                model = MODELS[self.baseline_model_type](12,9,**self.params,self_supervised_ckpt=self.ckpt_path,config=pl_module.config)
            callback_trainer = pl.Trainer(
                    accelerator='gpu',
                    devices=1,
                    max_epochs=10)
            callback_trainer.fit(model,self.train_dataloader,self.val_dataloader)
            pl_module.downstream_accuracy = model.accuracy_score
            pl_module.log_dict({'downstram_accuracy':model.accuracy_score})

    def on_train_epoch_end(self,trainer,pl_module):
        if pl_module.current_epoch%250==0 or pl_module.current_epoch==998:
            self.ckpt_path = f"{trainer.logger._project}/{trainer.logger.version}/checkpoints/epoch={trainer.current_epoch}-step={trainer.global_step}.ckpt"

#class SelfSupervisedCallback(Callback):
#    def __init__(self,hyper_parameter_pickle=None,pretrain_type='resmlp'):
#        super(SelfSupervisedCallback,self).__init__()
#        self.train_dataset = s2_loader.Sentinel2Dataset("../utils/h5_folder/train_sentinel_ts.hdf5")
#        self.train_dataloader = s2_loader.sentinel2_dataloader(self.train_dataset,256,8,True,True,True)
#        self.val_dataset = s2_loader.Sentinel2Dataset("../utils/h5_folder/val_sentinel_ts.hdf5")
#        self.val_dataloader = s2_loader.sentinel2_dataloader(self.val_dataset,256,8,True,False,True)
#        self.ckpt_path = ""
#        self.load_from_hyperparameter = (hyper_parameter_pickle is not None)
#        if hyper_parameter_pickle is not None:
#            study = joblib.load(hyper_parameter_pickle)
#            self.params = study.best_params
#        self.params['backbone_type'] = pretrain_type
#
#    def on_train_epoch_start(self,trainer,pl_module):
#        if (pl_module.current_epoch-1)%250==0 or pl_module.current_epoch==999:
#            if self.load_from_hyperparameter:
#                self.params['self_supervised_ckpt']=self.ckpt_path
#                lstm = LSTM(12,9,**self.params,config=pl_module.config)
#            else:
#                lstm = LSTM(12,9,self_supervised_ckpt=self.ckpt_path)
#            callback_trainer = pl.Trainer(
#                    accelerator='gpu',
#                    devices=1,
#                    max_epochs=10)
#            callback_trainer.fit(lstm,self.train_dataloader,self.val_dataloader)
#            pl_module.downstream_accuracy = lstm.accuracy_score
#            pl_module.log_dict({'downstram_accuracy':lstm.accuracy_score})
#
#    def on_train_epoch_end(self,trainer,pl_module):
#        if pl_module.current_epoch%250==0 or pl_module.current_epoch==998:
#            self.ckpt_path = f"{trainer.logger._project}/{trainer.logger.version}/checkpoints/epoch={trainer.current_epoch}-step={trainer.global_step}.ckpt"

