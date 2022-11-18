from models.lstm import LSTM
from models.inception import InceptionTime
from models.transformer import Transformer
import argparse

from datasets import sentinel2_dataloader as s2_loader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import optuna
import wandb
import copy
import joblib

from models.lstm import HyperParameterCallback
MODELS = {'lstm':LSTM,
          'inception':InceptionTime,
          'transformer':Transformer}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",type=str)
    temp_args,_ = parser.parse_known_args()
    model = MODELS[temp_args.method]
    model.add_model_specific_args(parser)
    parser.add_argument("--input_dim",type=int,default=12)
    parser.add_argument("--num_classes",type=int,default=9)
    parser.add_argument("--project",type=str,default="planet_sentinel_baseline")
    parser.add_argument("--hyperparameter_tuning",action='store_true')
    parser.add_argument("--self_supervised_ckpt",type=str,default=None)
    parser.add_argument("--pretrain_type",type=str,default="resmlp")
    parser.add_argument("--baseline_hyper_param_file",type=str,default=None)
    parser.add_argument("--is_normalize",action='store_true')
    parser.add_argument("--trial_number",type=int,default=-1)
    args = parser.parse_args()
    pl.trainer.seed_everything(32)
    wandb_logger = WandbLogger(project=args.project,
                             config=args.__dict__)
    trainer = pl.Trainer.from_argparse_args(
            args,
            accelerator='gpu',
            devices=1,
            max_epochs=50,
            logger=wandb_logger)
    args = args.__dict__
    study = joblib.load(args['baseline_hyper_param_file'])
    trial_params = study.get_trials()[args['trial_number']].params
    for key in model.return_hyper_parameter_args():
        args[key] = trial_params[key]
    train_dataset = s2_loader.Sentinel2Dataset("../utils/h5_folder/train_sentinel_ts.hdf5",is_normalize=args['is_normalize'])
    train_dataloader = s2_loader.sentinel2_dataloader(train_dataset,256,8,True,True,True)
    val_dataset = s2_loader.Sentinel2Dataset("../utils/h5_folder/val_sentinel_ts.hdf5",is_normalize=args['is_normalize'])
    val_dataloader = s2_loader.sentinel2_dataloader(val_dataset,256,8,True,False,True)

    lightning_model = model(**args)

    trainer.fit(lightning_model,train_dataloader,val_dataloader)
