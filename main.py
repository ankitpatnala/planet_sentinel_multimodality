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

from models.lstm import HyperParameterCallback

MODELS = {'lstm':LSTM,
          'inception':InceptionTime,
          'transformer':Transformer}
def objective(trial,model,model_args,args):
    train_dataset = s2_loader.Sentinel2Dataset("../utils/h5_folder/train_sentinel_ts.hdf5",is_normalize=model_args.is_normalize)
    train_dataloader = s2_loader.sentinel2_dataloader(train_dataset,256,8,True,True,True)
    val_dataset = s2_loader.Sentinel2Dataset("../utils/h5_folder/val_sentinel_ts.hdf5",is_normalize=model_args.is_normalize)
    val_dataloader = s2_loader.sentinel2_dataloader(val_dataset,256,8,True,False,True)

    pl.trainer.seed_everything(32)
    trial_args = model.return_hyper_parameter_args()
    model_args = model_args.__dict__

    copied_model_args = copy.deepcopy(model_args)
    
    for arg in trial_args:
        if arg == 'lr':
            copied_model_args[arg] = trial.suggest_float(arg,model_args[arg][0],model_args[arg][1])
        if arg == 'dropout':
            copied_model_args[arg] = trial.suggest_uniform(arg,model_args[arg][0],model_args[arg][1])
        if not (arg == 'lr' or arg == 'dropout'):
            copied_model_args[arg] = trial.suggest_categorical(arg,model_args[arg])

    wandb_logger = WandbLogger(project=copied_model_args['project'],
                             config=model_args)
    lightning_model = model(**copied_model_args)
    trainer = pl.Trainer.from_argparse_args(
            args,
            accelerator='gpu',
            devices=1,
            max_epochs=50,
            logger=wandb_logger)

    trainer.fit(lightning_model,train_dataloader,val_dataloader)
    wandb.finish()
    return lightning_model.accuracy_score,lightning_model.f1_score

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
    parser.add_argument("--backbone_type",type=str,default="resmlp")
    parser.add_argument("--hyperparameter_resume_file",type=str,default=None)
    parser.add_argument("--is_normalize",action='store_true')
    model_args,_ = parser.parse_known_args()
    pl.Trainer.add_argparse_args(parser)
    args = pl.Trainer.parse_argparser(parser.parse_args(""))
    args = parser.parse_args([])

    if model_args.hyperparameter_tuning:
        n_trials = 25
    else:
        n_trials = 1

    if model_args.hyperparameter_resume_file is None:
        study = optuna.create_study(directions=['maximize','maximize'])
    else :
        study = joblib.load(model_args.hyperparameter_resume_file)

    study.optimize(lambda trial: objective(
                        trial,
                        model,
                        model_args,
                        args),
                        n_trials=n_trials,
                        callbacks=[HyperParameterCallback(f"../hyp_tune_{temp_args.method}2.pkl")])














