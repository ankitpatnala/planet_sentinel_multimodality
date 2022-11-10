from self_supervised_models import SELF_SUPERVISED_TYPE
from callbacks.callbacks import SelfSupervisedCallback
from utils import SELF_SUPERVISED_LOSS_FUNC
import argparse

from datasets import sentinel2_dataloader as s2_loader
from datasets.pretrain_dataloader import PretrainingDataset,pretrain_dataloader
from datasets.pretrain_time_dataloader import PretrainingTimeDataset,pretrain_time_dataloader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import optuna
import wandb
import copy

from models.lstm import HyperParameterCallback

def objective(trial,self_supervised_model,args):

    pretraining_dataset = PretrainingDataset("../../planet_sentinel_multimodality/utils/h5_folder/pretraining_point.h5")
    pretraining_dataloader = pretrain_dataloader(pretraining_dataset,2048,32,True,True,True)
    pl.trainer.seed_everything(32)
    wandb_logger = WandbLogger(project="planet_sentinel_multimodality_self_supervised_point",
                               config=args.__dict__)
    #self_supervised_callback = SelfSupervisedCallback(baseline_hyper_parameter_file=args.baseline_hyper_param_file,pretrain_type=args.pretrain_type)
    args = args.__dict__
    print(args)
    self_supervised_callback = SelfSupervisedCallback(**args)
    trial_args = self_supervised_model.return_hyper_parameter_args()

    copied_args = copy.deepcopy(args)

    for arg in trial_args:
        if arg == 'lr':
            copied_args[arg] = trial.suggest_float(arg,args[arg][0],args[arg][1])
        if arg == 'dropout':
            copied_args[arg] = trial.suggest_uniform(arg,args[arg][1],args[arg][1])
        if not (arg == 'lr' or arg == 'dropout'):
            copied_args[arg] = trial.suggest_categorical(arg,args[arg])

    lightning_model = self_supervised_model(
                    36,
                    12,
                    **copied_args)
        
    trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=1000,
            logger=wandb_logger,
            callbacks=[self_supervised_callback])
    trainer.fit(lightning_model,pretraining_dataloader)

    wandb.finish()
    return lightning_model.downstream_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    self_supervised_model = SELF_SUPERVISED_TYPE['point']
    parser.add_argument("--loss",type=str,default="simclr")
    parser.add_argument("--temperature",type=float,default=0.07)
    parser.add_argument("--baseline_hyper_param_file",type=str,default=None)
    parser.add_argument("--baseline_model_type",type=str,default='lstm')
    parser.add_argument("--pretrain_type",type=str,default='resmlp')
    
    self_supervised_model.add_model_specific_args(parser)
    parser.add_argument("--hyperparameter_tuning",action='store_true')
    
    args = parser.parse_args()

    args.loss = SELF_SUPERVISED_LOSS_FUNC[args.loss]
  
    if args.hyperparameter_tuning:
        n_trials = 25
    else:
        n_trials = 1

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(
                        trial,
                        self_supervised_model,
                        args),
                        n_trials=n_trials,
                        callbacks=[HyperParameterCallback(f"../hyp_point_self_supervised_{args.pretrain_type}.pkl")])
  
  
  
  
  
  
                        


    







   



    




