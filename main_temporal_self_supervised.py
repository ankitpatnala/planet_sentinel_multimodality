from self_supervised_models import BACKBONE_POINT,SELF_SUPERVISED_TYPE
from callbacks.callbacks import SelfSupervisedCallback
from utils import SELF_SUPERVISED_LOSS_FUNC
import argparse

from datasets import sentinel2_dataloader as s2_loader
from datasets.pretrain_dataloader import PretrainingDataset,pretrain_dataloader
from datasets.pretrain_time_dataloader import PretrainingTimeDataset,pretrain_time_dataloader
from datasets.bert_style_dataloader import BertDataset,bert_dataloader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import optuna
import wandb
import copy

from models.lstm import HyperParameterCallback

def objective(trial,self_supervised_model,args):
    #pretraining_time_dataset = PretrainingTimeDataset("../utils/h5_folder/pretraining_time2.h5")
    #pretraining_time_dataloader = pretrain_time_dataloader(pretraining_time_dataset,256,16,True,True,True)
    pretraining_time_dataset  = BertDataset("../utils/h5_folder/pretraining_time2.h5")
    pretraining_time_dataloader = bert_dataloader(pretraining_time_dataset,256,16,True,True,True)
    pl.trainer.seed_everything(32)

    trial_args = self_supervised_model.return_hyper_parameter_args()
    args = args.__dict__
    self_supervised_callback = SelfSupervisedCallback(**args)
    copied_args = copy.deepcopy(args)

    for arg in trial_args:
        if arg == 'lr':
            copied_args[arg] = trial.suggest_float(arg,args[arg][0],args[arg][1],log=True)
        if arg == 'dropout':
            copied_args[arg] = trial.suggest_uniform(arg,args[arg][1],args[arg][1])
        if not (arg == 'lr' or arg == 'dropout'):
            copied_args[arg] = trial.suggest_categorical(arg,args[arg])
    
    lightning_model = self_supervised_model(
                    36,
                    12,
                    **copied_args)
    wandb_logger = WandbLogger(project="planet_sentinel_multimodality_self_supervised_temporal",
                               config=copied_args)
    wandb_logger.watch(lightning_model)
    dir_path_name = f"{args['pretrain_type']}"
    if args['is_mixup'] :
        dir_path_name+="_mixup"
    if args['is_seasonal']:
        dir_path_name+="_seasonal"

    checkpoint_callback = ModelCheckpoint(dirpath = f"./{dir_path_name}/{args['loss_name']}/{args['temperature']}/",every_n_epochs=5,save_top_k=-1)
    trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=100,
            logger=wandb_logger,
            callbacks=[checkpoint_callback]
            )
    trainer.fit(lightning_model,pretraining_time_dataloader)

    wandb.finish()
    return lightning_model.downstream_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    self_supervised_model = SELF_SUPERVISED_TYPE['bert']
    parser.add_argument("--loss",type=str,default="simclr")
    parser.add_argument("--temperature",type=float,default=0.07)
    parser.add_argument("--baseline_hyper_param_file",type=str,default=None)
    parser.add_argument("--baseline_model_type",type=str,default='lstm')
    parser.add_argument("--pretrain_type",type=str,default='temporal_transformer')
    parser.add_argument("--hyperparameter_tuning",action='store_true')
    parser.add_argument("--hyperparameter_resume_file",type=str,default=None)
    self_supervised_model.add_model_specific_args(parser)

    args = parser.parse_args()
    args.loss_name = args.loss
    args.loss = SELF_SUPERVISED_LOSS_FUNC[args.loss]

    if args.hyperparameter_tuning:
        n_trials = 25
    else:
        n_trials = 1

    if args.hyperparameter_resume_file is None:
        study = optuna.create_study(direction='maximize')
    else :
        study = joblib.load(model_args.hyperparameter_resume_file)
    
    study.optimize(lambda trial: objective(
                        trial,
                        self_supervised_model,
                        args),
                        n_trials=n_trials,
                        callbacks=[HyperParameterCallback("../hyp_bert_syle_temp_self_supervised_{args.baseline_model_type}.pkl")])
