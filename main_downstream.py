
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


if __name__ == "__main__":

