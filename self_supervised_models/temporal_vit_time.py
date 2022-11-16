import torch
from torch import nn

import math
import pickle

from self_supervised_models.backbones import MLP
from self_supervised_models.vit_for_time_series import EncoderTransformer

with open("../utils/h5_folder/time_stamp_sentinel_list.pkl",'rb') as pickle_reader:
    sentinel_days = (pickle.load(pickle_reader))

with open("../utils/h5_folder/time_stamp_planet_list.pkl",'rb') as pickle_reader:
    planet_days = (pickle.load(pickle_reader))*5

class TransformerEncoder(nn.Module):
    def __init__(self,num_inputs,d_model,n_head,num_layer,mlp_dim,dropout,projector_layer,mode_type='sentinel',activation=nn.GELU(),**kwargs):
        super(TransformerEncoder,self).__init__()
        self.num_days =  len(sentinel_days) if mode_type == "sentinel" else len(planet_days) 
        self.encoder_layer = EncoderTransformer(input_size=self.num_days,in_chans=num_inputs,num_classes=mlp_dim,embed_dim=d_model,depth=num_layer,num_heads=n_head,global_pool="avg")
        self.projector = (nn.Linear(mlp_dim,mlp_dim)
                           if projector_layer == 0 
                           else MLP(mlp_dim,projector_layer,mlp_dim))

    def forward(self,x):
        x = self.encoder_layer(x)
        x_proj = self.projector(x)
        return x,x_proj

    def return_embeddings(self,x):
        return self.encoder_layer.forward_features(x)[:,1:,:]
        

