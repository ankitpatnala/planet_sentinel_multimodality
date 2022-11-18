import torch
from torch import nn
import numpy as np

import math
import pickle
import datetime 
import pytz

from self_supervised_models.backbones import MLP
from self_supervised_models.vit_for_time_series import EncoderTransformer

with open("../utils/h5_folder/time_stamp_sentinel_list.pkl",'rb') as pickle_reader:
    sentinel_days = (pickle.load(pickle_reader))

with open("../utils/h5_folder/time_stamp_planet_list.pkl",'rb') as pickle_reader:
    planet_days = [i for i in range(365)]  #(pickle.load(pickle_reader))*5

def chunk_4_seasons(time_list):
    first_date = (datetime.datetime(2018,3,31,0,0,0,tzinfo=pytz.UTC) - datetime.datetime(2018,1,1,0,0,0,tzinfo=pytz.UTC)).days
    second_date = (datetime.datetime(2018,6,30,0,0,0,tzinfo=pytz.UTC) - datetime.datetime(2018,1,1,0,0,0,tzinfo=pytz.UTC)).days
    third_date = (datetime.datetime(2018,9,30,0,0,0,tzinfo=pytz.UTC) - datetime.datetime(2018,1,1,0,0,0,tzinfo=pytz.UTC)).days
    fourth_date = (datetime.datetime(2018,12,31,0,0,0,tzinfo=pytz.UTC) - datetime.datetime(2018,1,1,0,0,0,tzinfo=pytz.UTC)).days   
    time_index_list= [first_date,second_date,third_date,fourth_date]
    return np.searchsorted(time_list,time_index_list)
    

class TransformerEncoder(nn.Module):
    def __init__(self,num_inputs,d_model,n_head,num_layer,mlp_dim,dropout,projector_layer,mode_type='sentinel',activation=nn.GELU(),**kwargs):
        super(TransformerEncoder,self).__init__()
        self.num_days =  len(sentinel_days) if mode_type == "sentinel" else len(planet_days) 
        self.encoder_layer = EncoderTransformer(input_size=self.num_days,in_chans=num_inputs,num_classes=mlp_dim,embed_dim=d_model,depth=num_layer,num_heads=n_head,global_pool="avg")
        self.projector = (nn.Linear(mlp_dim,mlp_dim)
                           if projector_layer == 0 
                           else MLP(mlp_dim,projector_layer,mlp_dim))
        self.season_classifier = nn.Linear(d_model,4)
        self.chunk_4_seasons = chunk_4_seasons(sentinel_days if mode_type == "sentinel" else planet_days)

    def forward(self,x):
        x = self.encoder_layer(x)
        x_proj = self.projector(x)
        return x,x_proj

    def return_embeddings(self,x):
        return self.encoder_layer.forward_features(x)[:,1:,:]

    def return_chunk_embeddings(self,x):
        representation = self.encoder_layer.forward_features(x)[:,1:,:]
        return (self.season_classifier(torch.mean(representation[:,0:self.chunk_4_seasons[0],:],dim=1)),
                self.season_classifier(torch.mean(representation[:,self.chunk_4_seasons[0]:self.chunk_4_seasons[1],:],dim=1)),
                self.season_classifier(torch.mean(representation[:,self.chunk_4_seasons[1]:self.chunk_4_seasons[2],:],dim=1)),
                self.season_classifier(torch.mean(representation[:,self.chunk_4_seasons[2]:self.chunk_4_seasons[3],:],dim=1)))


        

