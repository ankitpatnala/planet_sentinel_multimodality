import torch
from torch import nn
import pytorch_lightning as pl

import pickle

from self_supervised_models.backbones import MLP

with open("../utils/h5_folder/time_stamp_sentinel_list.pkl",'rb') as pickle_reader:
    sentinel_days = (pickle.load(pickle_reader))

with open("../utils/h5_folder/time_stamp_planet_list.pkl",'rb') as pickle_reader:
    planet_days = (pickle.load(pickle_reader))
    
    
def pos_embedding(input_dim,days):
    doy_array = torch.unsqueeze(torch.arange(0,365),dim=0)
    div_array = torch.pow(10000,2*torch.arange(input_dim)/input_dim)
    pos_array = torch.einsum('ij,k->jk',doy_array,1/div_array)
    pos_array[0::2,:] = torch.sin(pos_array[0::2,:])
    pos_array[1::2,:] = torch.cos(pos_array[1::2,:])
    return pos_array[days,:]


class TransformerEncoder(nn.Module):
    def __init__(self,num_inputs,d_model,n_head,num_layer,mlp_dim,dropout,mode_type='sentinel'):
        super(TransformerEncoder,self).__init__()
        self.linear = nn.Linear(num_inputs,d_model)
        self.class_token = nn.Parameter(torch.randn(1,1,d_model))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model,n_head,dim_feedforward=mlp_dim,dropout=dropout)
        self.embedding = nn.Linear(d_model,mlp_dim)
        self.projector = MLP(mlp_dim,2,mlp_dim)
        self.pos_embedding = pos_embedding(d_model,sentinel_days if mode_type=="sentinel" else planet_days)

    def forward(self,x):
        n,t,c = x.shape
        repeat_class_token = torch.repeat_interleave(self.class_token,n,dim=0)
        x = self.linear(x.reshape(-1,c)).reshape(n,t,-1)
        x  = x + self.pos_embedding
        x = torch.cat([repeat_class_token,x],dim=1)
        x = self.encoder_layer(x)
        x_class_token = x[:,0,:]
        x_embedding = self.embedding(x_class_token)
        x_projector = self.projector(x_embedding)
        return x_embedding,x_projector

if __name__ == "__main__":
    transformer_encoder = TransformerEncoder(12,64,8,4,256,0.2,mode_type='sentinel')
    random_sentinel_array = torch.randn((32,144,12))
    emb,proj = transformer_encoder(random_sentinel_array)
    print(emb.shape,proj.shape)


    

    

    


