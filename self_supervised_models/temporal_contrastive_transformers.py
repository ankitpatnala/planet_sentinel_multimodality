import torch
import pytorch_lightning as pl

from utils.prepare_data import get_day_idx_of_sentinel2
from utils.prepare_data import time_stamp 

import pickle

def pos_embedding(input_dim,days):
    doy_array = torch.unsqueeze(torch.arange(0,365),dim=0)
    div_array = torch.pow(10000,2*torch.arange(input_dim)/input_dim)
    pos_array = torch.einsum('ij,k->jk',doy_array,1/div_array)
    pos_array[0::2,:] = torch.sin(pos_array[0::2,:])
    pos_array[1::2,:] = torch.cos(pos_array[1::2,:])
    return pos_array[days,:]




if __name__ == "__main__":
    with open(time_stamp,'rb') as pickle_reader:
        sentinel_days = get_day_idx_of_sentinel2(pickle.load(pickle_reader))

    print(pos_embedding(12,setinel_days).shape)


    

    

    


