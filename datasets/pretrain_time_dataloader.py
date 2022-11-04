from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch

import h5py
import pickle

class PretrainingTimeDataset(Dataset):
    def __init__(self,file_path):
        self.dataset = h5py.File(file_path)

    def __len__(self):
        return len(self.dataset['planet_data'])

    def __getitem__(self,idx):
        return ((self.dataset['sentinel2_data'][idx]/10000).astype(np.float32),
                (np.reshape(self.dataset['planet_data'][idx]/10000,(73,-1)).astype(np.float32)))


class PretrainingTimeDataset(Dataset):
    def __init__(self,file_path,is_normalize=True):
        self.dataset = h5py.File(file_path)
        self.is_normalize = is_normalize
        if is_normalize :
            with open("./mean_var_list.pkl","rb") as pickle_reader:
                data = pickle.load(pickle_reader)
                self.sentinel_mean = data['sentinel_mean']
                self.sentinel_var = data['sentinel_var']
                self.planet_mean = data['planet_mean']
                self.planet_var = data['planet_var'] 

    def __len__(self):
        return len(self.dataset['planet_data'])

    def __getitem__(self,idx):
        if not self.is_normalize:
            return ((self.dataset['sentinel2_data'][idx]/10000).astype(np.float32),
                    ((self.dataset['planet_data'][idx]/10000).astype(np.float32)))
        else :
            normalize_sentinel = (
                    (torch.Tensor((self.dataset['sentinel2_data'][idx]/10000).astype(np.float32))-self.sentinel_mean)/self.sentinel_var)
            normalize_planet = ((
                    (torch.Tensor(self.dataset['planet_data'][idx]/10000).permute(0,2,3,1)-self.planet_mean)/self.planet_var).permute(
                            0,3,1,2).reshape(73,-1))

            return normalize_sentinel,normalize_planet


def pretrain_time_dataloader(dataset,batch_size,num_workers,pin_memory,shuffle):
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=pin_memory)


if __name__ == "__main__":
    sentinel_mean = 0
    sentinel_std = 0
    planet_mean = 0
    planet_std = 0
    pretraining_time_dataset = PretrainingTimeDataset2("../utils/h5_folder/pretraining_time.h5")
    pretraining_data_loader = pretrain_time_dataloader(pretraining_time_dataset,3000,8,True,False)

    for idx,data in enumerate(pretraining_data_loader):
        sentinel_data,planet_data = data
        sentinel_var,sentinel_mean = torch.var_mean(sentinel_data,dim=(0,1))
        planet_var,planet_mean = torch.var_mean(planet_data,dim=(0,1,3,4))

    print(sentinel_mean,planet_mean)
    print(sentinel_var,planet_var)
    mean_var_values = {'sentinel_mean' : sentinel_mean,
                       'planet_mean' : planet_mean,
                       'sentinel_var' : sentinel_var,
                       'planet_var' : planet_var}
    with open("mean_var_list.pkl",'wb') as pickle_writer:
        pickle.dump(mean_var_values,pickle_writer)


    
