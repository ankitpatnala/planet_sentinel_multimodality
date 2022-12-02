from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np

import h5py
import random
import pickle

class PretrainingDataset(Dataset):
    def __init__(self,file_path,is_normalize=True):
        self.dataset = h5py.File(file_path)
        self.is_normalize = is_normalize
        if is_normalize:
            with open("./mean_var_list.pkl","rb") as pickle_reader:
                data = pickle.load(pickle_reader)
                self.sentinel_mean = data['sentinel_mean']
                self.sentinel_var = data['sentinel_var']
                self.planet_mean = data['planet_mean']
                self.planet_var = data['planet_var'] 
    
    def __len__(self):
        return len(self.dataset['sentinel2_data'])

    def __getitem__(self,idx):
        if not self.is_normalize:
            return ((self.dataset['sentinel2_data'][idx]/10000).astype(np.float32),
                    (self.dataset['planet_data'][idx]/10000).flatten().astype(np.float32))
        else :
            normalize_sentinel = (
                    (torch.Tensor((self.dataset['sentinel2_data'][idx]/10000).astype(np.float32))-self.sentinel_mean)/self.sentinel_var)
            normalize_planet = ((
                    (torch.Tensor(self.dataset['planet_data'][idx]/10000).permute(1,0)-self.planet_mean)/self.planet_var).permute(
                            1,0).flatten())

            return normalize_sentinel,normalize_planet

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

def pretrain_dataloader(dataset,batch_size,num_workers,pin_memory,shuffle,is_random_seed):
    return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker if is_random_seed else None,
            generator=g if is_random_seed else None)

if __name__ == "__main__":
    pretraining_dataset = PretrainingDataset("../utils/h5_folder/pretraining_point2.h5",is_normalize=True)
    print(len(pretraining_dataset))
    x_sentinel, x_planet = pretraining_dataset[3]
    print(x_sentinel.shape,x_planet.shape)

