from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np

import h5py
import random
import pickle

from utils.prepare_data import time_stamp
from utils.prepare_data import get_day_idx_of_sentinel2


with open(time_stamp,'rb') as pickle_reader:
    sentinel2_time_stamp = pickle.load(pickle_reader)

day_idx = np.array(get_day_idx_of_sentinel2(sentinel2_time_stamp),dtype=np.int32)

class BertDataset(Dataset):
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
        planet_data = torch.Tensor((self.dataset['planet_data'][idx]/10000).astype(np.float32))
        sentinel2_data = torch.Tensor((self.dataset['sentinel2_data'][idx]/10000).astype(np.float32))
        number_of_samples_to_mask = int(0.15*(sentinel2_data.shape[0]))
        index = np.random.choice(sentinel2_data.shape[0],number_of_samples_to_mask)
        index_mask = index[:int(0.8*len(index))]
        index_random = index[int(0.8*len(index)):int(0.9*len(index))]
        index_non_replacement = index[int(0.9*len(index)):]
        sentinel2_data[index_mask] = torch.zeros_like(sentinel2_data[index_mask])
        #sentinel2_data[index_random] = torch.rand_like(sentinel2_data[index_random]).uniform_(0,1)
        sentinel2_data[index_non_replacement] = sentinel2_data[index_non_replacement] 
        random_data = (self.dataset['sentinel2_data'][torch.randint(0,len(self),(1,))]/10000).astype(np.float32)
        sentinel2_data[index_random] = torch.Tensor(random_data[index_random])

        if not self.is_normalize:
            return sentinel2_data,index,planet_data.reshape(365,-1)[day_idx[index]]
        else :
            normalize_sentinel = (sentinel2_data-self.sentinel_mean)/self.sentinel_var
            normalize_planet = ((planet_data.permute(0,2,1)-self.planet_mean)/self.planet_var).reshape(365,-1)
            return normalize_sentinel,index,normalize_planet[day_idx[index]]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

def bert_dataloader(dataset,batch_size,num_workers,pin_memory,shuffle,is_random_seed):
    return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker if is_random_seed else None,
            generator=g if is_random_seed else None)

if __name__ == "__main__":
    bert_dataset = BertDataset("../utils/h5_folder/pretraining_time2.h5",is_normalize=False)
    #bert_dataset[5]
    dataloader = bert_dataloader(bert_dataset,10,8,True,True,True)

    for idx,data in enumerate(dataloader):
        a,b,c = data
        print(a.shape)
        print(b.shape)
        print(c.shape)
        break

