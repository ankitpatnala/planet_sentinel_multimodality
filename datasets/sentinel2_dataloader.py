from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch

import h5py
import pickle
import numpy as np
import random

class Sentinel2Dataset(Dataset):
    def __init__(self,file_path,is_normalize=True):
        super(Sentinel2Dataset,self).__init__()
        self.file_path = file_path
        self.dataset = h5py.File(file_path)
        self.is_normalize = is_normalize
        if is_normalize :
            with open("./mean_var_list.pkl","rb") as pickle_reader:
                data = pickle.load(pickle_reader)
                self.sentinel_mean = data['sentinel_mean']
                self.sentinel_var = data['sentinel_var']

    def __len__(self):
        return len(self.dataset['time_series'])


    def __getitem__(self,idx):
        if not self.is_normalize:
            return (self.dataset['time_series'][idx]/10000).astype(np.float32), np.squeeze(self.dataset['crop_labels'][idx])
        else :
            return ((torch.Tensor((self.dataset['time_series'][idx]/10000).astype(np.float32))-self.sentinel_mean)/self.sentinel_var, np.squeeze(self.dataset['crop_labels'][idx]))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

def sentinel2_dataloader(dataset,batch_size,num_workers,pin_memory,shuffle,is_random_seed):
    return DataLoader(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker if is_random_seed else None,
            generator=g if is_random_seed else None)

if __name__ == "__main__":
    sentinel2_dataset = Sentinel2Dataset("/p/project/deepacf/kiste/patnala1/planet_sentinel_multimodality/utils/h5_folder/validation_train_sentinel_ts.hdf5")
    print(sentinel2_dataset.dataset.keys())
            

