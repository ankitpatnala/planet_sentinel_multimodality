from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np

import h5py
import random


class PretrainingDataset(Dataset):
    def __init__(self,file_path):
        self.dataset = h5py.File(file_path)

    def __len__(self):
        return len(self.dataset['planet_data'])

    def __getitem__(self,idx):
        return ((self.dataset['sentinel2_data'][idx]/10000).astype(np.float32),
                (self.dataset['planet_data'][idx]/10000).flatten().astype(np.float32))

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
