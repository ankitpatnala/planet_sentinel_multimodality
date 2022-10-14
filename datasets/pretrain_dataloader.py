from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np

import h5py


class PretrainingDataset(Dataset):
    def __init__(self,file_path):
        self.dataset = h5py.File(file_path)

    def __len__(self):
        return len(self.dataset['planet_data'])

    def __getitem__(self,idx):
        return ((self.dataset['sentinel2_data'][idx]/10000).astype(np.float32),
                (self.dataset['planet_data'][idx]/10000).flatten().astype(np.float32))


def pretrain_dataloader(dataset,batch_size,num_workers,pin_memory,shuffle):
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=pin_memory)
