from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import h5py
import numpy as np

class Sentinel2Dataset(Dataset):
    def __init__(self,file_path):
        super(Sentinel2Dataset,self).__init__()
        self.file_path = file_path
        self.dataset = h5py.File(file_path)

    def __len__(self):
        return len(self.dataset['time_series'])


    def __getitem__(self,idx):
        return (self.dataset['time_series'][idx]/10000).astype(np.float32), np.squeeze(self.dataset['crop_labels'][idx])


def sentinel2_dataloader(dataset,batch_size,num_workers,pin_memory,shuffle):
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=pin_memory)
