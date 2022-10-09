from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import h5py

class PlanetDataset(Dataset):
    def __init__(self,file_path):
        self.file_path = file_path
        dataset = h5py.File(file_path)

    def __len__(self):
        return len(dataset['time_series'])


    def __getitem__(self,idx):
        return dataset['time_series'][idx], dataset['crop_labels'][idx]

def planet_dataloader(dataset,batch_size,num_workers,pin_memory,shuffle):
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory_device=pin_memory)
