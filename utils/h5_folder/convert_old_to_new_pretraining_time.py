import h5py
import numpy as np


if __name__ == "__main__":
     h5_file = h5py.File("./pretraining_time2_old.h5","r")
     h5_file_new = h5py.File("./pretraining_time2.h5",'w')

     sentinel2_data = h5_file['sentinel2_data'][:]
     planet_data = h5_file['planet_data'][:].reshape(365,4,150000,9).transpose(2,0,1,3)

     print(sentinel2_data.shape,planet_data.shape)

     h5_file_new.create_dataset('sentinel2_data',shape=sentinel2_data.shape,data=sentinel2_data)
     h5_file_new.create_dataset('planet_data',shape=planet_data.shape,data=planet_data)
     h5_file.close()
     h5_file_new.close()
