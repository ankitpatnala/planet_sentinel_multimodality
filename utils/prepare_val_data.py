import numpy as np
import rasterio
import rasterio.features
import geopandas as gpd
import pandas as pd
from utils.prepare_data import read_boundries_from_pickle
from utils.prepare_data import split_train_and_val
from utils.prepare_data import save_geojson,save_indices,get_time_series
from utils.prepare_data import get_sentinel_time_series

import os
import pickle


sentinel2_val_folder = "/p/scratch/deepacf/kiste/patnala1/planet_sentinel_multimodality/dlr_fusion_competition_germany_test_source_sentinel_2/dlr_fusion_competition_germany_test_source_sentinel_2_33N_17E_243N_2019/"

bbox = "bbox.pkl" 
bands = "bands.npy"


val_gpd_file = "/p/scratch/deepacf/kiste/patnala1/planet_sentinel_multimodality/dlr_fusion_competition_germany_test_labels/dlr_fusion_competition_germany_test_labels_33N_17E_243N/labels.geojson"

bound_data = read_boundries_from_pickle(os.path.join(sentinel2_val_folder,bbox))
sentinel2_data = np.load(os.path.join(sentinel2_val_folder,bands))
_,width,height,_ = sentinel2_data.shape
class_dict = {'Forage Crops': 9,
              'Wheat': 1,
              'Meadows': 8,
              'Rye': 2,
              'Barley': 3,
              'Corn': 5,
              'Oil Seeds': 6,
              'Root Crops': 7,
              'Oats': 4}

sentinel_transform = rasterio.transform.from_bounds(
        bound_data.min_x,
        bound_data.min_y,
        bound_data.max_x,
        bound_data.max_y,
        width,
        height)

def create_pickle_files(dataframe,transform,num_pixels_per_class,mode):
    width_indices,height_indices,crop_labels = get_time_series(dataframe,transform,num_pixels_per_class=num_pixels_per_class)
    save_indices(width_indices,"width_index.pkl",mode=f'validation_{mode}')
    save_indices(height_indices,"height_index.pkl",mode=f'validation_{mode}')
    save_indices(crop_labels,"crop_labels.pkl",mode=f'validation_{mode}')


def split_and_save_crop_field_train_and_val(val_gpd_file):
    train_dataframe,val_dataframe = split_train_and_val(gpd.read_file(val_gpd_file))
    save_geojson(train_dataframe,mode='val_training')
    save_geojson(val_dataframe,mode='val_validation')
    create_pickle_files(train_dataframe,sentinel_transform,5000,'train')
    create_pickle_files(val_dataframe,sentinel_transform,1000,'val')

def validation_time_series(mode="train"):
    with open(os.path.join(f"validation_{mode}_indices_folder","width_index.pkl"),"rb") as pickle_reader:
        width_index = pickle.load(pickle_reader)

    with open(os.path.join(f"validation_{mode}_indices_folder","height_index.pkl"),"rb") as pickle_reader:
        height_index = pickle.load(pickle_reader)

    with open(os.path.join(f"validation_{mode}_indices_folder","crop_labels.pkl"),"rb") as pickle_reader:
        crop_labels = pickle.load(pickle_reader)

    get_sentinel_time_series(sentinel2_data,(width_index,height_index,crop_labels),mode=f"validation_{mode}")

if __name__ == "__main__":
    split_and_save_crop_field_train_and_val(val_gpd_file)
    validation_time_series()
    validation_time_series("val")





