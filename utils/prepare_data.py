import rasterio
import rasterio.features
import numpy as np
import geopandas as gpd
import pandas as pd

import os
import pickle
import h5py

import datetime
import pytz
import matplotlib.pyplot as plt

sentinel2_folder = "/p/scratch/deepacf/kiste/patnala1/planet_sentinel_multimodality/dlr_fusion_competition_germany_train_source_sentinel_2/dlr_fusion_competition_germany_train_source_sentinel_2_33N_18E_242N_2018"
bounding_box = os.path.join(sentinel2_folder,"bbox.pkl")
clip = os.path.join(sentinel2_folder,"clp.npy")
sentinel2_numpy = os.path.join(sentinel2_folder,"bands.npy")
time_stamp = os.path.join(sentinel2_folder,"timestamp.pkl")

planet_folder = "/p/scratch/deepacf/kiste/patnala1/planet_sentinel_multimodality/dlr_fusion_competition_germany_train_source_planet/"
planet_day_folder_prefix = f"{planet_folder}dlr_fusion_competition_germany_train_source_planet_33N_18E_242N_"
crop_data_file = "/p/scratch/deepacf/kiste/patnala1/planet_sentinel_multimodality/dlr_fusion_competition_germany_train_labels/dlr_fusion_competition_germany_train_labels_33N_18E_242N/labels.geojson"

_,width,height,_ = 144,2400,2400,12

class_dict = {'Forage Crops': 9,
              'Wheat': 1,
              'Meadows': 8,
              'Rye': 2,
              'Barley': 3,
              'Corn': 5,
              'Oil Seeds': 6,
              'Root Crops': 7,
              'Oats': 4}



def read_sentinel2_day_wise(date,sentinel2_array):
    idx_of_sentinel2 = time_stamp_idx(date)
    sentinel_array = sentinel2_array[idx_of_sentinel2]
    return sentinel_array
    
def read_planet_day_wise(date):
    planet_file_name = os.path.join(planet_day_folder_prefix+date,"sr.tif")
    return rasterio.open(planet_file_name)

def return_crop_geometry(geojson_file):
    crop_data = gpd.read_file(geojson_file)
    return crop_data

def split_crops_for_pretrain_and_transfer(geo_data_frame,split_percentage=(0.7)):
    idx_array = np.arange(len(geo_data_frame))
    np.random.shuffle(idx_array)
    pre_training_indices = idx_array[0:int(split_percentage*len(idx_array))]
    remaining_indices = idx_array[int(split_percentage*len(idx_array)):]
    return (geo_data_frame.iloc[pre_training_indices],
            geo_data_frame.iloc[remaining_indices])

def split_train_and_val(remaining_data,split_percentage=0.7):
    crop_frame_train = []
    crop_frame_val = []
    for crop in remaining_data['crop_name'].unique():
        crop_wise = remaining_data[remaining_data['crop_name'] == crop]
        crop_frame_train.append(crop_wise.iloc[0:int(split_percentage*len(crop_wise))])
        crop_frame_val.append(crop_wise.iloc[int(split_percentage*len(crop_wise)):])
    return (pd.concat(crop_frame_train),pd.concat(crop_frame_val))

def get_time_series(dataframe,transform,mode='train',num_pixels_per_class=5000,all_touched=False):
    sentinel_indices_width = []
    sentinel_indices_height = []
    crop_labels = []
    for crop in dataframe['crop_name'].unique():
        crop_field_geoms = list(dataframe[dataframe['crop_name']==crop]['geometry'])
        mask_array = rasterio.features.rasterize(crop_field_geoms,out_shape=(width,height),transform=transform,all_touched=all_touched)
        field_mask = np.where(mask_array==1)
        idx = np.random.choice(np.arange(len(field_mask[0])),num_pixels_per_class)
        sentinel_indices_width.extend(field_mask[0][idx])
        sentinel_indices_height.extend(field_mask[1][idx])
        crop_labels.extend([class_dict[crop]]*num_pixels_per_class)
    return (sentinel_indices_width,
            sentinel_indices_height,
            crop_labels)

def convert_sentinel_to_planetscope(idx):
    return int(idx*10/3)

def get_sentinel_time_series(sentinel_array,idx_crop_list,output_file_name="sentinel_ts.hdf5",mode="train"):
    h5_folder = os.path.join("./h5_folder")
    if not os.path.isdir(h5_folder):
        os.makedirs(h5_folder)
    sentinel_array = np.transpose(sentinel_array,[0,3,1,2])
    t,c,w,h = sentinel_array.shape
    sentinel_array = sentinel_array[:,:,idx_crop_list[0],idx_crop_list[1]]
    sentinel_array = np.transpose(sentinel_array,axes=[2,0,1])
    crop_labels = np.expand_dims(np.array(idx_crop_list[2]),axis=-1)
    file_name = f"{mode}_{output_file_name}"
    file_path = os.path.join(h5_folder,file_name)
    h5_file = h5py.File(file_path,'w')
    h5_file.create_dataset('time_series',data=sentinel_array,shape=sentinel_array.shape)
    h5_file.create_dataset('crop_labels',shape=(sentinel_array.shape[0],1),data=crop_labels)
    h5_file.close()

def get_planet_time_series(planet_folder_name,idx_crop_list,output_file_name='planet_ts.hdf5',mode='train',neighbor_count=1):
    h5_folder = os.path.join("./h5_folder")
    if not os.path.isdir(h5_folder):
        os.makedirs(h5_folder)
    planet_array = read_planet_day_wise(planet_folder_name).read()
    c,w,h = planet_array.shape
    width_indices = np.array(list(map(convert_sentinel_to_planetscope,idx_crop_list[0])))
    height_indices = np.array(list(map(convert_sentinel_to_planetscope,idx_crop_list[1])))
    width_indices_list = []
    height_indices_list = []
    for i in range(-neighbor_count,neighbor_count+1):
        width_indices_list.append(np.expand_dims(width_indices+i,axis=-1))
        height_indices_list.append(np.expand_dims(height_indices+i,axis=-1))
    width_indices = np.repeat(
            np.concatenate(width_indices_list,axis=-1),
            (2*neighbor_count+1),
            axis=1).flatten()
    height_indices = np.repeat(
            np.concatenate(height_indices_list,axis=-1),
            3,
            axis=0).flatten()
    planet_array = planet_array[:,width_indices,height_indices].reshape(
            -1,1,c,2*neighbor_count+1,2*neighbor_count+1)
    return planet_array

def read_boundries_from_pickle(pickle_file):
    with open(pickle_file,'rb') as pickle_reader:
        bound_data = pickle.load(pickle_reader)
    return bound_data

def save_geojson(dataframe,mode='pretaining'):
    if not os.path.isdir("geojson"):
        os.makedirs("geojson")
    file_name = os.path.join("geojson",f"{mode}.geojson") 
    if not os.path.isfile(file_name):
        dataframe.to_file(file_name,driver='GeoJSON')
        print(f"{mode} file added")
    else :
        print(f"Already exisiting {mode} data")

def collect_pairs(sentinel_image,planet_image,crop_geometry):
    pass

def save_indices(indices,output_file_name,mode="training"):
    folder_path = f"{mode}_indices_folder"
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path,output_file_name)
    if not os.path.isfile(file_path):
        with open(file_path,'wb') as pickle_writer:
            pickle.dump(indices,pickle_writer)
    else :
        print(f"{file_path} already exists")

bound_data = read_boundries_from_pickle(bounding_box)
sentinel_transform = rasterio.transform.from_bounds(
        bound_data.min_x,
        bound_data.min_y,
        bound_data.max_x,
        bound_data.max_y,
        width,
        height)

class TimeseriesDataset :
    def __init__(self,geojson_folder=None,train_indices_folder=None,val_indices_folder=None):
        
        if geojson_folder is not None: 
            self.geojson_folder = geojson_folder
        else :
            crop_data = return_crop_geometry(crop_data_file)
            pretraining_data,remaining_data = split_crops_for_pretrain_and_transfer(crop_data)
            train_dataframe,val_dataframe = split_train_and_val(remaining_data)
            save_geojson(pretraining_data)
            save_geojson(train_dataframe,mode='training')
            save_geojson(val_dataframe,mode="val")
            self.geojson_folder = "./geojson"

        if train_indices_folder is not None :
            self.train_indices_folder = train_indices_folder
        else :
            train_dataframe = gpd.read_file("./geojson/training.geojson")
            width_indices,height_indices,crop_labels = get_time_series(train_dataframe,sentinel_transform)
            save_indices(width_indices,"width_index.pkl")
            save_indices(height_indices,"height_index.pkl")
            save_indices(crop_labels,"crop_labels.pkl")
            self.train_indices_folder = "./training_indices_folder"

        if val_indices_folder is not None :
            self.val_indices_folder = val_indices_folder
        else :
            val_dataframe = gpd.read_file("./geojson/val.geojson")
            width_indices_val,height_indices_val,crop_labels_val = get_time_series(val_dataframe,sentinel_transform,num_pixels_per_class=1000)
            save_indices(width_indices_val,"width_index.pkl",mode="validation")
            save_indices(height_indices_val,"height_index.pkl",mode="validation")
            save_indices(crop_labels_val,"crop_labels.pkl",mode="validation")
            self.val_indices_folder = "./validation_indices_folder"

        def return_train_data(self):
            pass

        def return_val_data(self):
            pass

        def get_data(self,mode=['train']):
            if 'train' in mode :
                self.return_train_data()

            if 'val' in mode:
                self.return_val_data()

class SentinelTimeSeries(TimeseriesDataset) :
    def __init__(self,geojson_folder=None,train_indices_folder=None,val_indices_folder=None):
        super(SentinelTimeSeries,self).__init__(geojson_folder,train_indices_folder,val_indices_folder)

    def return_train_data(self):
        sentinel_array = np.load(sentinel2_numpy)
        with open(os.path.join(self.train_indices_folder,"width_index.pkl"),"rb") as pickle_reader:
            width_index = pickle.load(pickle_reader)
        with open(os.path.join(self.train_indices_folder,"height_index.pkl"),"rb") as pickle_reader:
            height_index = pickle.load(pickle_reader)
        with open(os.path.join(self.train_indices_folder,"crop_labels.pkl"),"rb") as pickle_reader:
            crop_labels = pickle.load(pickle_reader)
        get_sentinel_time_series(sentinel_array,(width_index,height_index,crop_labels),mode="train")

    def return_val_data(self):
        sentinel_array = np.load(sentinel2_numpy)
        with open(os.path.join(self.val_indices_folder,"width_index.pkl"),"rb") as pickle_reader:
            width_index_val = pickle.load(pickle_reader)
        with open(os.path.join(self.val_indices_folder,"height_index.pkl"),"rb") as pickle_reader:
            height_index_val = pickle.load(pickle_reader)
        with open(os.path.join(self.val_indices_folder,"crop_labels.pkl"),"rb") as pickle_reader:
            crop_labels_val = pickle.load(pickle_reader)
        get_sentinel_time_series(sentinel_array,(width_index_val,height_index_val,crop_labels_val),mode="val")


class PlanetTimeSeries(TimeseriesDataset):
    def __init__(self,geojson_folder=None,planet_folder=None,train_indices_folder=None,val_indices_folder=None,neighbor_count=1):
        super(PlanetTimeSeries,self).__init__(geojson_folder,train_indices_folder,val_indices_folder)
        self.planet_folder = planet_folder
        self.neighbor_count = neighbor_count

    @staticmethod
    def sorted_planet(planet_folder):
        planet_dates = [folder[-10:] 
                for folder in os.listdir(planet_folder) 
                if os.path.isdir(os.path.join(planet_folder,folder))]
        planet_time_list = list(map(
            lambda x : (
                (datetime.datetime(int(x[0:4]),int(x[5:7]),int(x[8:10])) - 
                    datetime.datetime(2018,1,1)).days,
                         x)
                     ,planet_dates))
        planet_time_list = sorted(
                planet_time_list,
                key=lambda x : x[0])
        return planet_time_list

        
    def return_train_data(self,output_file_name="train_planet_ts.hdf5"):
        with open(os.path.join(self.train_indices_folder,"width_index.pkl"),"rb") as pickle_reader:
            width_index = pickle.load(pickle_reader)
        with open(os.path.join(self.train_indices_folder,"height_index.pkl"),"rb") as pickle_reader:
            height_index = pickle.load(pickle_reader)
        with open(os.path.join(self.train_indices_folder,"crop_labels.pkl"),"rb") as pickle_reader:
            crop_labels = pickle.load(pickle_reader)
        crop_labels = np.expand_dims(np.array(crop_labels),axis=-1)
        planet_folders_list = self.sorted_planet(self.planet_folder)

        if len(planet_folders_list) > 0 :
            for file_idx,planet_folder in enumerate(planet_folders_list):
                if file_idx == 0 :
                    time_series_array = get_planet_time_series(
                            planet_folder[1],
                            (width_index,height_index,crop_labels),
                            mode='train',
                            neighbor_count=self.neighbor_count)
                else:
                    time_series_array = np.concatenate(
                            [time_series_array,
                                get_planet_time_series(
                                    planet_folder[1],
                                    (width_index,height_index,crop_labels),
                                mode='train',
                                neighbor_count=self.neighbor_count)],axis=1)
        h5_file = h5py.File(os.path.join("./h5_folder",output_file_name),mode='w')
        h5_file.create_dataset('time_series',data=time_series_array,shape=time_series_array.shape)
        h5_file.create_dataset('crop_labels',shape=(time_series_array.shape[0],1),data=crop_labels)
        h5_file.close()
    
    def return_val_data(self,output_file_name="val_planet_ts.hdf5"):
        with open(os.path.join(self.val_indices_folder,"width_index.pkl"),"rb") as pickle_reader:
            width_index = pickle.load(pickle_reader)
        with open(os.path.join(self.val_indices_folder,"height_index.pkl"),"rb") as pickle_reader:
            height_index = pickle.load(pickle_reader)
        with open(os.path.join(self.val_indices_folder,"crop_labels.pkl"),"rb") as pickle_reader:
            crop_labels = pickle.load(pickle_reader)
        crop_labels = np.expand_dims(np.array(crop_labels),axis=-1)
        planet_folders_list = self.sorted_planet(self.planet_folder)
        if len(planet_folders_list) > 0 :
            for file_idx,planet_folder in enumerate(planet_folders_list):
                if file_idx == 0 :
                    time_series_array = get_planet_time_series(
                            planet_folder[1],
                            (width_index,height_index,crop_labels),
                            mode='val',
                            neighbor_count=self.neighbor_count)
                else:
                    time_series_array = np.concatenate(
                            [time_series_array,
                                get_planet_time_series(
                                    planet_folder[1],
                                    (width_index,height_index,crop_labels),
                                mode='val',
                                neighbor_count=self.neighbor_count)],axis=1)
        h5_file = h5py.File(os.path.join("./h5_folder",output_file_name),mode='w')
        h5_file.create_dataset('time_series',data=time_series_array,shape=time_series_array.shape)
        h5_file.create_dataset('crop_labels',shape=(time_series_array.shape[0],1),data=crop_labels)
        h5_file.close()

def get_day_idx_of_sentinel2(sentinel2_time_stamp):
    return list(map(lambda x : (x - datetime.datetime(2018,1,1,0,0,0,tzinfo=pytz.UTC)).days,sentinel2_time_stamp))

def match_planet_to_sentinel2(planet_folder_list,sentinel2_time_stamp):
    planet_day_list = [planet_folder[0] for planet_folder in planet_folder_list]
    return (np.arange(len(planet_day_list)),
                np.searchsorted(sentinel2_time_stamp,planet_day_list))

def match_sentinel2_to_planet(planet_folder_list,sentinel2_time_stamp):
    planet_day_list = [planet_folder[0] for planet_folder in planet_folder_list]
    return (np.arange(len(sentinel2_time_stamp)),
                np.searchsorted(planet_day_list,sentinel2_time_stamp))

def get_pretraining_data(time_idx_tuples,indices,num_points,planet_folder,is_less_data_in_sentinel2):
    indices_list = []
    for i in range(time_idx_tuples[0].shape[0]):
        indices_list.append(np.random.choice(len(indices[0]),num_points,replace=False))
    sentinel2_array = np.transpose(np.load(sentinel2_numpy),axes=[0,3,1,2])
    sentinel_2_data  = []
    planet_data = []
    if not is_less_data_in_sentinel2:
        print(time_idx_tuples)
        for idx,time_index in enumerate(time_idx_tuples[1]):
            print(idx)
            sentinel_2_data.extend(sentinel2_array[
                time_index,
                :,
                indices[0][indices_list[idx]],
                indices[1][indices_list[idx]]])
            planet_data.extend(get_planet_time_series(
                planet_folder[idx][1],
                (indices[0][indices_list[idx]],indices[1][indices_list[idx]]),
                    neighbor_count=1))
    else :
        for idx,time_index in enumerate(time_idx_tuples[0]):
            print(idx)
            sentinel_2_data.extend(sentinel2_array[
                idx,
                :,
                indices[0][indices_list[idx]],
                indices[1][indices_list[idx]]])
            planet_data.extend(get_planet_time_series(
                planet_folder[time_index][1],
                (indices[0][indices_list[idx]],indices[1][indices_list[idx]]),
                neighbor_count=1))
    return np.array(sentinel_2_data),np.array(planet_data)



def get_time_wise_pretraining_data(indices,num_points,planet_folder):
    selected_indices = np.random.choice(len(indices[0]),num_points,replace=False)
    sentinel2_array = np.transpose(np.load(sentinel2_numpy),axes=[0,3,1,2])
    sentinel_2_data = sentinel2_array[:,
                                      :,
                                      indices[0][selected_indices],
                                      indices[1][selected_indices]]
    planet_folder_list_day_wise = PlanetTimeSeries.sorted_planet(planet_folder)
    for folder_idx,a_folder in enumerate(planet_folder_list_day_wise):
        if folder_idx == 0:
            planet_data = get_planet_time_series(
                    a_folder[1],
                    (indices[0][selected_indices],
                     indices[1][selected_indices]),
                    neighbor_count=1)
        else:
            planet_data = np.concatenate([planet_data,
                (get_planet_time_series(
                a_folder[1],
                (indices[0][selected_indices],
                indices[1][selected_indices]),
                neighbor_count=1))],axis=1)
        print(folder_idx)
    return np.transpose(sentinel_2_data,axes=[2,0,1]),planet_data

def planet_sentinel2_pairing(planet_folder,sentinel2_time_stamp,indices,num_points):
    planet_folder_list_day_wise = PlanetTimeSeries.sorted_planet(planet_folder)
    with open(sentinel2_time_stamp,'rb') as pickle_reader:
        sentinel2_time_stamp_data = get_day_idx_of_sentinel2(pickle.load(pickle_reader))
    if len(planet_folder_list_day_wise) > len(sentinel2_time_stamp_data):
        time_idx_tuples = match_sentinel2_to_planet(planet_folder_list_day_wise,sentinel2_time_stamp_data)
        print(time_idx_tuples)
        return get_pretraining_data(time_idx_tuples,indices,num_points,planet_folder_list_day_wise,is_less_data_in_sentinel2=True)
    else :
        time_idx_tuples = match_planet_to_sentinel2(planet_folder_list_day_wise,sentinel2_time_stamp_data)
        return get_pretraining_data(time_idx_tuples,indices,num_points,planet_folder_list_day_wise,is_less_data_in_sentinel2=False)

class PretrainingDataset:
    def __init__(self,pretraining_dataframe,pretraining_type='point',planet_folder=None,sentinel2_time_stamp=None,num_points=3000):
        self.pretraining_dataframe = gpd.read_file(pretraining_dataframe)
        self.pretraining_type = pretraining_type
        self.planet_folder = planet_folder
        self.sentinel2_time_stamp = sentinel2_time_stamp
        self.num_points = num_points
        
        pretraining_field_geoms = list(self.pretraining_dataframe['geometry'])
        mask_array = rasterio.features.rasterize(pretraining_field_geoms,out_shape=(width,height),transform=sentinel_transform)
        field_mask = np.where(mask_array==1)
        self.indices = (field_mask[0],field_mask[1])

    def return_pretaining_data(self,output_file_name="pretraining_point2.h5"):
        if self.pretraining_type == "point":
            sentinel2_data, planet_data = planet_sentinel2_pairing(self.planet_folder,self.sentinel2_time_stamp,self.indices,self.num_points)
            h5_file = h5py.File(os.path.join("./h5_folder",output_file_name),mode="w")
            h5_file.create_dataset("planet_data",shape=planet_data.shape,data=planet_data)
            h5_file.create_dataset("sentinel2_data",shape= sentinel2_data.shape,data=sentinel2_data)
            h5_file.close()

        if self.pretraining_type == "time":
            sentinel2_data, planet_data = get_time_wise_pretraining_data(self.indices,self.num_points,self.planet_folder)
            print(sentinel2_data.shape,planet_data.shape)
            output_file_name = "pretraining_time2.h5"
            h5_file = h5py.File(os.path.join("./h5_folder",output_file_name),mode="w")
            h5_file.create_dataset("planet_data",shape=planet_data.shape,data=planet_data)
            h5_file.create_dataset("sentinel2_data",shape= sentinel2_data.shape,data=sentinel2_data)
            h5_file.close()

def plot_all(planet_folder):
    #sentinel2_data = np.load(sentinel2_numpy)
    #max_value = sentinel2_data.max()
    #for i in range(sentinel2_data.shape[0]):
    #    sentinel_image = sentinel2_data[i]/(max_value)
    #    print(sentinel_image.shape)
    #    plt.axis('off')
    #    plt.imsave(f"/p/scratch/deepacf/kiste/patnala1/multimodal_images/sentinel2/{i}.png",sentinel_image[:,:,[4,3,2]])
    planet_folder_list_day_wise = PlanetTimeSeries.sorted_planet(planet_folder)
    for idx,planet_folder in enumerate(planet_folder_list_day_wise):
        planet_array = read_planet_day_wise(planet_folder[1])
        sample_array = np.transpose(planet_array.read([3,2,1]),[1,2,0])
        print(sample_array.shape)
        plt.axis('off')
        plt.imsave(f"/p/scratch/deepacf/kiste/patnala1/multimodal_images/planet/{idx}.png",sample_array/10000)



if __name__ == "__main__" :
    #time_series = TimeseriesDataset(geojson_folder=None,train_indices_folder=None,val_indices_folder=None)
    #sentinel_time_series_data = SentinelTimeSeries(geojson_folder="./geojson",train_indices_folder="./training_indices_folder",val_indices_folder="./validation_indices_folder")
    #sentinel_time_series_data.get_data(mode=['train','val'])
    #sentinel_time_series_data.return_train_data()
    #sentinel_time_series_data.return_val_data()
    #planet_time_series_data = PlanetTimeSeries(
    #                            geojson_folder="./geojson",
    #                            planet_folder=planet_folder,
    #                            train_indices_folder="./training_indices_folder",
    #                            val_indices_folder="./validation_indices_folder",
    #                            neighbor_count=1)
    #planet_time_series_data.return_train_data()
    #planet_time_series_data.return_val_data()
    #pretraining_dataset = PretrainingDataset("./geojson/pretaining.geojson",pretraining_type='point',planet_folder=planet_folder,sentinel2_time_stamp=time_stamp,num_points=100000)
   # pretraining_dataset = PretrainingDataset("./geojson/pretaining.geojson",pretraining_type='time',planet_folder=planet_folder,sentinel2_time_stamp=time_stamp,num_points=150000)
    #pretraining_dataset.return_pretaining_data()
    plot_all(planet_folder)

