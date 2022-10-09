from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

import h5py

class RandomForest :
    def __init__(self,training_file,validation_file,normalizer=None):
        self.training_file = training_file
        self.validation_file = validation_file
        self.normalizer = normalizer
        self.random_forest = RandomForestClassifier()

    def forward(self):
        self()

    def training_step(self):
        data = h5py.File(self.training_file)
        x = data['time_series'][:]
        num_features,t,c = x.shape
        x = np.reshape(x,(num_features,t*c)) 
        y = np.squeeze(data['crop_labels'][:])
        print(y.shape)
        data.close()
        if self.normalizer is not None:
            self.normalizer.fit(x)
            x = self.normalizer.transform(x)
        self.random_forest.fit(x,y)
        y_pred = self.random_forest.predict(x)
        return y,y_pred

    def validation_step(self):
        data = h5py.File(self.validation_file)
        x = data['time_series'][:]
        num_features,t,c = x.shape
        x = np.reshape(x,(num_features,t*c)) 
        y = np.squeeze(data['crop_labels'][:])
        data.close()
        if self.normalizer is not None:
            x = self.normalizer.transform(x)
        y_pred = self.random_forest.predict(x)
        return y,y_pred

    def __call__(self):
        y,y_pred = self.training_step()
        y_val,y_val_pred = self.validation_step()
        print("Train data Classification Report")
        print(classification_report(y,y_pred))

        print("Validation data Classification Report")
        print(classification_report(y_val,y_val_pred))


if __name__ == "__main__":
    random_forest = RandomForest("../utils/h5_folder/train_sentinel_ts.hdf5","../utils/h5_folder/val_sentinel_ts.hdf5")
    random_forest()








