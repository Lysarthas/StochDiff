import wfdb
import numpy as np
import pandas as pd
import scipy.io as io
from os import error
from torch.utils.data import Dataset
from tools import IndividualScaler, TimeFeature, MinMaxScaler


class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    

class RegDataset(Dataset):
    def __init__(self, ts_set, lag_set, time_set):
        self.ts_set = ts_set
        self.lag_set = lag_set
        self.time_set = time_set

    def __len__(self):
        return len(self.ts_set)

    def __getitem__(self, index):
        return self.ts_set[index], self.lag_set[index], self.time_set[index]
    
def load_dataset(ds_name, ws_size, time_dim=None, predict_len=10, scale_sty='stand', training = True):
    np.random.seed(0)
    mat_old = io.loadmat('datasets/apollo_data.mat')
    mat_new = io.loadmat('datasets/apollo_new.mat')
    time_feat = TimeFeature(f_type='F')

    all_data = []
    for mat in [mat_old, mat_new]:
        for raw_cm, time_elps in zip(mat['cmAmp'][0], mat['timeElapse'][0]):
            if raw_cm.shape[1] < ws_size or np.isnan(raw_cm).any():
                continue
            time_embd = time_feat.get_features(time_elapse=time_elps.squeeze(), dim=time_dim)
            all_data.append(np.concatenate((raw_cm[:218, :].T, time_embd), axis=1))
#                 all_data.append(raw_cm[:218, :].T)

    # Randomly shuffle the data
    np.random.shuffle(all_data)
    # Split data into training and testing sets
    split_idx = int(len(all_data)*0.7)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
        
    if training:
        X_train = []
        for signal in train_data:
            for j in range(len(signal)-ws_size):
                x = signal[j:j+ws_size]
                if (x.sum(axis=0) == 0).any():
                    continue
                X_train.append(x)
        scaler = IndividualScaler(style=scale_sty)
        X_train = scaler.fit_transform(X_train)
        return TimeSeriesDataset(X_train)
    else:
        X_test = []
        for signal in test_data:
            if len(signal)<ws_size+predict_len:
                continue
            for j in range(len(signal)-ws_size-predict_len):
                x = signal[j:j+ws_size+predict_len]
                if (x.sum(axis=0) == 0).any():
                    continue
                X_test.append(x)
        X_test = np.array(X_test)
        return TimeSeriesDataset(X_test)