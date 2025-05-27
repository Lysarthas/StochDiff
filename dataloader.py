import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tools import IndividualScaler, TimeFeature

# for heterogeneous time series
class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
# for regular time series
class LaggedDataset(Dataset):
    def __init__(self, ts_set, lag_set, time_set):
        self.ts_set = ts_set
        self.lag_set = lag_set
        self.time_set = time_set

    def __len__(self):
        return len(self.ts_set)

    def __getitem__(self, index):
        return self.ts_set[index], self.lag_set[index], self.time_set[index]

# example for exchange dataset, treating as heterogeneous series    
def load_dataset(ws_size, predict_len=10, scale_sty='stand', training = True):
    np.random.seed(0)
    df = pd.read_csv('exchange.csv')
    all_data = df.to_numpy()[:, 1:].astype(np.float32)
    time_serie = df['Time Serie']
    time_feat = TimeFeature(f_type='T')
    time_emb = time_feat.get_features(data=time_serie, freq=['year','month','day'], dayfirst=False)
    split_idx = int(all_data.shape[0]*0.7)
    train_set = all_data[:split_idx]
    test_set = all_data[split_idx:]
    train_time = time_emb[:split_idx]
    test_time = time_emb[split_idx:]
    X_train, X_test = [], []
    if training:
        for j in range(len(train_set)-ws_size):
            x = train_set[j:j+ws_size]
            if (x.sum(axis=0) == 0).any():
                continue
            t = train_time[j:j+ws_size]
            X_train.append(np.concatenate([x, t], axis=-1))
        scaler = IndividualScaler(style=scale_sty)
        X_train = scaler.fit_transform(X_train)
        return TimeSeriesDataset(X_train)
    else:
        for j in range(test_set.shape[0]-ws_size-predict_len):
            x = test_set[j:j+ws_size+predict_len]
            if (x.sum(axis=0) == 0).any():
                continue
            t = test_time[j:j+ws_size+predict_len]
            X_test.append(np.concatenate([x, t], axis=-1))
        return TimeSeriesDataset(X_test)