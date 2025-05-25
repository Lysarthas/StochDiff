import numpy as np
import pandas as pd

class MinMaxScaler():

    def __init__(self):
        self.mini = None
        self.range = None

    def fit_transform(self, data): 
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data

    def fit(self, data):    
        self.mini = np.min(data, 0)
        self.range = np.max(data, 0) - self.mini

    def transform(self, data):
        numerator = data - self.mini
        scaled_data = numerator / (self.range + 1e-7)
        return scaled_data
    
    def inverse_transform(self, data):
        data *= self.range
        data += self.mini
        return data
    
class StandardScaler():

    def __init__(self) -> None:
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
    
    def transform(self, data):
        return (data-self.mean)/(self.std + 1e-07)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def inverse(self, data):
        return data*self.std + self.mean


class IndividualScaler():
    def __init__(self, style='stand'):
        self.mean = None
        self.std = None
        self.mini = None
        self.range = None
        self.denominator = None
        self.style = style

    def fit(self, data):
        if self.style == 'stand':
            self.mean = np.mean(data, axis=1)
            self.std = np.std(data, axis=1)
            self.denominator = np.maximum(self.std, np.ones_like(self.std))
        elif self.style == 'minmax':
            self.mini = np.min(data, axis=1)
            self.range = np.max(data, axis=1) - self.mini
            self.denominator = np.maximum(self.range, np.ones_like(self.range))
        elif self.style == 'mean':
            self.denominator = np.mean(np.abs(data), axis=1)
            self.denominator = np.maximum(self.denominator, np.ones_like(self.denominator))
        
    def transform(self, data):
        if self.style == 'stand':
            return np.array([(x-m)/(d) for x,m,d in zip(data, self.mean, self.denominator)])
        elif self.style == 'minmax':
            return np.array([(x-m)/d for x,m,d in zip(data, self.mini, self.denominator)])
        elif self.style == 'mean':
            return np.array([x/d for x,d in zip(data, self.denominator)])
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def inverse(self, data):
        if self.style == 'stand':
            self.mean = self.mean[:, :data.shape[-1]]
            self.denominator = self.denominator[:, :data.shape[-1]]
            return np.array([d*s + m for d,m,s in zip(data, self.mean, self.denominator)])
        elif self.style == 'minmax':
            self.mini = self.mini[:, :data.shape[-1]]
            self.denominator = self.denominator[:, :data.shape[-1]]
            return np.array([d*s + m for d,m,s in zip(data, self.mini, self.denominator)])
        elif self.style == 'mean':
            self.denominator = self.denominator[:, :data.shape[-1]]
            return np.array([d*s for d,s in zip(data, self.denominator)])
        
class TimeFeature:
    def __init__(self, f_type='F'):
        self.featuretype = f_type

    def get_features(self, **kwargs):
        if self.featuretype == 'F':
            return self.fourierpositionfeatures(kwargs['time_elapse'], kwargs['dim'])
        elif self.featuretype == 'T':
            return self.timedatefeatures(kwargs['data'], kwargs['freq'], kwargs['dayfirst'])
        
    def timedatefeatures(self, data, freq, dayfirst):
        time_feat = pd.DatetimeIndex(data, dayfirst=dayfirst)
        features = []
        for f in freq:
            features.append(self.fouriertimefeature(time_feat, f))
        return np.concatenate(features, axis=0).T
    
    def fouriertimefeature(self, data, freq):
        time_value = np.array(getattr(data, freq))
        max_value = max(time_value)+1
        steps = [x * 2.0 * np.pi / max_value for x in time_value]
        return np.vstack([np.cos(steps), np.sin(steps)])    
    
    def fourierpositionfeatures(self, time_elapsed, enc_dim, n=10000):
        elapsed_time = np.expand_dims(time_elapsed, axis=1)
        position = np.arange(enc_dim)[np.newaxis, :]
        div_term = 1 / (n ** (2 * (position // 2) / enc_dim))
        angles = elapsed_time * div_term
        positional_encoding = np.zeros_like(angles)
        positional_encoding[:, 0::2] = np.sin(angles[:, 0::2])
        positional_encoding[:, 1::2] = np.cos(angles[:, 1::2])
        return positional_encoding