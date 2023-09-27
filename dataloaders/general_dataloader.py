import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dataloaders.data.extract_data import stock

class singleStockDataset(torch.utils.data.Dataset):
    def __init__(self, stock_name, lookback, train_test_split, train_scaler, trade):
        self.lookback = lookback
        length = len(stock(stock_name, trade_type=trade).df)
        # make train, test, val such that each element of the three sets is of the size lookback
        # instead of splitting the data into train, test, val, we can just split the indices
        # first seven train next 2 val next 1 test and so on
        train_data = []
        val_data = []
        test_data = []
        stock_data = stock(stock_name, trade_type=trade).df['close'].values
        # use only the first 80% of the data for training
        stock_data = stock_data[:int(0.8*length)]
        for i in range(0, length):
            if i%10 != 8 and i%10 != 9:
                if i < length - lookback:
                    train_data.append(stock_data[i:i+lookback+1])
                else:
                    train_data.append(stock_data[-self.lookback:-1])  
            else:
                if i < length - lookback:
                    val_data.append(stock_data[i:i+lookback+1])
                else:
                    val_data.append(stock_data[-self.lookback:-1])  
        
        if train_test_split == 'train':    
            # MinMax Scalar tends to bound the output at multiple places
            # train_scaler = MinMaxScaler(feature_range=(-1, 1))
            # data is of the type list(list(float))
            # print(train_data)
            train_scaler = StandardScaler()
            flatten_data = np.concatenate(train_data, axis=0).flatten()
            print(flatten_data)
            self.data = train_scaler.fit_transform(flatten_data.reshape(-1, 1))
        elif train_test_split == 'val':
            flatten_data = np.concatenate(val_data, axis=0 ).flatten()
            self.data = train_scaler.transform(flatten_data.reshape(-1, 1))
            train_scaler = train_scaler
        elif train_test_split == 'test':
            st_data = stock(stock_name, trade_type=trade).df['close'].values
            st_data = st_data[int(0.8*length):]
            test_data = [st_data[i:i+lookback+1] for i in range(0, len(st_data))]
            flatten_data = np.concatenate(test_data, axis=0 ).flatten()
            self.data = train_scaler.transform(flatten_data.reshape(-1, 1))
            train_scaler = train_scaler
        elif train_test_split == "GENERAL":
            stock_data = stock(stock_name, trade_type=trade).df['close'].values
            data = []
            for i in range(0, length):
                if i < length - lookback:
                    data.append(stock_data[i:i+lookback+1])
                else:
                    data.append(stock_data[-self.lookback:-1])
            flatten_data = np.concatenate(data, axis=0 ).flatten()
            self.data = train_scaler.transform(flatten_data.reshape(-1, 1))
        self.scalar = train_scaler
    def __len__(self):
        num_samples = len(self.data)//((self.lookback) +1)
        return num_samples
    def __getitem__(self, idx):
        # return closing prices of last lookback minutes
        return self.data[idx*self.lookback:(idx+1)*self.lookback], self.data[(idx+1)*self.lookback]
    
def prepare_dataloader(
    stock_name: list[str],
    trade: str,
    lookback: int,
    train_test_split: str,
    params: dict,
    train_scaler: StandardScaler,
):
    print("Preparing dataloader...")
    if len(stock_name) == 1:
        stock_dataset = singleStockDataset(stock_name[0], lookback, train_test_split, train_scaler, trade)
        print(f"Dataset Size: {len(stock_dataset)}")
        print("X shape: ", stock_dataset[0][0].shape)
        print("y shape: ", stock_dataset[0][1].shape)
        batch_size = params['batch_size']
        shuffle = params['shuffle']
        stock_dataloader = DataLoader(stock_dataset, batch_size = batch_size, shuffle = shuffle)

    else: 
        raise NotImplementedError
    return stock_dataset, stock_dataloader

          
if __name__ == "__main__":
    x , y = prepare_dataloader(['AAPL'], 'day', 60, 'train', {'batch_size': 32, 'shuffle': True}, None)