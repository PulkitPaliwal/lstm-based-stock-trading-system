import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dataloaders.data.extract_data import *

class stockDataset(torch.utils.data.Dataset):
    def __init__(self, stock_name, lookback, train_test_split, train_scaler):
        self.lookback = lookback
        # take 70% train, 20% val and 10% test
        if train_test_split == 'train':    
            self.stock_data_df = stock(stock_name).df[:int(len(stock(stock_name).df)*0.7)]
            train_scaler = MinMaxScaler(feature_range=(0, 1))
            self.data = train_scaler.fit_transform((np.array(self.stock_data_df['close'].values)).reshape(-1, 1))
        elif train_test_split == 'val':
            self.stock_data_df = stock(stock_name).df[int(len(stock(stock_name).df)*0.7):int(len(stock(stock_name).df)*0.9)]
            print("Using scaler: ", train_scaler)
            self.data = train_scaler.transform(np.array(self.stock_data_df['close'].values).reshape(-1, 1))
        elif train_test_split == 'test':
            self.stock_data_df = stock(stock_name).df[int(len(stock(stock_name).df)*0.9):]    
            print("Using scaler: ", train_scaler)
            self.data = train_scaler.transform(np.array(self.stock_data_df['close'].values).reshape(-1, 1))
        self.scalar = train_scaler
    def __len__(self):
        num_samples = len(self.stock_data_df)
        return num_samples
    def __getitem__(self, idx):
        # return closing prices of last lookback minutes
        if idx+self.lookback >= len(self.data):
            return (self.data[-self.lookback:]), self.data[-1]
        else:
            return (self.data[idx:idx+self.lookback]), self.data[idx+self.lookback]
    
def prepare_dataloader(
    stock_name: str,
    lookback: int,
    train_test_split: str,
    params: dict,
    train_scaler: StandardScaler,
):
    print("Preparing dataloader...")
    stock_dataset = stockDataset(stock_name, lookback, train_test_split, train_scaler)
    print(f"Dataset Size: {len(stock_dataset)}")
    print("X shape: ", stock_dataset[0][0].shape)
    print("y shape: ", stock_dataset[0][1].shape)
    batch_size = params['batch_size']
    shuffle = params['shuffle']
    stock_dataloader = DataLoader(stock_dataset, batch_size = batch_size, shuffle = shuffle)
    return stock_dataset, stock_dataloader

          
if __name__ == "__main__":
    x , y = prepare_dataloader('AAPL', 60, 'train', {'batch_size': 32, 'shuffle': True}, None)