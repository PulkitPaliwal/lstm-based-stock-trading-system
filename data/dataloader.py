import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

from data.extract_data import stock


class stockDataset(torch.utils.data.Dataset):
    def __init__(self, stock_name, lookback):
        self.lookback = lookback
        self.stock_data_df = stock(stock_name).df
    def __len__(self):
        num_samples = len(self.stock_data_df)
        return num_samples
    def __getitem__(self, idx):
        # return closing prices of last lookback minutes
        return self.stock_data_df.iloc[idx-self.lookback:idx]['close'].values