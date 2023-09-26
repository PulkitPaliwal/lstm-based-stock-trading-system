'''

'''
import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import argparse
import os
from tqdm import tqdm
import sys
import time
import datetime
import pickle
import random
import math
from dataloader import taxinet_prepare_dataloader
# set random seed
random.seed(0)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout = dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)        
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out
    
def train(model, train_data, val_data, learning_rate, num_epochs):
    train_dataset, train_dataloader = train_data
    val_dataset, val_dataloader = val_data
    best_val_loss = float('inf')
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in tqdm(range(num_epochs)):
        model.train()        
        for X_batch, y_batch in tqdm(train_dataloader):
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.float().to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for X_batch, y_batch in val_dataloader:
                X_batch = X_batch.float().to(device)
                y_batch = y_batch.float().to(device)
                y_pred = model(X_batch)
                val_loss += loss_fn(y_pred, y_batch)
            
            # check val_loss and update best_model if new val loss is less than the current best val loss
            # avoids overfitting
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if not os.path.exists('./trained_models/'):
                    os.mkdir('./trained_models/')
                torch.save(model.state_dict(), './trained_models/best_model.pth')
                print('Model saved')
            print(f'Epoch: {epoch}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}')

if __name__ == "__main__":
    input_size = 1
    output_size = 1
    hidden_size = 64
    num_layers = 2
    batch_first = True
    dropout = 0.2
    learning_rate = 0.001
    stock_lstm = LSTM(input_dim = input_size, hidden_dim=hidden_size, num_layers=num_layers, output_dim=output_size, dropout=dropout).to(device)
    train_data = taxinet_prepare_dataloader('A', 60, 'train', {'batch_size': 100, 'shuffle': True}, None)
    val_data = taxinet_prepare_dataloader('A', 60, 'val', {'batch_size': 100, 'shuffle': True}, train_data[0].scalar)
    train(stock_lstm, train_data, val_data, learning_rate, 1000)