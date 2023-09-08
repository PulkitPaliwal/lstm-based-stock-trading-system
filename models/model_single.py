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

# set random seed
random.seed(0)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTM(torch.nn.module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first, dropout):
        # define a generalized LSTM class
        super(LSTM, self).__init__(input_size, hidden_size, num_layers, batch_first, dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first, dropout).to(device)
        self.linear = nn.ModuleList([nn.Linear(hidden_size, 1).to(device) for i in range(len(num_layers))])
    
    def forward(self, x):
        x, _ = self.lstm(x)
        for i in range(len(self.linear)):
            x = self.linear[i](x)
        return x
    
def train(model, train_data, val_data, learning_rate, num_epochs):
    train_dataset, train_dataloader = train_data
    val_dataset, val_dataloader = val_data
    best_val_loss = float('inf')
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for X_batch, y_batch in train_dataloader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for X_batch, y_batch in val_dataloader:
                y_pred = model(X_batch)
                val_loss += loss_fn(y_pred, y_batch)
            
            # check val_loss and update best_model if new val loss is less than the current best val loss
            # avoids overfitting
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if not os.path.exists('/home/pulkit/trained_models/'):
                    os.mkdir('/home/pulkit/trained_models/')
                torch.save(model.state_dict(), '/home/pulkit/trained_models/best_model.pth')
                print('Model saved')
            print(f'Epoch: {epoch}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}')
        