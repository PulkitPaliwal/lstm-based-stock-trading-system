'''
train an LSTM model for a single stock and save the best model. options for training include:
    - stock_name: stock name
    - lookback: lookback period
    - input_size: input feature size
    - batch_size: batch size
    - num_epochs: number of epochs
    - learning_rate: learning rate
    - hidden_size: hidden size
    - num_layers: number of layers
    - dropout: dropout
    - trade: trade type (day or hft)
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
from dataloaders.data.extract_data import stock
from dataloaders.general_dataloader import prepare_dataloader
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
    
def train(model, train_data, val_data, learning_rate, num_epochs, info):
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
            # val_loss = val_loss
            # check val_loss and update best_model if new val loss is less than the current best val loss
            # avoids overfitting
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                if not os.path.exists('./trained_models/'):
                    os.mkdir('./trained_models/')
                torch.save(model.state_dict(), './trained_models/best_model_'+ info[0] + '_' + info[1] +'.pt')
                print('Model saved')
            print(f'Epoch: {epoch}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}')
    return best_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_name', type=str, default='AMAT', help='stock name')
    parser.add_argument('--lookback', type=int, default=60, help='lookback period')
    parser.add_argument('--input_size', type=int, default=1, help='input feature size')
    parser.add_argument('--batch_size', type=int, default=700, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers')
    parser.add_argument('--dropout', type=float, default=0.4, help='dropout')
    parser.add_argument('--trade', type=str, default="day", help='trade')

    args = parser.parse_args()
    stock_name = args.stock_name
    lookback = args.lookback
    input_size = args.input_size
    output_size = args.input_size
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    num_layers = args.num_layers
    num_epochs = args.num_epochs
    batch_first = True
    dropout = args.dropout
    learning_rate = args.learning_rate
    stock_name = args.stock_name
    stock_lstm = LSTM(input_dim = input_size, hidden_dim=hidden_size, num_layers=num_layers, output_dim=output_size, dropout=dropout).to(device)
    train_data = prepare_dataloader([stock_name], args.trade, lookback, 'train', {'batch_size': batch_size, 'shuffle': True}, None)
    val_data = prepare_dataloader([stock_name], args.trade, lookback, 'val', {'batch_size': batch_size, 'shuffle': True}, train_data[0].scalar)
    info = [stock_name, args.trade]
    model = train(stock_lstm, train_data, val_data, learning_rate, num_epochs, info)
    test_dataset, test_loader = prepare_dataloader([args.stock_name], args.trade, args.lookback, 'test', {'batch_size': args.batch_size, 'shuffle': False}, train_data[0].scalar)

    stock_closing_prices = []
    stock_df = stock(args.stock_name, trade_type=args.trade).df
    for i in range(len(stock_df)):
        stock_closing_prices.append(stock_df['close'][i])
    stock_closing_prices = np.array(stock_closing_prices)
    GDSET, GDLOADER = prepare_dataloader([args.stock_name], args.trade, args.lookback, 'GENERAL', {'batch_size': args.batch_size, 'shuffle': False}, train_data[0].scalar)
    stock_closing_prices = []
    stock_df = stock(args.stock_name, trade_type=args.trade).df
    for i in range(len(stock_df)):
        stock_closing_prices.append(stock_df['close'][i])
    stock_closing_prices = np.array(stock_closing_prices)
    predicted_closing_prices = []
    with torch.no_grad():
        for X_test, y in GDLOADER:
            y_pred = model(X_test.float().to(device))
            y_pred = y_pred.cpu().numpy()
            # inverse transform of y_pred
            y_pred = train_data[0].scalar.inverse_transform(y_pred)
            for i in range(len(y_pred)):
                predicted_closing_prices.append(y_pred[i])
    predicted_closing_prices = np.array(predicted_closing_prices)
    print(predicted_closing_prices.shape)
    plt.plot(range(len(stock_closing_prices)), stock_closing_prices, color = 'black', label = f'Real {args.stock_name} Stock Price')
    plt.plot(np.array(range(len(predicted_closing_prices))), predicted_closing_prices, color = 'red', label = f'Predicted {args.stock_name} Stock Price')
    plt.title(f'{args.stock_name} Stock Price Prediction')
    if args.trade == 'day':
        plt.xlabel('Days')
    elif args.trade == 'hft':
        plt.xlabel('Minutes')
    plt.ylabel(f'{args.stock_name} Stock Price')
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig(f'./plots_standard/{args.stock_name}_{args.trade}.png')