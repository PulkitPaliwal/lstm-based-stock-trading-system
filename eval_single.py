import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from train_single import LSTM, train
import argparse
import os
from tqdm import tqdm
import sys
import time
import datetime
import pickle
import random
import math
from dataloaders.general_dataloader import prepare_dataloader
from dataloaders.data.extract_data import stock
# set random seed
random.seed(0)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ABM
    parser.add_argument('--analyse_stock_name', type=str, default='ABMD', help='stock name to be analysed')
    parser.add_argument('--model_stock_name', type=str, default='AMAT', help='stock model tp be used')
    parser.add_argument('--lookback', type=int, default=60, help='lookback period')
    parser.add_argument('--input_size', type=int, default=1, help='input feature size')
    parser.add_argument('--batch_size', type=int, default=700, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers')
    parser.add_argument('--dropout', type=float, default=0.4, help='dropout')
    parser.add_argument('--trade', type=str, default="day", help='trade')
    args = parser.parse_args()
    model = LSTM(input_dim = args.input_size, hidden_dim=args.hidden_size, num_layers=args.num_layers, output_dim=args.input_size, dropout=args.dropout).to(device)
    model.load_state_dict(torch.load(f"trained_models/best_model_{args.model_stock_name}_{args.trade}.pt"))
    model.eval()
    train_data = prepare_dataloader([args.analyse_stock_name], args.trade, args.lookback, 'train', {'batch_size': args.batch_size, 'shuffle': True}, None)
    GDSET, GDLOADER = prepare_dataloader([args.analyse_stock_name], args.trade, args.lookback, 'GENERAL', {'batch_size': args.batch_size, 'shuffle': False}, train_data[0].scalar)
    stock_closing_prices = []
    stock_df = stock(args.analyse_stock_name, trade_type=args.trade).df
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
    plt.plot(range(len(stock_closing_prices)), stock_closing_prices, color = 'black', label = f'Real {args.analyse_stock_name} Stock Price')
    plt.plot(np.array(range(len(predicted_closing_prices))), predicted_closing_prices, color = 'red', label = f'Predicted {args.analyse_stock_name} Stock Price')
    plt.title(f'{args.analyse_stock_name} Stock Price Prediction on {args.model_stock_name} model')
    plt.xlabel('Time')
    plt.ylabel(f'{args.analyse_stock_name} Stock Price')
    plt.legend()
    plt.show()