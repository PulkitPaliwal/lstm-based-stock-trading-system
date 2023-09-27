'''
run python train_single.py --stock_name= {item from dataloaders/data/sp500tickers...../}
'''
import os

stocks = os.listdir('./dataloaders/data/sp500_tickers_A-D_1min_1pppix/')
stock_names = [stocks[i].split('_')[0] for i in range(len(stocks))]
for stock_name in stock_names:
    # run python train_single.py --stock_name=stock_name
    os.system(f'python train_single.py --stock_name={stock_name}')