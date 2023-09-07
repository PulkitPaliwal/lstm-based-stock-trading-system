'''
Question.1 Familiarize yourself with the input data sp500_tickers_A-D_1min_1pppix.zip: [1]
a) Plot the minute-by-minute closing price series of few stocks
b) Plot the day-by-day closing price series of a few stocks
c) Plot a complete candlestick chart with volume on secondary y-axis for a few stocks with a
time period of your choice
d) Note down your observations, e.g. are there any data issues, unexpected jumps,
unexpected missing data etc.
'''

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import argparse 
# a) Plot the minute-by-minute closing price series of few stocks
class stock:
    def __init__(self, stock_name, plot_mode) -> None:
        self.stock_name = stock_name
        self.df = None
        self.plot = None
        self.get_df()
        # self.plot_day_by_day_closing_prices()
        # self.plot_minute_by_minute_closing_prices()
        self.plot_candlesticks()
        self.analyze_missing()

    def get_df(self):
        # text files with data
        file_name = '/home/pulkit/lstm-based-stock-trading-system/data/sp500_tickers_A-D_1min_1pppix/' + self.stock_name + '.txt'
        # read data
        df = pd.read_csv(file_name, sep=',', parse_dates=[0] , header=None)
        # set column names
        df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        # set index
        df.set_index('datetime', inplace=True)
        # extract time from datetime
        self.df = df

    def plot_minute_by_minute_closing_prices(self):
        df = self.df
        self.plot = df['close'].resample('1Min').mean().dropna().plot(figsize=(15, 5), title=self.stock_name)
        # add title
        self.plot.set_title(self.stock_name + ' minute by minute closing prices')
        # add x label
        self.plot.set_xlabel('Date')
        # add y label
        self.plot.set_ylabel('Closing Price')
        # show plot
        plt.show()

    def plot_day_by_day_closing_prices(self):
        # plot prices corresponing to last trade everyday
        df = self.df
        self.plot = df['close'].resample('D').last().dropna().plot(figsize=(15, 5), title=self.stock_name)
        self.plot.set_title(self.stock_name + ' day by day closing prices')
        # add x labels as dates 
        self.plot.set_xlabel('Day')
        # add y label
        self.plot.set_ylabel('Closing Price')
        # show plot
        plt.show()
    
    def plot_candlesticks(self):  
        # plot using plotly's candlestick chart plotting tool
        # visible spaces on the plot are the time periods when the market is closed
        df = self.df[5000:]   
        df['high'] = df['high'].resample('D').max()
        df['low'] = df['low'].resample('D').min()
        df['open'] = df['open'].resample('D').first()
        df['close'] = df['close'].resample('D').last()
        plot = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'])])
        plot.update_layout(xaxis_rangeslider_visible=False)
        plot.show()

    def analyze_missing(self):
        df = self.df
        # check for missing values
        # print all NaN closings    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_name', type=str, default='AAPL_1min')
    parser.add_argument('--plot_type', type=str, default='day_by_day_closing_prices')
    args = parser.parse_args()
    stock_cl = stock(args.stock_name, args.plot_type)
    stock_cl.analyze_missing()