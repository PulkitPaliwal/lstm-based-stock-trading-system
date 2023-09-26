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

class stock:
    def __init__(self, stock_name, plot_mode: str = None, verbose: bool = False) -> None:
        self.stock_name = stock_name
        self.df = None
        self.plot = None
        self.verbose = verbose
        self.get_df()
        if plot_mode is not None:
            if "mbm" in plot_mode:
                self.plot_minute_by_minute_closing_prices()
            if "dbd" in plot_mode:
                self.plot_day_by_day_closing_prices()
            if "candle" in plot_mode:
                self.plot_day_candlesticks()
            if "analyze" in plot_mode:
                self.analyze_missing()

    def get_df(self):
        # text files with data
        file_name = './data/sp500_tickers_A-D_1min_1pppix/' + self.stock_name + '_1min.txt'
        # read data
        df = pd.read_csv(file_name, sep=',', parse_dates=[0] , header=None)
        # set column names
        df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        # set index
        df.set_index('datetime', inplace=True)
        self.df = df

    def plot_minute_by_minute_closing_prices(self):
        # plot prices corresponing to last trade of every minute
        df = self.df
        # resample to get last trade of the minute
        self.plot = df['close'].resample('1Min').last().dropna().plot(figsize=(15, 5), title=self.stock_name)
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
        # resample to get last trade of the day
        self.plot = df['close'].resample('D').last().dropna().plot(figsize=(15, 5), title=self.stock_name)
        # add title
        self.plot.set_title(self.stock_name + ' day by day closing prices')
        # add x labels as dates 
        self.plot.set_xlabel('Day')
        # add y label
        self.plot.set_ylabel('Closing Price')
        # show plot
        plt.show()
    
    def plot_day_candlesticks(self):  
        # plot using plotly's candlestick chart plotting tool
        # visible spaces on the plot are the time periods when the market is closed
        df = self.df
        dfN = pd.DataFrame()
        dfN['high'] = df['high'].resample('D').max().dropna()
        dfN['low'] = df['low'].resample('D').min().dropna()
        dfN['open'] = df['open'].resample('D').first().dropna()
        dfN['close'] = df['close'].resample('D').last().dropna()
        plot = go.Figure(data=[go.Candlestick(x=dfN.index,
                open=dfN['open'],
                high=dfN['high'],
                low=dfN['low'],
                close=dfN['close'])])
        plot.update_layout(xaxis_rangeslider_visible=False, title=self.stock_name + ' candlesticks chart')
        plot.show()

    def analyze_missing(self):
        df = self.df
        # check for dates with trades outside 09:30 to 16:00
        df['time'] = df.index.time
        df['date'] = df.index.date
        for date in df['date'].unique():
            df_date = df[df['date'] == date]
            if df_date['time'].max() > pd.Timestamp('16:00:00').time() or df_date['time'].min() < pd.Timestamp('09:30:00').time():
                if self.verbose:
                    print("Date with trades outside 09:30 to 16:00:", date)
        # check for missing dates
        date_range = pd.date_range(start=df['date'].min(), end=df['date'].max())
        missing_dates = date_range.difference(df['date'])
        if self.verbose:
            print("Missing dates:", missing_dates)
        # check for missing minutes
        df['minute'] = df.index.minute
        for date in df['date'].unique():
            df_date = df[df['date'] == date]
            minute_range = pd.date_range(start=df_date.index.min(), end=df_date.index.max(), freq='1Min')
            missing_minutes = minute_range.difference(df_date.index)
            if self.verbose:
                print("Missing minutes on date:", date, missing_minutes)

          
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_name', type=str, default='AAPL')
    parser.add_argument('--plot_type', type=str, default='mbm')
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()
    stock_cl = stock(args.stock_name, args.plot_type, args.verbose)
    stock_cl.analyze_missing()