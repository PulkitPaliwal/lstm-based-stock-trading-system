'''
Assume an initial investment of 1000$. We trade in the last 20% of the dataset and see how much we make.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from train_single import LSTM, train
from dataloaders.general_dataloader import prepare_dataloader
from dataloaders.data.extract_data import stock
portfolio = 1000
portfolio_history = []
portfolio_history.append(portfolio)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyse_stock_name', type=str, default='ADSK', help='stock name to be analysed')
    parser.add_argument('--model_stock_name', type=str, default='ADSK', help='stock model tp be used')
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
    test_data, test_loader = prepare_dataloader([args.analyse_stock_name], args.trade, args.lookback, 'GENERAL', {'batch_size': args.batch_size, 'shuffle': False}, train_data[0].scalar)

    stock_closing_prices = []

    predicted_closing_prices = []
    with torch.no_grad():
        for X_test, y in test_loader:
            y_pred = model(X_test.float().to(device))
            y_pred = y_pred.cpu().numpy()
            # inverse transform of y_pred
            y_pred = train_data[0].scalar.inverse_transform(y_pred)
            for i in range(len(y_pred)):
                predicted_closing_prices.append(y_pred[i])
                stock_closing_prices.append(y[i].cpu().numpy())
    predicted_closing_prices = np.array(predicted_closing_prices)

    # we assume that the person invests as soon as the market opens
    # depending upon whether the predicted price is higher or lower than the current price, we short or trade
    invested = False
    if predicted_closing_prices[0] > stock_closing_prices[0]:
        invested = True
        num_shares_current = portfolio/stock_closing_prices[0]

    def find_next_high_day(current_day):
        return_day = current_day
        for i in range(1 + current_day, len(predicted_closing_prices)):
            if predicted_closing_prices[i] > predicted_closing_prices[current_day]:
                return_day = i
            elif return_day != current_day:
                return return_day
        return None

    def find_next_low_day(current_day):
        return_day = current_day
        for i in range(1 + current_day, len(predicted_closing_prices)):
            if predicted_closing_prices[i] < predicted_closing_prices[current_day]:
                return_day = i
            elif return_day != current_day:
                return return_day
        return None

    trading_days = []
    trading_decisions = []
    trading_outcome = []

    def make_decision(current_day):
        # find the next day where the rising or the falling trend changes
        # if the trend is rising, we sell
        # if the trend is falling, we buy
        next_high_day = find_next_high_day(current_day)
        next_low_day = find_next_low_day(current_day)

        if next_high_day is not None and next_low_day is not None:
            if next_low_day < next_high_day:
                    return "sell", next_low_day
            else:
                    return "buy", next_high_day
        elif next_high_day is not None:
            return "buy", next_high_day
        else:
            return "sell", next_low_day
        
    def judge_outcome(buy_price, action, nd):
        if action == "buy":
            if stock_closing_prices[nd] > buy_price:
                return "profit"
            else:
                return "loss"
        elif action == "sell":
            if stock_closing_prices[nd] < buy_price:
                return "profit"
            else:
                return "loss"

    current_day = 0
    strategy, next_day = make_decision(current_day)
    control_percentage_for_short = 0.2
    while current_day is not None and current_day < len(test_data):
        print(current_day)
        trading_days.append(current_day)
        trading_decisions.append(strategy)
        if strategy == "buy":
            shares_traded = portfolio/stock_closing_prices[current_day]
            portfolio = 0
            trading_outcome.append(judge_outcome(stock_closing_prices[current_day], strategy, next_day))
            if trading_outcome[-1] == "profit":
                portfolio = shares_traded*stock_closing_prices[next_day]
            elif trading_outcome[-1] == "loss":
                portfolio = shares_traded*stock_closing_prices[next_day]
        elif strategy == "sell":
            shares_traded = control_percentage_for_short*portfolio/stock_closing_prices[current_day]
            trading_outcome.append(judge_outcome(stock_closing_prices[current_day], strategy, next_day))
            portfolio = portfolio - shares_traded*(stock_closing_prices[next_day] - stock_closing_prices[current_day])
        current_day = next_day
        portfolio_history.append(portfolio)
        if current_day is not None : strategy, next_day = make_decision(current_day)
        if next_day is None: break

    print(portfolio)
    print(portfolio_history)

    # plot portfolio history
    plt.plot(range(len(portfolio_history)), portfolio_history, color = 'black', label = f'Portfolio')
    plt.title(f'Portfolio over time for {args.analyse_stock_name}')
    plt.xlabel('Time')
    plt.ylabel(f'Portfolio')
    plt.legend()
    plt.show()