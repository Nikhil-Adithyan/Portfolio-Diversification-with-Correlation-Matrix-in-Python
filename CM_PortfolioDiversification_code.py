# IMPORTING PACKAGES

import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import floor
from termcolor import colored as cl

from matplotlib import style
from matplotlib import rcParams

style.use('fivethirtyeight')
rcParams['figure.figsize'] = (20, 10)

# EXTRACTING STOCKS DATA

def get_historical_data(symbol, start_date, end_date):
    api_key = 'YOUR API KEY'
    api_url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&outputsize=5000&apikey={api_key}'
    raw_df = requests.get(api_url).json()
    df = pd.DataFrame(raw_df['values']).iloc[::-1].set_index('datetime').astype(float)
    df = df[df.index >= start_date]
    df = df[df.index <= end_date]
    df.index = pd.to_datetime(df.index)
    return df

fb = get_historical_data('FB', '2020-01-01', '2021-01-01')
amzn = get_historical_data('AMZN', '2020-01-01', '2021-01-01')
aapl = get_historical_data('AAPL', '2020-01-01', '2021-01-01')
nflx = get_historical_data('NFLX', '2020-01-01', '2021-01-01')
googl = get_historical_data('GOOGL', '2020-01-01', '2021-01-01')

# CALCULATING RETURNS

fb_rets, fb_rets.name = fb['close'] / fb['close'].iloc[0], 'fb'
amzn_rets, amzn_rets.name = amzn['close'] / amzn['close'].iloc[0], 'amzn'
aapl_rets, aapl_rets.name = aapl['close'] / aapl['close'].iloc[0], 'aapl'
nflx_rets, nflx_rets.name = nflx['close'] / nflx['close'].iloc[0], 'nflx'
googl_rets, googl_rets.name = googl['close'] / googl['close'].iloc[0], 'googl'

plt.plot(fb_rets, label = 'FB')
plt.plot(amzn_rets, label = 'AMZN')
plt.plot(aapl_rets, label = 'AAPL')
plt.plot(nflx_rets, label = 'NFLX')
plt.plot(googl_rets, label = 'GOOGL', color = 'purple')
plt.legend(fontsize = 16)
plt.title('FAANG CUMULATIVE RETURNS')
plt.show()

# CREATING THE CORRELATION MATRIX

rets = [fb_rets, amzn_rets, aapl_rets, nflx_rets, googl_rets]
rets_df = pd.DataFrame(rets).T.dropna()
rets_corr = rets_df.corr()

plt.style.use('default')
sns.heatmap(rets_corr, annot = True, linewidths = 0.5)
plt.show()

# BACKTESTING

investment_value = 100000
N = 2
nflx_allocation = investment_value / N
googl_allocation = investment_value / N

nflx_stocks = floor(nflx_allocation / nflx['close'][0])
googl_stocks = floor(googl_allocation / googl['close'][0])

nflx_investment_rets = nflx_rets * nflx_stocks
googl_investment_rets = googl_rets * googl_stocks
total_rets = round(sum(((nflx_investment_rets + googl_investment_rets) / 2).dropna()), 3)
total_rets_pct = round((total_rets / investment_value) * 100, 3)

print(cl(f'Profit gained from the investment : {total_rets} USD', attrs = ['bold']))
print(cl(f'Profit percentage of our investment : {total_rets_pct}%', attrs = ['bold']))

 # VOLATILITY CALCULATION
    
rets_df['Portfolio'] = (rets_df[['googl', 'nflx']].sum(axis = 1)) / 2
daily_pct_change = rets_df.pct_change()
volatility = round(np.log(daily_pct_change + 1).std() * np.sqrt(252), 5)

companies = ['FACEBOOK', 'APPLE', 'AMAZON', 'NFLX', 'GOOGL', 'PORTFOLIO']
for i in range(len(volatility)):
    if i == 5:
        print(cl(f'{companies[i]} VOLATILITY : {volatility[i]}', attrs = ['bold'], color = 'green'))
    else:
        print(cl(f'{companies[i]} VOLATILITY : {volatility[i]}', attrs = ['bold']))