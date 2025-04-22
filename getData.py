# GetData.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import backtrader as bt

import requests
from bs4 import BeautifulSoup

def get_sp500_tickers():
    url = 'https://www.slickcharts.com/sp500'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    
    if not response.ok:
        raise Exception(f"Request failed: {response.status_code}")

    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'table table-hover table-borderless table-sm'})

    if table is None:
        raise ValueError("Could not find the table. Check if the class name has changed.")

    tickers = []
    for row in table.tbody.find_all('tr'):
        cols = row.find_all('td')
        if len(cols) >= 3:
            ticker = cols[2].text.strip()
            tickers.append(ticker)
    return tickers


def get_sp500_data():
    # tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    tickers = get_sp500_tickers()
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)

    data_dict = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date)

            if not df.empty:
                df.dropna(inplace=True)

                # Flatten MultiIndex if necessary
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Drop 'Adj Close' if present
                if 'Adj Close' in df.columns:
                    df.drop(columns=['Adj Close'], inplace=True)

                data_dict[ticker] = df
        except Exception as e:
            print(f"Failed for {ticker}: {e}")
    return data_dict

def print_data(data_dict):
    for ticker, df in data_dict.items():
        print(f"\nTicker: {ticker}")
        print(df.head())


# #Get hourly

# def get_sp500_tickers():
#     url = 'https://www.slickcharts.com/sp500'
#     headers = {'User-Agent': 'Mozilla/5.0'}
#     response = requests.get(url, headers=headers)
    
#     if not response.ok:
#         raise Exception(f"Request failed: {response.status_code}")

#     soup = BeautifulSoup(response.text, 'html.parser')
#     table = soup.find('table', {'class': 'table table-hover table-borderless table-sm'})

#     if table is None:
#         raise ValueError("Could not find the table. Check if the class name has changed.")

#     tickers = []
#     for row in table.tbody.find_all('tr'):
#         cols = row.find_all('td')
#         if len(cols) >= 3:
#             ticker = cols[2].text.strip()
#             tickers.append(ticker)
#     return tickers

# sp500_tickers = get_sp500_tickers()



# def get_sp500_data(interval='60m', period='730d', tickers=None, chunk_size=10, delay=2, save_path=None):
#     if tickers is None:
#         # If no tickers are provided, use the S&P 500 tickers
#         tickers = sp500_tickers

#     data_dict = {}

#     for i in range(0, len(tickers), chunk_size):
#         chunk = tickers[i:i + chunk_size]
#         print(f"Downloading chunk {i//chunk_size + 1}: {chunk}")

#         for ticker in chunk:
#             try:
#                 df = yf.download(ticker, interval=interval, period=period, progress=False)
#                 if not df.empty:
#                     df.dropna(inplace=True)
#                     data_dict[ticker] = df
#                     if save_path:
#                         os.makedirs(save_path, exist_ok=True)
#                         df.to_csv(os.path.join(save_path, f"{ticker}.csv"))
#             except Exception as e:
#                 print(f"Failed for {ticker}: {e}")
#             time.sleep(delay)  # Be nice to the server

#     return data_dict

