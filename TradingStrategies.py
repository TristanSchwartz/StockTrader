# TradingStrategies.py
import backtrader as bt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os
import csv
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from scipy import stats

class MLForecastingStrategy(bt.Strategy):
    params = (
        ('lookback_period', 252),
        ('prediction_horizon', 5),
        ('min_training_period', 1000),
        ('train_interval', 252),
        ('threshold', 0.01),
        ('log_file', 'trade_log_ml.csv'),
    )

    def __init__(self):
        self.bar_count = 0
        self.features = []
        self.targets = []
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.model_trained = False
        self.last_train_bar = 0
        self.data_name = self.datas[0]._name if self.datas[0]._name else "Unknown"

        # Prepare CSV file
        if os.path.exists(self.params.log_file):
            os.remove(self.params.log_file)
        with open(self.params.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Ticker', 'Action', 'Price'])

    def log_trade(self, action, price):
        with open(self.params.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.datas[0].datetime.date(0).isoformat(),
                self.data_name,
                action,
                round(price, 2)
            ])

    def next(self):
        self.bar_count += 1

        if len(self.data) < self.params.lookback_period + self.params.prediction_horizon:
            return

        closes = self.data.close.get(size=self.params.lookback_period)
        
        # Enhanced feature engineering
        ma_short = np.mean(closes[-20:])
        ma_long = np.mean(closes)
        volatility = np.std(closes)
        momentum = closes[-1] - closes[-11]
        
        # Add volume-based features
        volumes = self.data.volume.get(size=self.params.lookback_period)
        vol_avg = np.mean(volumes)
        vol_trend = np.mean(volumes[-10:]) / vol_avg if vol_avg > 0 else 1.0
        
        # Add price pattern features
        returns = np.diff(closes) / closes[:-1]
        mean_return = np.mean(returns)
        return_volatility = np.std(returns)
        
        feature_vector = [
            ma_short, ma_long, volatility, momentum, 
            vol_avg, vol_trend, mean_return, return_volatility
        ]

        # Capture target (future return)
        try:
            future_price = self.data.close[self.params.prediction_horizon]
            current_price = self.data.close[0]
            price_change = (future_price - current_price) / current_price
            self.features.append(feature_vector)
            self.targets.append(price_change)
        except IndexError:
            return

        # Train model
        if len(self.features) >= self.params.min_training_period and \
           (self.bar_count - self.last_train_bar) >= self.params.train_interval:

            X = np.array(self.features)
            y = np.array(self.targets)
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            r2 = r2_score(y, self.model.predict(X_scaled))
            self.model_trained = True
            self.last_train_bar = self.bar_count
            print(f"[{self.data_name}] ML Model Trained | R²: {r2:.4f}")

        # Predict & trade
        if self.model_trained:
            current_features_scaled = self.scaler.transform([feature_vector])
            predicted_change = self.model.predict(current_features_scaled)[0]

            if predicted_change > self.params.threshold and not self.position:
                self.buy()
                self.log_trade('BUY', self.data.close[0])

            elif predicted_change < -self.params.threshold and self.position:
                self.sell()
                self.log_trade('SELL', self.data.close[0])


class MeanReversionStrategy(bt.Strategy):
    """
    Mean Reversion Strategy based on Bollinger Bands
    
    Research basis:
    - "The Statistics of Ratios of Stock Prices" by Poterba and Summers (1988)
    - "Trading on Mean-Reversion in Equity Markets" by Soros Fund Management (2014)
    
    The strategy buys when price falls below lower Bollinger Band (oversold)
    and sells when price rises above upper Bollinger Band (overbought).
    """
    params = (
        ('period', 20),
        ('devfactor', 2),
        ('log_file', 'trade_log_mean_reversion.csv'),
    )
    
    def __init__(self):
        # Bollinger Bands
        self.bband = bt.indicators.BollingerBands(
            self.data.close, 
            period=self.params.period, 
            devfactor=self.params.devfactor
        )
        
        # RSI to confirm mean reversion conditions
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.data_name = self.datas[0]._name if self.datas[0]._name else "Unknown"
        
        # Prepare CSV file
        if os.path.exists(self.params.log_file):
            os.remove(self.params.log_file)
        with open(self.params.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Ticker', 'Action', 'Price'])
    
    def log_trade(self, action, price):
        with open(self.params.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.datas[0].datetime.date(0).isoformat(),
                self.data_name,
                action,
                round(price, 2)
            ])
    
    def next(self):
        # Buy when price is below lower band and RSI indicates oversold
        if self.data.close[0] < self.bband.lines.bot[0] and self.rsi < 30 and not self.position:
            self.buy()
            self.log_trade('BUY', self.data.close[0])
        
        # Sell when price is above upper band and RSI indicates overbought
        elif self.data.close[0] > self.bband.lines.top[0] and self.rsi > 70 and self.position:
            self.sell()
            self.log_trade('SELL', self.data.close[0])


class TrendFollowingStrategy(bt.Strategy):
    """
    Trend Following Strategy using Moving Average Crossover
    
    Research basis:
    - "The Inefficient Stock Market" by Andrew Lo (2002)
    - "Trend Following: How to Make Money in Bull, Bear and Black Swan Markets" by Michael Covel
    
    The strategy buys when the short-term moving average crosses above the long-term moving average
    (golden cross) and sells when the short-term moving average crosses below the long-term moving 
    average (death cross).
    """
    params = (
        ('fast_period', 50),
        ('slow_period', 200),
        ('log_file', 'trade_log_trend_following.csv'),
    )
    
    def __init__(self):
        # Moving averages
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.fast_period
        )
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.slow_period
        )
        
        # Crossover indicator
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        
        # ADX for trend strength
        self.adx = bt.indicators.AverageDirectionalMovementIndex(self.data, period=14)
        self.data_name = self.datas[0]._name if self.datas[0]._name else "Unknown"
        
        # Prepare CSV file
        if os.path.exists(self.params.log_file):
            os.remove(self.params.log_file)
        with open(self.params.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Ticker', 'Action', 'Price'])
    
    def log_trade(self, action, price):
        with open(self.params.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.datas[0].datetime.date(0).isoformat(),
                self.data_name,
                action,
                round(price, 2)
            ])
    
    def next(self):
        # Only trade if there's a strong trend (ADX > 25)
        if self.adx > 25:
            # Buy on golden cross
            if self.crossover > 0 and not self.position:
                self.buy()
                self.log_trade('BUY', self.data.close[0])
            
            # Sell on death cross
            elif self.crossover < 0 and self.position:
                self.sell()
                self.log_trade('SELL', self.data.close[0])


class PairsTradingStrategy(bt.Strategy):
    """
    Statistical Arbitrage Strategy based on Pairs Trading
    
    Research basis:
    - "Pairs Trading: Performance of a Relative Value Arbitrage Rule" by Gatev et al. (2006)
    - "Statistical Arbitrage in the US Equities Market" by Marco Avellaneda (2010)
    
    The strategy identifies pairs of stocks that are cointegrated and trades on the mean reversion
    of their spread. When the spread deviates significantly from its mean, we take positions in 
    both stocks expecting the spread to revert to its mean.
    """
    params = (
        ('lookback', 60),         # Period to calculate z-score
        ('entry_threshold', 2.0),  # Z-score threshold for trade entry
        ('exit_threshold', 0.5),   # Z-score threshold for trade exit
        ('log_file', 'trade_log_pairs_trading.csv'),
    )
    
    def __init__(self):
        # Ensure we have at least two data feeds
        if len(self.datas) < 2:
            raise ValueError("PairsTradingStrategy requires at least two data feeds")
        
        # Store names for logging
        self.data_names = [d._name if d._name else f"Data{i}" for i, d in enumerate(self.datas)]
        
        # Prepare for storing price history
        self.prices = [[] for _ in range(len(self.datas))]
        
        # Hedging ratios (beta) for each pair
        self.betas = {}
        
        # Will be set to True when we've collected enough data
        self.ready = False
        
        # Current positions for each asset
        self.positions = [0] * len(self.datas)
        
        # Prepare CSV file
        if os.path.exists(self.params.log_file):
            os.remove(self.params.log_file)
        with open(self.params.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Ticker1', 'Ticker2', 'Action', 'Price1', 'Price2', 'Z-Score'])
    
    def log_trade(self, ticker1, ticker2, action, price1, price2, zscore):
        with open(self.params.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.datas[0].datetime.date(0).isoformat(),
                ticker1,
                ticker2,
                action,
                round(price1, 2),
                round(price2, 2),
                round(zscore, 2)
            ])
    
    def next(self):
        # Collect price data
        for i, data in enumerate(self.datas):
            self.prices[i].append(data.close[0])
            
            # Trim to lookback period
            if len(self.prices[i]) > self.params.lookback:
                self.prices[i].pop(0)
        
        # Wait until we have enough data
        if len(self.prices[0]) < self.params.lookback:
            return
        
        if not self.ready:
            self._find_cointegrated_pairs()
            self.ready = True
            return
        
        # Process each pair
        for (i, j), beta in self.betas.items():
            # Calculate the spread
            spread = np.array(self.prices[i]) - beta * np.array(self.prices[j])
            
            # Calculate z-score
            mean_spread = np.mean(spread)
            std_spread = np.std(spread)
            
            if std_spread == 0:
                continue
                
            current_spread = self.prices[i][-1] - beta * self.prices[j][-1]
            zscore = (current_spread - mean_spread) / std_spread
            
            # Trading logic
            if abs(zscore) > self.params.entry_threshold:
                # If we're not in a position
                if self.positions[i] == 0 and self.positions[j] == 0:
                    # Determine trade direction
                    if zscore > 0:  # Spread is too high
                        # Sell the first asset and buy the second
                        self.sell(data=self.datas[i])
                        self.buy(data=self.datas[j])
                        self.positions[i] = -1
                        self.positions[j] = 1
                    else:  # Spread is too low
                        # Buy the first asset and sell the second
                        self.buy(data=self.datas[i])
                        self.sell(data=self.datas[j])
                        self.positions[i] = 1
                        self.positions[j] = -1
                    
                    self.log_trade(
                        self.data_names[i],
                        self.data_names[j],
                        'ENTER',
                        self.datas[i].close[0],
                        self.datas[j].close[0],
                        zscore
                    )
            
            # Exit position if spread reverts
            elif abs(zscore) < self.params.exit_threshold:
                # If we're in a position
                if self.positions[i] != 0 and self.positions[j] != 0:
                    # Close positions
                    if self.positions[i] > 0:
                        self.sell(data=self.datas[i])
                    else:
                        self.buy(data=self.datas[i])
                        
                    if self.positions[j] > 0:
                        self.sell(data=self.datas[j])
                    else:
                        self.buy(data=self.datas[j])
                    
                    self.positions[i] = 0
                    self.positions[j] = 0
                    
                    self.log_trade(
                        self.data_names[i],
                        self.data_names[j],
                        'EXIT',
                        self.datas[i].close[0],
                        self.datas[j].close[0],
                        zscore
                    )
    
    def _find_cointegrated_pairs(self):
        """Find pairs of stocks that are cointegrated."""
        n = len(self.datas)
        self.betas = {}
        
        # Check each pair of stocks
        for i in range(n):
            for j in range(i+1, n):
                # Get price series
                y = np.array(self.prices[i])
                x = np.array(self.prices[j])
                
                # Test for cointegration (Engle-Granger method)
                # First check correlation
                corr = np.corrcoef(x, y)[0, 1]
                if abs(corr) < 0.7:  # Require strong correlation
                    continue
                    
                # Add constant for regression
                x_const = sm.add_constant(x)
                
                # Perform regression to find hedge ratio
                model = sm.OLS(y, x_const).fit()
                beta = model.params[1]
                
                # Calculate residuals
                residuals = y - (model.params[0] + beta * x)
                
                # Test if residuals are stationary (ADF test)
                adf_result = sm.tsa.stattools.adfuller(residuals)
                if adf_result[1] < 0.05:  # p-value < 0.05 indicates stationarity
                    self.betas[(i, j)] = beta
                    print(f"Found cointegrated pair: {self.data_names[i]} and {self.data_names[j]}")


class MomentumStrategy(bt.Strategy):
    """
    Momentum Strategy based on relative strength indicator
    
    Research basis:
    - "Returns to Buying Winners and Selling Losers" by Jegadeesh and Titman (1993)
    - "Momentum" by Cliff Asness (Journal of Portfolio Management, 2013)
    
    The strategy buys assets that have performed well in the past and sells assets
    that have performed poorly, based on the observation that assets that have
    performed well (poorly) in the recent past tend to continue performing well (poorly).
    """
    params = (
        ('momentum_period', 90),   # Period for calculating momentum
        ('buy_threshold', 0.8),    # Percentile threshold for buying
        ('sell_threshold', 0.2),   # Percentile threshold for selling
        ('log_file', 'trade_log_momentum.csv'),
    )
    
    def __init__(self):
        # Store names for logging
        self.data_names = [d._name if d._name else f"Data{i}" for i, d in enumerate(self.datas)]
        
        # Returns array for each asset
        self.returns = [[] for _ in range(len(self.datas))]
        
        # Dictionary to track current positions
        self.current_positions = {i: False for i in range(len(self.datas))}
        
        # Prepare CSV file
        if os.path.exists(self.params.log_file):
            os.remove(self.params.log_file)
        with open(self.params.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Ticker', 'Action', 'Price', 'Momentum'])
    
    def log_trade(self, ticker_idx, action, price, momentum):
        with open(self.params.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.datas[0].datetime.date(0).isoformat(),
                self.data_names[ticker_idx],
                action,
                round(price, 2),
                round(momentum, 4)
            ])
    
    def next(self):
        # Wait until we have enough data
        if len(self.datas[0]) < self.params.momentum_period:
            return
        
        # Calculate momentum for each asset
        momentums = []
        for i, data in enumerate(self.datas):
            # Calculate return over the momentum period
            current_price = data.close[0]
            past_price = data.close[-self.params.momentum_period]
            
            # Avoid division by zero
            if past_price == 0:
                momentum = 0
            else:
                momentum = (current_price - past_price) / past_price
            
            momentums.append((i, momentum))
        
        # Sort by momentum
        momentums.sort(key=lambda x: x[1])
        
        # Determine buy and sell thresholds
        num_assets = len(momentums)
        buy_index = int(num_assets * (1 - self.params.buy_threshold))
        sell_index = int(num_assets * self.params.sell_threshold)
        
        # Buy top performers
        top_performers = momentums[buy_index:]
        for i, momentum in top_performers:
            if not self.current_positions[i] and momentum > 0:  # Only buy positive momentum
                self.buy(data=self.datas[i])
                self.current_positions[i] = True
                self.log_trade(i, 'BUY', self.datas[i].close[0], momentum)
        
        # Sell bottom performers
        bottom_performers = momentums[:sell_index]
        for i, momentum in bottom_performers:
            if self.current_positions[i]:
                self.sell(data=self.datas[i])
                self.current_positions[i] = False
                self.log_trade(i, 'SELL', self.datas[i].close[0], momentum)