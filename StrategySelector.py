import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class MarketRegimeAnalyzer:
    """
    Identifies the current market regime (trending, mean-reverting, volatile, etc.)
    based on various market indicators.
    
    This helps determine which strategies are likely to perform best in current conditions.
    """
    
    def __init__(self, lookback_period=60):
        self.lookback_period = lookback_period
        self.regimes = ['Trending_Up', 'Trending_Down', 'Mean_Reverting', 'Volatile', 'Ranging']
        self.kmeans = KMeans(n_clusters=len(self.regimes), random_state=42)
        self.scaler = StandardScaler()
        self.fitted = False
    
    def calculate_features(self, price_data):
        """Calculate regime-identifying features from price data."""
        if len(price_data) < self.lookback_period:
            return None
        
        # Use the most recent data up to lookback_period
        recent_data = price_data[-self.lookback_period:]
        
        # Calculate returns
        returns = np.diff(recent_data) / recent_data[:-1]
        
        # Features
        features = {
            # Trend features
            'return': (recent_data[-1] / recent_data[0]) - 1,  # Total return
            'ma_ratio': np.mean(recent_data[-20:]) / np.mean(recent_data),  # Short/long MA
            
            # Volatility features
            'volatility': np.std(returns),
            'high_low_range': np.mean(np.max(recent_data) - np.min(recent_data)) / np.mean(recent_data),
            
            # Mean reversion features
            'mean_deviation': np.mean(np.abs(recent_data - np.mean(recent_data))) / np.mean(recent_data),
            'autocorrelation': np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0,
            
            # Other features
            'skew': pd.Series(returns).skew(),
            'kurtosis': pd.Series(returns).kurtosis()
        }
        
        return features
    
    def fit(self, historical_prices):
        """Train the regime classifier on historical data."""
        # Extract features from each window of the historical data
        feature_list = []
        
        for i in range(self.lookback_period, len(historical_prices)):
            window = historical_prices[i-self.lookback_period:i]
            features = self.calculate_features(window)
            if features:
                feature_list.append(list(features.values()))
        
        if not feature_list:
            return False
        
        # Scale features and fit KMeans
        X = self.scaler.fit_transform(feature_list)
        self.kmeans.fit(X)
        self.fitted = True
        return True
    
    def predict_regime(self, recent_prices):
        """Predict the current market regime."""
        if not self.fitted or len(recent_prices) < self.lookback_period:
            return "Unknown"
        
        features = self.calculate_features(recent_prices)
        if not features:
            return "Unknown"
        
        # Scale and predict
        X = self.scaler.transform([list(features.values())])
        cluster = self.kmeans.predict(X)[0]
        
        # Map cluster to regime name
        # This is a simple mapping - in practice, you would analyze each cluster
        # to determine its characteristics
        return self.regimes[cluster]


class StrategyEvaluator:
    """
    Evaluates performance of different strategies and recommends allocations
    based on historical performance and current market regime.
    """
    
    def __init__(self, strategies, initial_cash=10000, evaluation_period=60):
        self.strategies = strategies
        self.initial_cash = initial_cash
        self.evaluation_period = evaluation_period
        self.performance_history = {}
        self.regime_analyzer = MarketRegimeAnalyzer()
    
    def initialize_with_historical_data(self, historical_data):
        """Initialize with historical data to train regime analyzer."""
        # Extract price series from historical data for regime analysis
        if len(historical_data) > 0:
            # Just use the first ticker's data for regime analysis
            ticker = list(historical_data.keys())[0]
            prices = historical_data[ticker]['Close'].values
            self.regime_analyzer.fit(prices)
    
    def evaluate_strategies(self, data_dict, recent_period=60):
        """
        Evaluate strategies on recent data to determine performance in current regime.
        """
        results = {}
        regime = "Unknown"
        
        # Determine current market regime
        if len(data_dict) > 0:
            ticker = list(data_dict.keys())[0]
            prices = data_dict[ticker]['Close'].values[-max(recent_period, self.regime_analyzer.lookback_period):]
            regime = self.regime_analyzer.predict_regime(prices)
        
        # Run backtest for each strategy on recent data
        from TradingSim import run_single_strategy
        
        recent_data = {}
        for ticker, df in data_dict.items():
            recent_data[ticker] = df.tail(recent_period).copy()
        
        for strategy in self.strategies:
            strategy_name = strategy.__name__
            result = run_single_strategy(strategy, initial_cash=self.initial_cash, data_dict=recent_data)
            
            if result:
                results[strategy_name] = result
                
                # Update performance history
                if strategy_name not in self.performance_history:
                    self.performance_history[strategy_name] = []
                
                self.performance_history[strategy_name].append({
                    'date': datetime.now().strftime("%Y-%m-%d"),
                    'regime': regime,
                    'return': result['return'],
                    'sharpe': result['sharpe']
                })
        
        return {
            'current_regime': regime,
            'strategy_results': results
        }
    
    def get_strategy_allocations(self, min_allocation=0.05):
        """
        Determine optimal capital allocation across strategies based on:
        1. Recent performance in the current market regime
        2. Overall risk-adjusted returns (Sharpe ratio)
        
        Returns a dictionary of strategy names and their recommended allocation percentages.
        """
        if not self.performance_history:
            # Equal allocation if no history
            return {s.__name__: 1.0 / len(self.strategies) for s in self.strategies}
        
        # Get the most recent evaluation results for each strategy
        latest_results = {}
        current_regime = None
        
        for strategy_name, history in self.performance_history.items():
            if history:
                latest = max(history, key=lambda x: x['date'])
                latest_results[strategy_name] = latest
                current_regime = latest['regime']  # All strategies evaluated in same regime
        
        if not latest_results:
            return {s.__name__: 1.0 / len(self.strategies) for s in self.strategies}
        
        # Calculate regime-specific performance
        regime_performance = {}
        for strategy_name, result in latest_results.items():
            # Use Sharpe ratio as the main performance metric
            regime_performance[strategy_name] = max(0.0001, result['sharpe'])  # Avoid negative/zero values
        
        # Calculate allocations proportional to performance
        total_performance = sum(regime_performance.values())
        allocations = {
            name: perf / total_performance for name, perf in regime_performance.items()
        }
        
        # Apply minimum allocation constraint
        # This ensures even poorly performing strategies get a small allocation
        # which provides more robust overall performance
        min_strategies = len(allocations)
        if min_allocation * min_strategies > 1.0:
            min_allocation = 1.0 / min_strategies
            
        remaining = 1.0 - (min_allocation * min_strategies)
        if remaining < 0:
            remaining = 0
            
        # Calculate excess above minimum for each strategy
        excess = {
            name: alloc - min_allocation for name, alloc in allocations.items()
        }
        
        total_excess = sum(max(0, e) for e in excess.values())
        
        # Recalculate allocations with minimum constraint
        final_allocations = {}
        for name in allocations:
            if total_excess > 0:
                final_allocations[name] = min_allocation + (max(0, excess[name]) / total_excess) * remaining
            else:
                final_allocations[name] = min_allocation
        
        return final_allocations
    
    def get_strategy_recommendation(self):
        """
        Get a recommendation for which strategy is likely to perform best in the current regime,
        along with suggested allocations for all strategies.
        """
        allocations = self.get_strategy_allocations()
        
        # Get top strategy
        top_strategy = max(allocations.items(), key=lambda x: x[1])
        
        # Get current regime
        current_regime = "Unknown"
        for strategy, history in self.performance_history.items():
            if history:
                current_regime = history[-1]['regime']
                break
        
        return {
            'top_strategy': top_strategy[0],
            'current_regime': current_regime,
            'allocations': allocations
        }


# Example usage:
def demo_strategy_selector():
    from TradingStrategies import (
        MLForecastingStrategy,
        MeanReversionStrategy, 
        TrendFollowingStrategy, 
        PairsTradingStrategy,
        MomentumStrategy
    )
    
    # List of available strategies
    strategies = [
        MLForecastingStrategy,
        MeanReversionStrategy,
        TrendFollowingStrategy,
        PairsTradingStrategy,
        MomentumStrategy
    ]
    
    # Create evaluator
    evaluator = StrategyEvaluator(strategies)
    
    # Get historical data
    import getData as gd
    data_dict = gd.get_sp500_data()
    
    # Initialize with historical data
    evaluator.initialize_with_historical_data(data_dict)
    
    # Evaluate current performance
    results = evaluator.evaluate_strategies(data_dict)
    
    # Get recommendations
    recommendation = evaluator.get_strategy_recommendation()
    
    print(f"Current Market Regime: {recommendation['current_regime']}")
    print(f"Top Recommended Strategy: {recommendation['top_strategy']}")
    print("\nRecommended Allocations:")
    for strategy, allocation in recommendation['allocations'].items():
        print(f"  {strategy}: {allocation:.2%}")

if __name__ == "__main__":
    demo_strategy_selector()