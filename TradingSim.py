import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime
import argparse

# Import custom modules
import getData as gd
from TradingStrategies import (
    MLForecastingStrategy,
    MeanReversionStrategy,
    TrendFollowingStrategy,
    PairsTradingStrategy,
    MomentumStrategy
)
from StrategySelector import StrategyEvaluator, MarketRegimeAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_sim.log"),
        logging.StreamHandler()
    ]
)

def run_single_strategy(strategy_class, initial_cash=10000, commission=0, data_dict=None, plot=False):
    """
    Run a single trading strategy and return performance metrics.
    
    Parameters:
    - strategy_class: The strategy class to backtest
    - initial_cash: Starting capital
    - commission: Trading commission rate
    - data_dict: Dictionary of price data {ticker: dataframe}
    - plot: Whether to plot the results
    
    Returns:
    - Dictionary of performance metrics
    """
    if not data_dict:
        return None
    
    # Create a cerebro instance
    cerebro = bt.Cerebro()
    
    # Add data feeds
    for ticker, df in data_dict.items():
        # Convert to bt.feeds.PandasData
        data = bt.feeds.PandasData(
            dataname=df,
            name=ticker
        )
        cerebro.adddata(data)
    
    # Add strategy
    cerebro.addstrategy(strategy_class)
    
    # Set initial cash
    cerebro.broker.setcash(initial_cash)
    
    # Set commission
    cerebro.broker.setcommission(commission=commission)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # Run backtest
    results = cerebro.run()
    
    # Extract metrics
    strategy = results[0]
    
    # Plot if requested
    if plot:
        cerebro.plot(style='candlestick', barup='green', bardown='red')
    
    # Return performance metrics
    return {
        'strategy': strategy_class.__name__,
        'final_value': cerebro.broker.getvalue(),
        'return': strategy.analyzers.returns.get_analysis()['rtot'],
        'sharpe': strategy.analyzers.sharpe.get_analysis()['sharperatio'],
        'max_drawdown': strategy.analyzers.drawdown.get_analysis()['max']['drawdown'],
        'max_drawdown_money': strategy.analyzers.drawdown.get_analysis()['max']['moneydown']
    }

def compare_strategies(strategies, initial_cash=10000, commission=0.001, data_dict=None):
    """
    Compare multiple trading strategies and return performance metrics.
    
    Parameters:
    - strategies: List of strategy classes to backtest
    - initial_cash: Starting capital
    - commission: Trading commission rate
    - data_dict: Dictionary of price data {ticker: dataframe}
    
    Returns:
    - DataFrame of performance metrics for each strategy
    """
    if not data_dict:
        return None
    
    results = []
    
    for strategy in strategies:
        strategy_name = strategy.__name__
        logging.info(f"Running strategy: {strategy_name}")
        
        try:
            result = run_single_strategy(
                strategy,
                initial_cash=initial_cash,
                commission=commission,
                data_dict=data_dict
            )
            
            if result:
                results.append(result)
                
        except Exception as e:
            logging.error(f"Error running {strategy_name}: {e}")
    
    if not results:
        return None
    
    # Convert results to DataFrame
    return pd.DataFrame(results)

def run_trading_simulation(mode='compare', strategy_name=None, initial_cash=10000, 
                          commission=0, data_limit=None, adaptive=False):
    """
    Main function to run the trading simulation.
    
    Parameters:
    - mode: 'single', 'compare', or 'adaptive'
    - strategy_name: Name of strategy to run in 'single' mode
    - initial_cash: Starting capital
    - commission: Trading commission rate
    - data_limit: Limit number of tickers to use (for faster testing)
    - adaptive: Whether to use adaptive strategy selection
    """
    logging.info(f"Starting trading simulation in {mode} mode")
    
    # Define available strategies
    all_strategies = {
        'MLForecasting': MLForecastingStrategy,
        'MeanReversion': MeanReversionStrategy,
        'TrendFollowing': TrendFollowingStrategy,
        'PairsTrading': PairsTradingStrategy,
        'Momentum': MomentumStrategy
    }
    
    # Get data
    logging.info("Fetching market data...")
    try:
        data_dict = gd.get_sp500_data()
        if data_limit and data_limit > 0:
            # Limit data for testing
            tickers = list(data_dict.keys())[:data_limit]
            data_dict = {t: data_dict[t] for t in tickers}
        logging.info(f"Data retrieved for {len(data_dict)} tickers")
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None
    
    # Create output directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    if mode == 'single':
        # Run a single strategy
        if strategy_name not in all_strategies:
            logging.error(f"Strategy '{strategy_name}' not found")
            return None
            
        from TradingSim import run_single_strategy
        result = run_single_strategy(
            all_strategies[strategy_name],
            initial_cash=initial_cash,
            commission=commission,
            data_dict=data_dict,
            plot=True
        )
        
        # Save result to CSV
        if result:
            pd.DataFrame([result]).to_csv(f"{output_dir}/result_{strategy_name}.csv", index=False)
            logging.info(f"Results saved to {output_dir}/result_{strategy_name}.csv")
        
        return result
        
    elif mode == 'compare':
        # Run and compare all strategies
        from TradingSim import compare_strategies
        
        if strategy_name:
            # Compare specific strategies if provided
            strategy_names = strategy_name.split(',')
            strategies = [all_strategies[s] for s in strategy_names if s in all_strategies]
        else:
            # Compare all strategies
            strategies = list(all_strategies.values())
        
        results = compare_strategies(
            strategies,
            initial_cash=initial_cash,
            commission=commission,
            data_dict=data_dict
        )
        
        if results is not None:
            # Save detailed results
            results.to_csv(f"{output_dir}/strategy_comparison.csv", index=False)
            logging.info(f"Comparison results saved to {output_dir}/strategy_comparison.csv")
        
        return results
        
    elif mode == 'adaptive':
        # Run in adaptive mode using strategy selector
        evaluator = StrategyEvaluator(list(all_strategies.values()), initial_cash=initial_cash)
        
        # Initialize with historical data
        evaluator.initialize_with_historical_data(data_dict)
        
        # Evaluate strategies in current market conditions
        evaluation = evaluator.evaluate_strategies(data_dict)
        recommendation = evaluator.get_strategy_recommendation()
        
        # Save recommendation
        with open(f"{output_dir}/strategy_recommendation.txt", "w") as f:
            f.write(f"Current Market Regime: {recommendation['current_regime']}\n")
            f.write(f"Top Recommended Strategy: {recommendation['top_strategy']}\n\n")
            f.write("Recommended Allocations:\n")
            for strategy, allocation in recommendation['allocations'].items():
                f.write(f"  {strategy}: {allocation:.2%}\n")
        
        # Create pie chart of allocations
        plt.figure(figsize=(10, 8))
        plt.pie(
            recommendation['allocations'].values(),
            labels=recommendation['allocations'].keys(),
            autopct='%1.1f%%',
            explode=[0.05] * len(recommendation['allocations']),
            shadow=True
        )
        plt.title('Recommended Strategy Allocations')
        plt.savefig(f"{output_dir}/strategy_allocations.png")
        
        # Run backtest with top strategy or using the allocations
        if not adaptive:
            # Just run the top strategy
            top_strategy = recommendation['top_strategy']
            strategy_class = None
            for s in all_strategies.values():
                if s.__name__ == top_strategy:
                    strategy_class = s
                    break
            
            if strategy_class:
                from TradingSim import run_single_strategy
                result = run_single_strategy(
                    strategy_class,
                    initial_cash=initial_cash,
                    commission=commission,
                    data_dict=data_dict,
                    plot=True
                )
                
                if result:
                    pd.DataFrame([result]).to_csv(f"{output_dir}/result_{top_strategy}.csv", index=False)
                    
                return result
        else:
            # TODO: Implement a combined strategy that allocates capital according to recommendations
            logging.info("Fully adaptive mode with capital allocation not yet implemented")
            return recommendation
    
    else:
        logging.error(f"Invalid mode: {mode}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trading Simulation System')
    parser.add_argument('--mode', type=str, choices=['single', 'compare', 'adaptive'], 
                        default='compare', help='Simulation mode')
    parser.add_argument('--strategy', type=str, help='Strategy name(s) to run')
    parser.add_argument('--cash', type=float, default=10000, help='Initial capital')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate')
    parser.add_argument('--data_limit', type=int, help='Limit number of tickers (for testing)')
    parser.add_argument('--adaptive', action='store_true', help='Use adaptive strategy selection')
    
    args = parser.parse_args()
    
    # Run the simulation
    results = run_trading_simulation(
        mode=args.mode,
        strategy_name=args.strategy,
        initial_cash=args.cash,
        commission=args.commission,
        data_limit=args.data_limit,
        adaptive=args.adaptive
    )
    
    if results is not None:
        if isinstance(results, pd.DataFrame):
            print("\nStrategy Comparison Results:")
            print(results)
        elif isinstance(results, dict) and 'allocations' in results:
            print("\nStrategy Recommendations:")
            print(f"Current Market Regime: {results['current_regime']}")
            print(f"Top Strategy: {results['top_strategy']}")
            print("\nAllocations:")
            for strat, alloc in results['allocations'].items():
                print(f"  {strat}: {alloc:.2%}")
        else:
            print("\nSimulation Results:")
            print(results)