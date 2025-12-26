"""
Backtest Runner for Zynapse Capital
Run strategy backtests and generate performance reports
"""
import pandas as pd
import argparse
from pathlib import Path
from backtest import BacktestEngine
import matplotlib.pyplot as plt


def run_backtest_analysis(input_file: str, strategy: str = 'momentum',
                         initial_capital: float = 1000000,
                         position_size: float = 0.05,
                         output_dir: str = 'data/backtest'):
    """
    Run complete backtest analysis
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*80}")
    print(f"🎯 ZYNAPSE CAPITAL - BACKTESTING ENGINE")
    print(f"{'='*80}")
    print(f"Strategy: {strategy.upper()}")
    print(f"Input: {input_file}")
    print(f"Initial Capital: ₹{initial_capital:,.0f}")
    print(f"Position Size: {position_size*100}%")
    print(f"{'='*80}\n")
    
    # Load analyzed data
    print("📂 Loading analyzed data...")
    df = pd.read_csv(input_file)
    
    # Verify required columns
    required_cols = ['date', 'symbol', 'close', 'momentum_score', 'risk_score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Error: Missing required columns: {missing_cols}")
        print("   Run analytics first: python scraper/analyze.py --input <file>")
        return
    
    print(f"✅ Loaded {len(df):,} records")
    print(f"   Symbols: {df['symbol'].nunique()}")
    print(f"   Date Range: {df['date'].min()} to {df['date'].max()}")
    
    # Initialize backtest engine
    engine = BacktestEngine(
        initial_capital=initial_capital,
        position_size=position_size
    )
    
    # Run backtest
    print(f"\n{'='*80}")
    print("RUNNING BACKTEST")
    print(f"{'='*80}\n")
    
    trades_df, performance = engine.run_backtest(df, strategy=strategy)
    
    # Generate report
    print(f"\n{'='*80}")
    print("PERFORMANCE REPORT")
    print(f"{'='*80}")
    
    report = engine.generate_report(
        performance, 
        trades_df,
        output_path / f'backtest_report_{strategy}.txt'
    )
    print(report)
    
    # Save trades
    if len(trades_df) > 0:
        trades_file = output_path / f'trades_{strategy}.csv'
        trades_df.to_csv(trades_file, index=False)
        print(f"💾 Trades saved to: {trades_file}")
        
        # Show sample trades
        print(f"\n📊 SAMPLE TRADES:")
        sell_trades = trades_df[trades_df['action'] == 'SELL'].head(10)
        if len(sell_trades) > 0:
            print(sell_trades[['date', 'symbol', 'action', 'price', 'pnl', 'pnl_pct', 'hold_days']].to_string(index=False))
    
    # Save equity curve data
    equity_curve = pd.DataFrame({
        'date': performance.get('dates', []),
        'portfolio_value': performance.get('portfolio_values', [])
    })
    
    # Generate comparison report
    print(f"\n{'='*80}")
    print("STRATEGY PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"Strategy:            {strategy.upper()}")
    print(f"Total Return:        {performance['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio:        {performance['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:        {performance['max_drawdown_pct']:.2f}%")
    print(f"Win Rate:            {performance.get('win_rate', 0):.2f}%")
    print(f"Profit Factor:       {performance.get('profit_factor', 0):.2f}")
    print(f"{'='*80}\n")
    
    return trades_df, performance


def compare_strategies(input_file: str, output_dir: str = 'data/backtest'):
    """
    Compare multiple strategies
    """
    strategies = ['momentum', 'quality', 'low_risk']
    
    print(f"{'='*80}")
    print(f"🔬 STRATEGY COMPARISON")
    print(f"{'='*80}\n")
    
    results = []
    
    for strategy in strategies:
        print(f"\nTesting Strategy: {strategy.upper()}")
        print("-" * 80)
        
        engine = BacktestEngine()
        df = pd.read_csv(input_file)
        trades_df, performance = engine.run_backtest(df, strategy=strategy)
        
        results.append({
            'strategy': strategy,
            'total_return_pct': performance['total_return_pct'],
            'annualized_return': performance['annualized_return'],
            'sharpe_ratio': performance['sharpe_ratio'],
            'max_drawdown_pct': performance['max_drawdown_pct'],
            'win_rate': performance.get('win_rate', 0),
            'total_trades': performance.get('total_trades', 0),
            'profit_factor': performance.get('profit_factor', 0)
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    output_path = Path(output_dir)
    comparison_file = output_path / 'strategy_comparison.csv'
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n💾 Comparison saved to: {comparison_file}")
    
    # Find best strategy
    print(f"\n🏆 BEST STRATEGIES:")
    print(f"   Highest Return:     {comparison_df.loc[comparison_df['total_return_pct'].idxmax(), 'strategy'].upper()}")
    print(f"   Best Sharpe Ratio:  {comparison_df.loc[comparison_df['sharpe_ratio'].idxmax(), 'strategy'].upper()}")
    print(f"   Lowest Drawdown:    {comparison_df.loc[comparison_df['max_drawdown_pct'].idxmax(), 'strategy'].upper()}")
    print(f"{'='*80}\n")
    
    return comparison_df


def main():
    parser = argparse.ArgumentParser(description='Run trading strategy backtest')
    parser.add_argument('--input', required=True, help='Analyzed data CSV file')
    parser.add_argument('--strategy', default='momentum', 
                       choices=['momentum', 'quality', 'low_risk'],
                       help='Trading strategy to test')
    parser.add_argument('--capital', type=float, default=1000000,
                       help='Initial capital (default: 1000000)')
    parser.add_argument('--position-size', type=float, default=0.05,
                       help='Position size as decimal (default: 0.05 = 5%%)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all strategies')
    parser.add_argument('--output', default='data/backtest',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare all strategies
        compare_strategies(args.input, args.output)
    else:
        # Run single strategy
        run_backtest_analysis(
            args.input,
            strategy=args.strategy,
            initial_capital=args.capital,
            position_size=args.position_size,
            output_dir=args.output
        )
    
    print("✅ BACKTEST COMPLETE!\n")


if __name__ == '__main__':
    main()