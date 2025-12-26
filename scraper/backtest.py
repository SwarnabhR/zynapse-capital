"""
Backtesting Engine for Zynapse Capital
Simulates trading strategies and calculates performance metrics
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging


class BacktestEngine:
    """
    Backtesting engine for strategy simulation and performance analysis
    """
    
    def __init__(self, initial_capital: float = 1000000, 
                 position_size: float = 0.05,
                 risk_free_rate: float = 0.065):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital: Starting portfolio value (default: ₹10 lakhs)
            position_size: Percentage of capital per position (default: 5%)
            risk_free_rate: Annual risk-free rate for Sharpe ratio (default: 6.5% - Indian T-Bill rate)
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.risk_free_rate = risk_free_rate
        self.daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
        
        self.logger = logging.getLogger('BacktestEngine')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def run_backtest(self, df: pd.DataFrame, strategy: str = 'momentum') -> Tuple[pd.DataFrame, Dict]:
        """
        Run backtest simulation
        
        Args:
            df: DataFrame with analyzed data (must have date, symbol, close, signal columns)
            strategy: Strategy type ('momentum', 'quality', 'low_risk')
        
        Returns:
            Tuple of (trades_df, performance_metrics)
        """
        self.logger.info(f"🎯 Running backtest - Strategy: {strategy}")
        self.logger.info(f"   Initial Capital: ₹{self.initial_capital:,.0f}")
        self.logger.info(f"   Position Size: {self.position_size*100}%")
        
        # Prepare data
        df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
        dates = sorted(df['date'].unique())
        
        self.logger.info(f"   Date Range: {dates[0]} to {dates[-1]}")
        self.logger.info(f"   Trading Days: {len(dates)}")
        
        # Initialize portfolio
        portfolio = {
            'cash': self.initial_capital,
            'positions': {},  # {symbol: {'shares': qty, 'entry_price': price, 'entry_date': date}}
            'portfolio_value': [self.initial_capital],
            'dates': [dates[0]],
            'trades': []
        }
        
        # Simulate trading day by day
        for i, date in enumerate(dates):
            day_data = df[df['date'] == date].copy()
            
            # Apply strategy filter
            buy_candidates = self._filter_by_strategy(day_data, strategy, 'BUY')
            sell_signals = self._filter_by_strategy(day_data, strategy, 'SELL')
            
            # Execute sells first (free up capital)
            self._execute_sells(portfolio, sell_signals, date)
            
            # Execute buys
            self._execute_buys(portfolio, buy_candidates, date)
            
            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(portfolio, day_data, date)
            portfolio['portfolio_value'].append(portfolio_value)
            portfolio['dates'].append(date)
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(portfolio['trades'])
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(portfolio, trades_df)
        
        self.logger.info(f"✅ Backtest complete!")
        self.logger.info(f"   Total Trades: {len(trades_df)}")
        self.logger.info(f"   Final Portfolio Value: ₹{portfolio['portfolio_value'][-1]:,.0f}")
        self.logger.info(f"   Total Return: {performance['total_return_pct']:.2f}%")
        
        return trades_df, performance
    
    def _filter_by_strategy(self, df: pd.DataFrame, strategy: str, signal_type: str) -> pd.DataFrame:
        """Filter stocks based on strategy criteria"""
        
        if signal_type == 'BUY':
            if strategy == 'momentum':
                # High momentum, low risk
                filtered = df[
                    (df['momentum_score'] > 70) & 
                    (df['risk_score'] < 30)
                ].nlargest(10, 'momentum_score')
            
            elif strategy == 'quality':
                # High delivery, positive returns, low risk
                filtered = df[
                    (df['delivery_pct'] > 70) & 
                    (df['day_return_pct'] > 0) &
                    (df['risk_score'] < 20) &
                    (df['momentum_score'] > 60)
                ].nlargest(10, 'composite_score')
            
            elif strategy == 'low_risk':
                # Minimize risk, moderate momentum
                filtered = df[
                    (df['risk_score'] < 15) & 
                    (df['momentum_score'] > 55)
                ].nlargest(10, 'momentum_score')
            
            else:
                # Default: use signals from analytics
                filtered = df[df['signal'].isin(['BUY', 'STRONG_BUY'])]
        
        else:  # SELL
            # Sell on high risk or negative signals
            filtered = df[
                (df['risk_score'] > 50) | 
                (df['signal'] == 'SELL') |
                (df['day_return_pct'] < -5)
            ]
        
        return filtered
    
    def _execute_buys(self, portfolio: Dict, candidates: pd.DataFrame, date: str):
        """Execute buy orders"""
        
        if len(candidates) == 0:
            return
        
        # Calculate position size
        available_cash = portfolio['cash']
        max_positions = int(1 / self.position_size)  # e.g., 5% = 20 max positions
        current_positions = len(portfolio['positions'])
        
        # Limit new positions
        max_new_positions = min(
            max_positions - current_positions,
            int(available_cash / (self.initial_capital * self.position_size))
        )
        
        if max_new_positions <= 0:
            return
        
        # Buy top candidates
        for idx, row in candidates.head(max_new_positions).iterrows():
            symbol = row['symbol']
            price = row['close']
            
            # Skip if already holding
            if symbol in portfolio['positions']:
                continue
            
            # Calculate shares to buy
            position_value = self.initial_capital * self.position_size
            shares = int(position_value / price)
            
            if shares == 0:
                continue
            
            cost = shares * price
            
            if cost <= portfolio['cash']:
                # Execute buy
                portfolio['cash'] -= cost
                portfolio['positions'][symbol] = {
                    'shares': shares,
                    'entry_price': price,
                    'entry_date': date
                }
                
                portfolio['trades'].append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': shares,
                    'price': price,
                    'value': cost,
                    'momentum_score': row.get('momentum_score', np.nan),
                    'risk_score': row.get('risk_score', np.nan)
                })
    
    def _execute_sells(self, portfolio: Dict, signals: pd.DataFrame, date: str):
        """Execute sell orders"""
        
        if len(signals) == 0:
            return
        
        sell_symbols = signals['symbol'].tolist()
        
        # Sell positions that have sell signals
        for symbol in list(portfolio['positions'].keys()):
            if symbol in sell_symbols:
                position = portfolio['positions'][symbol]
                
                # Find current price
                price_row = signals[signals['symbol'] == symbol]
                if len(price_row) == 0:
                    continue
                
                price = price_row.iloc[0]['close']
                shares = position['shares']
                proceeds = shares * price
                
                # Calculate P&L
                cost_basis = shares * position['entry_price']
                pnl = proceeds - cost_basis
                pnl_pct = (pnl / cost_basis) * 100
                
                # Execute sell
                portfolio['cash'] += proceeds
                del portfolio['positions'][symbol]
                
                portfolio['trades'].append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': shares,
                    'price': price,
                    'value': proceeds,
                    'entry_price': position['entry_price'],
                    'entry_date': position['entry_date'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'hold_days': (pd.to_datetime(date) - pd.to_datetime(position['entry_date'])).days
                })
    
    def _calculate_portfolio_value(self, portfolio: Dict, day_data: pd.DataFrame, date: str) -> float:
        """Calculate total portfolio value"""
        
        total_value = portfolio['cash']
        
        for symbol, position in portfolio['positions'].items():
            # Find current price
            price_row = day_data[day_data['symbol'] == symbol]
            
            if len(price_row) > 0:
                current_price = price_row.iloc[0]['close']
                position_value = position['shares'] * current_price
                total_value += position_value
        
        return total_value
    
    def _calculate_performance_metrics(self, portfolio: Dict, trades_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        metrics = {}
        
        # Portfolio value series
        values = np.array(portfolio['portfolio_value'])
        dates = portfolio['dates']
        
        # 1. Total Return
        metrics['initial_capital'] = self.initial_capital
        metrics['final_capital'] = values[-1]
        metrics['total_return'] = values[-1] - self.initial_capital
        metrics['total_return_pct'] = ((values[-1] / self.initial_capital) - 1) * 100
        
        # 2. Daily Returns
        daily_returns = np.diff(values) / values[:-1]
        metrics['avg_daily_return'] = np.mean(daily_returns)
        metrics['daily_volatility'] = np.std(daily_returns)
        
        # 3. Annualized Metrics
        trading_days = len(values) - 1
        years = trading_days / 252
        metrics['annualized_return'] = ((values[-1] / self.initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        metrics['annualized_volatility'] = metrics['daily_volatility'] * np.sqrt(252) * 100
        
        # 4. Sharpe Ratio
        excess_returns = daily_returns - self.daily_rf_rate
        metrics['sharpe_ratio'] = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        # 5. Maximum Drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        metrics['max_drawdown_pct'] = np.min(drawdown) * 100
        metrics['max_drawdown_value'] = np.min(values - peak)
        
        # 6. Trade Statistics
        if len(trades_df) > 0:
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            
            if len(sell_trades) > 0:
                metrics['total_trades'] = len(sell_trades)
                metrics['winning_trades'] = len(sell_trades[sell_trades['pnl'] > 0])
                metrics['losing_trades'] = len(sell_trades[sell_trades['pnl'] < 0])
                metrics['win_rate'] = (metrics['winning_trades'] / metrics['total_trades']) * 100
                
                metrics['avg_win'] = sell_trades[sell_trades['pnl'] > 0]['pnl'].mean() if metrics['winning_trades'] > 0 else 0
                metrics['avg_loss'] = sell_trades[sell_trades['pnl'] < 0]['pnl'].mean() if metrics['losing_trades'] > 0 else 0
                metrics['avg_win_pct'] = sell_trades[sell_trades['pnl'] > 0]['pnl_pct'].mean() if metrics['winning_trades'] > 0 else 0
                metrics['avg_loss_pct'] = sell_trades[sell_trades['pnl'] < 0]['pnl_pct'].mean() if metrics['losing_trades'] > 0 else 0
                
                metrics['profit_factor'] = abs(sell_trades[sell_trades['pnl'] > 0]['pnl'].sum() / 
                                              sell_trades[sell_trades['pnl'] < 0]['pnl'].sum()) if metrics['losing_trades'] > 0 else np.inf
                
                metrics['avg_hold_days'] = sell_trades['hold_days'].mean()
            else:
                metrics['total_trades'] = 0
        
        return metrics
    
    def generate_report(self, performance: Dict, trades_df: pd.DataFrame, 
                       output_path: Optional[Path] = None) -> str:
        """Generate detailed backtest report"""
        
        report = f"""
{'='*80}
BACKTEST PERFORMANCE REPORT
{'='*80}

Portfolio Summary:
  Initial Capital:        ₹{performance['initial_capital']:,.0f}
  Final Capital:          ₹{performance['final_capital']:,.0f}
  Total P&L:              ₹{performance['total_return']:,.0f}
  Total Return:           {performance['total_return_pct']:.2f}%
  Annualized Return:      {performance['annualized_return']:.2f}%

Risk Metrics:
  Daily Volatility:       {performance['daily_volatility']*100:.2f}%
  Annualized Volatility:  {performance['annualized_volatility']:.2f}%
  Maximum Drawdown:       {performance['max_drawdown_pct']:.2f}%
  Sharpe Ratio:           {performance['sharpe_ratio']:.2f}

Trade Statistics:
  Total Trades:           {performance.get('total_trades', 0)}
  Winning Trades:         {performance.get('winning_trades', 0)}
  Losing Trades:          {performance.get('losing_trades', 0)}
  Win Rate:               {performance.get('win_rate', 0):.2f}%
  Avg Win:                ₹{performance.get('avg_win', 0):,.0f} ({performance.get('avg_win_pct', 0):.2f}%)
  Avg Loss:               ₹{performance.get('avg_loss', 0):,.0f} ({performance.get('avg_loss_pct', 0):.2f}%)
  Profit Factor:          {performance.get('profit_factor', 0):.2f}
  Avg Hold Period:        {performance.get('avg_hold_days', 0):.1f} days

Sharpe Ratio Interpretation:
"""
        
        # Sharpe ratio interpretation
        sharpe = performance['sharpe_ratio']
        if sharpe < 0:
            report += "  ❌ Poor (Below risk-free rate)\n"
        elif sharpe < 1.0:
            report += "  ⚠️  Sub-optimal\n"
        elif sharpe < 2.0:
            report += "  ✅ Good\n"
        elif sharpe < 3.0:
            report += "  🎯 Very Good\n"
        else:
            report += "  🏆 Outstanding\n"
        
        report += f"\n{'='*80}\n"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            self.logger.info(f"📄 Report saved to {output_path}")
        
        return report