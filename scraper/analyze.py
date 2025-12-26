"""
NSE Data Analysis Runner
Combines parser, validator, and analytics for comprehensive analysis
"""
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from parser import NSEParser
from validator import ValidationEngine
from analytics import AnalyticsEngine


def analyze_dataset(input_file: str, output_dir: str = 'data/analytics'):
    """
    Run comprehensive analysis on parsed dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*80}")
    print(f"📊 NSE DATA ANALYTICS")
    print(f"{'='*80}")
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    # Load data
    print("📂 Loading data...")
    df = pd.read_csv(input_file)
    print(f"✅ Loaded {len(df):,} records, {df['symbol'].nunique()} unique symbols")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Initialize engines
    analytics = AnalyticsEngine()
    
    # Step 1: Calculate technical indicators
    print(f"\n{'='*80}")
    print("STEP 1: Technical Indicators")
    print(f"{'='*80}")
    df = analytics.calculate_technical_indicators(df)
    
    # Step 2: Calculate momentum scores
    print(f"\n{'='*80}")
    print("STEP 2: Momentum Scoring")
    print(f"{'='*80}")
    df = analytics.calculate_momentum_score(df)
    
    # Step 3: Calculate risk scores
    print(f"\n{'='*80}")
    print("STEP 3: Risk Scoring")
    print(f"{'='*80}")
    df = analytics.calculate_risk_score(df)
    
    # Step 4: Detect patterns
    print(f"\n{'='*80}")
    print("STEP 4: Pattern Detection")
    print(f"{'='*80}")
    df = analytics.detect_patterns(df)
    
    # Step 5: Generate rankings (for latest date)
    print(f"\n{'='*80}")
    print("STEP 5: Stock Rankings")
    print(f"{'='*80}")
    latest_date = df['date'].max()
    df = analytics.generate_ranking(df, date=latest_date)
    
    # Save full analyzed dataset
    output_file = output_path / f"analyzed_{Path(input_file).stem}.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved analyzed dataset: {output_file}")
    
    # Generate insights reports
    generate_insights(df, output_path, latest_date)
    
    return df


def generate_insights(df: pd.DataFrame, output_path: Path, date: str):
    """Generate actionable insights and reports"""
    
    print(f"\n{'='*80}")
    print(f"📈 GENERATING INSIGHTS FOR {date}")
    print(f"{'='*80}\n")
    
    latest_df = df[df['date'] == date].copy()
    
    # Report 1: Top Momentum Stocks
    print("🚀 TOP 10 MOMENTUM STOCKS (High Score + Low Risk):")
    top_momentum = latest_df.nlargest(10, 'composite_score')[
        ['symbol', 'close', 'momentum_score', 'risk_score', 'composite_score', 
         'delivery_pct', 'day_return_pct', 'signal']
    ]
    print(top_momentum.to_string(index=False))
    top_momentum.to_csv(output_path / f'top_momentum_{date}.csv', index=False)
    
    # Report 2: Strong Buy Signals
    strong_buys = latest_df[latest_df['signal'] == 'STRONG_BUY']
    if len(strong_buys) > 0:
        print(f"\n💎 STRONG BUY SIGNALS ({len(strong_buys)} stocks):")
        print(strong_buys[['symbol', 'close', 'momentum_score', 'risk_score', 
                           'pattern', 'delivery_pct', 'volume_ratio']].to_string(index=False))
        strong_buys.to_csv(output_path / f'strong_buys_{date}.csv', index=False)
    
    # Report 3: Technical Patterns
    patterns = latest_df[latest_df['pattern'] != '']
    if len(patterns) > 0:
        print(f"\n🔍 PATTERN DETECTIONS ({len(patterns)} stocks):")
        print(patterns[['symbol', 'close', 'pattern', 'signal', 'momentum_score']].head(10).to_string(index=False))
        patterns.to_csv(output_path / f'patterns_{date}.csv', index=False)
    
    # Report 4: High Risk Stocks to Avoid
    print(f"\n⚠️  TOP 10 HIGH RISK STOCKS (Avoid):")
    high_risk = latest_df.nlargest(10, 'risk_score')[
        ['symbol', 'close', 'risk_score', 'momentum_score', 'validation_flags', 
         'delivery_pct', 'day_return_pct']
    ]
    print(high_risk.to_string(index=False))
    high_risk.to_csv(output_path / f'high_risk_{date}.csv', index=False)
    
    # Report 5: Quality Accumulation (High delivery, positive momentum, low risk)
    quality_stocks = latest_df[
        (latest_df['delivery_pct'] > 70) & 
        (latest_df['momentum_score'] > 60) &
        (latest_df['risk_score'] < 30)
    ].sort_values('composite_score', ascending=False)
    
    if len(quality_stocks) > 0:
        print(f"\n💰 QUALITY ACCUMULATION ({len(quality_stocks)} stocks):")
        print(quality_stocks[['symbol', 'close', 'momentum_score', 'risk_score', 
                              'delivery_pct', 'day_return_pct']].head(10).to_string(index=False))
        quality_stocks.to_csv(output_path / f'quality_stocks_{date}.csv', index=False)
    
    # Report 6: RSI Analysis
    if 'rsi_14' in latest_df.columns:
        oversold = latest_df[latest_df['rsi_14'] < 30].sort_values('rsi_14')
        overbought = latest_df[latest_df['rsi_14'] > 70].sort_values('rsi_14', ascending=False)
        
        if len(oversold) > 0:
            print(f"\n📉 RSI OVERSOLD (<30) - Potential Bounce ({len(oversold)} stocks):")
            print(oversold[['symbol', 'close', 'rsi_14', 'delivery_pct', 'momentum_score']].head(10).to_string(index=False))
        
        if len(overbought) > 0:
            print(f"\n📈 RSI OVERBOUGHT (>70) - Potential Reversal ({len(overbought)} stocks):")
            print(overbought[['symbol', 'close', 'rsi_14', 'delivery_pct', 'risk_score']].head(10).to_string(index=False))
    
    # Summary Statistics
    print(f"\n{'='*80}")
    print("📊 SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total Stocks Analyzed: {len(latest_df)}")
    print(f"\nSignal Distribution:")
    print(latest_df['signal'].value_counts())
    print(f"\nMomentum Score Stats:")
    print(f"  Mean: {latest_df['momentum_score'].mean():.2f}")
    print(f"  Median: {latest_df['momentum_score'].median():.2f}")
    print(f"  Std Dev: {latest_df['momentum_score'].std():.2f}")
    print(f"\nRisk Score Stats:")
    print(f"  Mean: {latest_df['risk_score'].mean():.2f}")
    print(f"  Median: {latest_df['risk_score'].median():.2f}")
    print(f"  Std Dev: {latest_df['risk_score'].std():.2f}")
    
    print(f"\n✅ All reports saved to {output_path}/")


def main():
    parser = argparse.ArgumentParser(description='Analyze NSE parsed data')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', default='data/analytics', help='Output directory')
    
    args = parser.parse_args()
    
    # Run analysis
    df = analyze_dataset(args.input, args.output)
    
    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()