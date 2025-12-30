"""
Export & Reports Page - Zynapse Capital
Export data and generate comprehensive reports
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime
import io
import base64

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Page config
st.set_page_config(
    page_title="Export & Reports - Zynapse Capital",
    page_icon="📥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .export-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data():
    """Load latest analyzed data"""
    analytics_dir = Path('data/analytics')
    
    if not analytics_dir.exists():
        return None, None
    
    analyzed_files = list(analytics_dir.glob('analyzed_equity_full_*.csv'))
    
    if not analyzed_files:
        return None, None
    
    latest_file = max(analyzed_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file)
    
    # Fill NaN values
    numeric_cols = ['momentum_score', 'risk_score', 'delivery_pct', 'day_return_pct', 
                    'volume', 'composite_score', 'close']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    if 'signal' in df.columns:
        df['signal'] = df['signal'].fillna('HOLD')
    
    return df, latest_file.name

def create_excel_report(df, filename="zynapse_report.xlsx"):
    """Create multi-sheet Excel report"""
    
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Full Data
        df.to_excel(writer, sheet_name='Full Data', index=False)
        
        # Sheet 2: Summary Statistics
        summary_cols = ['momentum_score', 'risk_score', 'delivery_pct', 
                       'day_return_pct', 'volume', 'composite_score']
        summary_cols = [col for col in summary_cols if col in df.columns]
        
        if summary_cols:
            summary = df[summary_cols].describe()
            summary.to_excel(writer, sheet_name='Summary Statistics')
        
        # Sheet 3: Top Performers
        if len(df) > 0:
            top_momentum = df.nlargest(50, 'momentum_score')
            top_momentum.to_excel(writer, sheet_name='Top Momentum', index=False)
        
        # Sheet 4: Quality Stocks
        quality = df[
            (df['delivery_pct'] > 70) & 
            (df['risk_score'] < 20) &
            (df['momentum_score'] > 60)
        ]
        if len(quality) > 0:
            quality.to_excel(writer, sheet_name='Quality Stocks', index=False)
        
        # Sheet 5: Buy Signals
        buy_signals = df[df['signal'].isin(['BUY', 'STRONG_BUY'])]
        if len(buy_signals) > 0:
            buy_signals.to_excel(writer, sheet_name='Buy Signals', index=False)
        
        # Sheet 6: Signal Distribution
        signal_dist = df['signal'].value_counts().reset_index()
        signal_dist.columns = ['Signal', 'Count']
        signal_dist.to_excel(writer, sheet_name='Signal Distribution', index=False)
    
    output.seek(0)
    return output

def create_text_report(df, date):
    """Create detailed text report"""
    
    report = f"""
{'='*80}
ZYNAPSE CAPITAL - MARKET ANALYSIS REPORT
{'='*80}

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Date: {date}

{'='*80}
EXECUTIVE SUMMARY
{'='*80}

Total Securities Analyzed: {len(df):,}
Date Range: {df['date'].min()} to {df['date'].max()}

{'='*80}
MARKET OVERVIEW
{'='*80}

Average Momentum Score:    {df['momentum_score'].mean():.2f} (Baseline: 60.00)
Average Risk Score:        {df['risk_score'].mean():.2f} (Threshold: 30.00)
Average Delivery %:        {df['delivery_pct'].mean():.2f}%
Average Daily Return:      {df['day_return_pct'].mean():.2f}%

Market Breadth:
  - Gainers:  {len(df[df['day_return_pct'] > 0]):,} stocks ({(len(df[df['day_return_pct'] > 0])/len(df)*100):.1f}%)
  - Losers:   {len(df[df['day_return_pct'] < 0]):,} stocks ({(len(df[df['day_return_pct'] < 0])/len(df)*100):.1f}%)
  - Unchanged: {len(df[df['day_return_pct'] == 0]):,} stocks

{'='*80}
TRADING SIGNALS DISTRIBUTION
{'='*80}

"""
    
    signal_counts = df['signal'].value_counts()
    for signal, count in signal_counts.items():
        pct = (count / len(df)) * 100
        report += f"{signal:15s}: {count:5,} stocks ({pct:5.1f}%)\n"
    
    report += f"""
{'='*80}
TOP 10 MOMENTUM LEADERS
{'='*80}

"""
    
    top_momentum = df.nlargest(10, 'momentum_score')[
        ['symbol', 'close', 'momentum_score', 'risk_score', 'delivery_pct', 'day_return_pct', 'signal']
    ]
    
    report += "Symbol        Close    Momentum  Risk   Delivery  Return   Signal\n"
    report += "-" * 75 + "\n"
    
    for _, row in top_momentum.iterrows():
        report += f"{row['symbol']:12s}  ₹{row['close']:7.2f}  {row['momentum_score']:6.1f}  {row['risk_score']:5.1f}  {row['delivery_pct']:6.1f}%  {row['day_return_pct']:+6.2f}%  {row['signal']}\n"
    
    report += f"""
{'='*80}
QUALITY STOCK OPPORTUNITIES
{'='*80}

Criteria: Delivery >70%, Momentum >60, Risk <20

"""
    
    quality = df[
        (df['delivery_pct'] > 70) & 
        (df['momentum_score'] > 60) &
        (df['risk_score'] < 20)
    ].nlargest(10, 'composite_score')
    
    report += f"Found {len(quality)} quality opportunities\n\n"
    
    if len(quality) > 0:
        report += "Top 10 Quality Stocks:\n"
        report += "Symbol        Close    Momentum  Risk   Delivery  Signal\n"
        report += "-" * 65 + "\n"
        
        for _, row in quality.iterrows():
            report += f"{row['symbol']:12s}  ₹{row['close']:7.2f}  {row['momentum_score']:6.1f}  {row['risk_score']:5.1f}  {row['delivery_pct']:6.1f}%  {row['signal']}\n"
    
    report += f"""
{'='*80}
HIGH RISK STOCKS (AVOID)
{'='*80}

"""
    
    high_risk = df.nlargest(10, 'risk_score')[
        ['symbol', 'close', 'risk_score', 'momentum_score', 'delivery_pct', 'day_return_pct']
    ]
    
    report += "Symbol        Close    Risk   Momentum  Delivery  Return\n"
    report += "-" * 65 + "\n"
    
    for _, row in high_risk.iterrows():
        report += f"{row['symbol']:12s}  ₹{row['close']:7.2f}  {row['risk_score']:5.1f}  {row['momentum_score']:6.1f}  {row['delivery_pct']:6.1f}%  {row['day_return_pct']:+6.2f}%\n"
    
    report += f"""
{'='*80}
RISK METRICS
{'='*80}

Stocks by Risk Category:
  - Very Low Risk (<10):    {len(df[df['risk_score'] < 10]):,} stocks
  - Low Risk (10-20):       {len(df[df['risk_score'].between(10, 20)]):,} stocks
  - Medium Risk (20-40):    {len(df[df['risk_score'].between(20, 40)]):,} stocks
  - High Risk (40-60):      {len(df[df['risk_score'].between(40, 60)]):,} stocks
  - Very High Risk (>60):   {len(df[df['risk_score'] > 60]):,} stocks

{'='*80}
DELIVERY ANALYSIS
{'='*80}

Stocks by Delivery Category:
  - Very Strong (>70%):     {len(df[df['delivery_pct'] > 70]):,} stocks
  - Strong (50-70%):        {len(df[df['delivery_pct'].between(50, 70)]):,} stocks
  - Mixed (30-50%):         {len(df[df['delivery_pct'].between(30, 50)]):,} stocks
  - Speculative (<30%):     {len(df[df['delivery_pct'] < 30]):,} stocks

{'='*80}
DISCLAIMER
{'='*80}

This report is for informational and educational purposes only.
It does not constitute financial advice or a recommendation to buy/sell securities.
Trading in financial markets involves substantial risk of loss.
Always conduct your own research and consult with a qualified financial advisor.

{'='*80}
END OF REPORT
{'='*80}
"""
    
    return report

def main():
    # Title
    st.title("📥 Export & Reports")
    st.markdown("**Download Analysis Results and Generate Reports**")
    
    # Load data
    df, filename = load_data()
    
    if df is None:
        st.error("⚠️ No data found. Please download and analyze data from the home page.")
        if st.button("🏠 Go to Home"):
            st.switch_page("app.py")
        return
    
    # Get latest date
    unique_dates = sorted(df['date'].unique())
    latest_date = unique_dates[-1]
    
    st.success(f"✅ Loaded: **{filename}** | {len(df):,} records")
    
    # SIDEBAR
    with st.sidebar:
        st.header("📋 Export Options")
        
        # Date selector
        if len(unique_dates) > 1:
            selected_date = st.selectbox(
                "📅 Select Date",
                options=unique_dates,
                index=len(unique_dates)-1
            )
        else:
            selected_date = latest_date
            st.info(f"📅 Date: {selected_date}")
        
        df_date = df[df['date'] == selected_date].copy()
        
        st.divider()
        
        # Filters
        st.subheader("🔍 Data Filters")
        
        min_momentum = st.slider("Min Momentum", 0, 100, 0)
        max_risk = st.slider("Max Risk", 0, 100, 100)
        min_delivery = st.slider("Min Delivery %", 0, 100, 0)
        
        signals = st.multiselect(
            "Signals",
            ['BUY', 'STRONG_BUY', 'HOLD', 'SELL'],
            default=['BUY', 'STRONG_BUY', 'HOLD', 'SELL']
        )
        
        # Apply filters
        filtered_df = df_date[
            (df_date['momentum_score'] >= min_momentum) &
            (df_date['risk_score'] <= max_risk) &
            (df_date['delivery_pct'] >= min_delivery) &
            (df_date['signal'].isin(signals))
        ].copy()
        
        st.success(f"✅ {len(filtered_df):,} stocks")
        
        st.divider()
        
        # Column selection
        st.subheader("📊 Select Columns")
        
        all_columns = filtered_df.columns.tolist()
        
        preset_columns = st.radio(
            "Preset",
            ["Essential", "All", "Custom"]
        )
        
        if preset_columns == "Essential":
            export_columns = ['symbol', 'close', 'momentum_score', 'risk_score', 
                            'delivery_pct', 'day_return_pct', 'volume', 'signal']
        elif preset_columns == "All":
            export_columns = all_columns
        else:
            export_columns = st.multiselect(
                "Choose Columns",
                options=all_columns,
                default=['symbol', 'close', 'momentum_score', 'risk_score', 
                        'delivery_pct', 'signal']
            )
    
    # Filter export columns
    export_columns = [col for col in export_columns if col in filtered_df.columns]
    export_df = filtered_df[export_columns].copy()
    
    # MAIN CONTENT
    
    # Preview
    st.header("👁️ Data Preview")
    st.dataframe(export_df.head(20), use_container_width=True, height=400)
    
    st.info(f"📊 Exporting {len(export_df):,} rows × {len(export_columns)} columns")
    
    st.divider()
    
    # Export options
    st.header("📥 Download Options")
    
    col1, col2, col3 = st.columns(3)
    
    # CSV Export
    with col1:
        st.markdown("""
        <div class='export-card'>
        <h3>📄 CSV Export</h3>
        <p>Lightweight format for Excel, Python, R</p>
        </div>
        """, unsafe_allow_html=True)
        
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"zynapse_export_{selected_date}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.caption(f"Size: {len(csv.encode('utf-8')) / 1024:.1f} KB")
    
    # Excel Export
    with col2:
        st.markdown("""
        <div class='export-card'>
        <h3>📊 Excel Report</h3>
        <p>Multi-sheet workbook with analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        excel_buffer = create_excel_report(filtered_df)
        
        st.download_button(
            label="📥 Download Excel",
            data=excel_buffer,
            file_name=f"zynapse_report_{selected_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        st.caption(f"Size: {len(excel_buffer.getvalue()) / 1024:.1f} KB")
        st.caption("Sheets: 6 (Full Data, Stats, Top Performers, etc.)")
    
    # Text Report
    with col3:
        st.markdown("""
        <div class='export-card'>
        <h3>📝 Text Report</h3>
        <p>Comprehensive analysis summary</p>
        </div>
        """, unsafe_allow_html=True)
        
        text_report = create_text_report(filtered_df, selected_date)
        
        st.download_button(
            label="📥 Download Report",
            data=text_report,
            file_name=f"zynapse_report_{selected_date}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        st.caption(f"Size: {len(text_report.encode('utf-8')) / 1024:.1f} KB")
    
    st.divider()
    
    # Quick Export Presets
    st.header("⚡ Quick Export Presets")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.subheader("💎 Quality Stocks")
        quality = filtered_df[
            (filtered_df['delivery_pct'] > 70) & 
            (filtered_df['momentum_score'] > 60) &
            (filtered_df['risk_score'] < 20)
        ]
        st.metric("Count", len(quality))
        
        if len(quality) > 0:
            csv = quality.to_csv(index=False)
            st.download_button(
                "📥 Export Quality",
                data=csv,
                file_name=f"quality_stocks_{selected_date}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        st.subheader("🚀 Top Momentum")
        top_momentum = filtered_df.nlargest(50, 'momentum_score')
        st.metric("Count", len(top_momentum))
        
        csv = top_momentum.to_csv(index=False)
        st.download_button(
            "📥 Export Top 50",
            data=csv,
            file_name=f"top_momentum_{selected_date}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        st.subheader("📈 Buy Signals")
        buy_signals = filtered_df[filtered_df['signal'].isin(['BUY', 'STRONG_BUY'])]
        st.metric("Count", len(buy_signals))
        
        if len(buy_signals) > 0:
            csv = buy_signals.to_csv(index=False)
            st.download_button(
                "📥 Export Buys",
                data=csv,
                file_name=f"buy_signals_{selected_date}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col4:
        st.subheader("⚠️ High Risk")
        high_risk = filtered_df.nlargest(50, 'risk_score')
        st.metric("Count", len(high_risk))
        
        csv = high_risk.to_csv(index=False)
        st.download_button(
            "📥 Export High Risk",
            data=csv,
            file_name=f"high_risk_{selected_date}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.divider()
    
    # Report Preview
    st.header("📄 Report Preview")
    
    with st.expander("📝 View Text Report"):
        st.text(create_text_report(filtered_df, selected_date))
    
    # Export history
    st.divider()
    st.header("📂 Available Datasets")
    
    analytics_dir = Path('data/analytics')
    if analytics_dir.exists():
        files = list(analytics_dir.glob('*.csv'))
        
        if files:
            file_data = []
            for file in files:
                size = file.stat().st_size / 1024  # KB
                modified = datetime.fromtimestamp(file.stat().st_mtime)
                file_data.append({
                    'Filename': file.name,
                    'Size (KB)': f"{size:.1f}",
                    'Modified': modified.strftime('%Y-%m-%d %H:%M')
                })
            
            files_df = pd.DataFrame(file_data)
            st.dataframe(files_df, use_container_width=True, height=300)
        else:
            st.info("No datasets found in analytics directory")

if __name__ == "__main__":
    main()