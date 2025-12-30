"""
Stock Analysis Page - Zynapse Capital
Individual stock deep-dive with technical charts and indicators
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Page config
st.set_page_config(
    page_title="Stock Analysis - Zynapse Capital",
    page_icon="🚀",
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
    .metric-row {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data():
    """Load latest analyzed data"""
    analytics_dir = Path('data/analytics')
    
    if not analytics_dir.exists():
        return None
    
    analyzed_files = list(analytics_dir.glob('analyzed_equity_full_*.csv'))
    
    if not analyzed_files:
        return None
    
    latest_file = max(analyzed_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file)
    
    # Fill NaN values
    numeric_cols = ['momentum_score', 'risk_score', 'delivery_pct', 'day_return_pct', 
                    'volume', 'composite_score', 'close', 'open', 'high', 'low']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    if 'signal' in df.columns:
        df['signal'] = df['signal'].fillna('HOLD')
    
    return df

def get_peer_comparison(df, symbol, date):
    """Get peer stocks for comparison"""
    stock_data = df[(df['symbol'] == symbol) & (df['date'] == date)]
    
    if len(stock_data) == 0:
        return pd.DataFrame()
    
    stock_momentum = stock_data['momentum_score'].values[0]
    stock_delivery = stock_data['delivery_pct'].values[0]
    
    # Find similar stocks (±10 momentum, ±10 delivery)
    peers = df[
        (df['date'] == date) &
        (df['symbol'] != symbol) &
        (df['momentum_score'].between(stock_momentum - 10, stock_momentum + 10)) &
        (df['delivery_pct'].between(stock_delivery - 10, stock_delivery + 10))
    ].nlargest(10, 'composite_score')
    
    return peers

def create_candlestick_chart(stock_df):
    """Create candlestick chart with volume"""
    if len(stock_df) < 2:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price Action', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=stock_df['date'],
            open=stock_df['open'],
            high=stock_df['high'],
            low=stock_df['low'],
            close=stock_df['close'],
            name='Price',
            increasing_line_color='#00cc00',
            decreasing_line_color='#ff6666'
        ),
        row=1, col=1
    )
    
    # Add moving averages if available
    if 'ma_5' in stock_df.columns:
        fig.add_trace(
            go.Scatter(
                x=stock_df['date'],
                y=stock_df['ma_5'],
                mode='lines',
                name='MA 5',
                line=dict(color='#FF6B6B', width=1, dash='dot')
            ),
            row=1, col=1
        )
    
    if 'ma_20' in stock_df.columns:
        fig.add_trace(
            go.Scatter(
                x=stock_df['date'],
                y=stock_df['ma_20'],
                mode='lines',
                name='MA 20',
                line=dict(color='#4ECDC4', width=2)
            ),
            row=1, col=1
        )
    
    # Volume bars
    colors = ['#00cc00' if row['close'] >= row['open'] else '#ff6666' 
              for _, row in stock_df.iterrows()]
    
    fig.add_trace(
        go.Bar(
            x=stock_df['date'],
            y=stock_df['volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def create_technical_indicators_chart(stock_df):
    """Create chart with technical indicators"""
    if len(stock_df) < 2:
        return None
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('RSI (14)', 'MACD', 'Bollinger Bands %'),
        row_heights=[0.33, 0.33, 0.34]
    )
    
    # RSI
    if 'rsi_14' in stock_df.columns:
        fig.add_trace(
            go.Scatter(
                x=stock_df['date'],
                y=stock_df['rsi_14'],
                mode='lines',
                name='RSI',
                line=dict(color='#9D4EDD', width=2)
            ),
            row=1, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=1, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=1, col=1)
    
    # MACD
    if 'macd' in stock_df.columns and 'macd_signal' in stock_df.columns:
        fig.add_trace(
            go.Scatter(
                x=stock_df['date'],
                y=stock_df['macd'],
                mode='lines',
                name='MACD',
                line=dict(color='#2E86AB', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=stock_df['date'],
                y=stock_df['macd_signal'],
                mode='lines',
                name='Signal',
                line=dict(color='#F77F00', width=2)
            ),
            row=2, col=1
        )
        
        # MACD histogram
        if 'macd_histogram' in stock_df.columns:
            colors = ['#00cc00' if val >= 0 else '#ff6666' for val in stock_df['macd_histogram']]
            fig.add_trace(
                go.Bar(
                    x=stock_df['date'],
                    y=stock_df['macd_histogram'],
                    name='Histogram',
                    marker_color=colors
                ),
                row=2, col=1
            )
    
    # Bollinger Position
    if 'bb_position' in stock_df.columns:
        fig.add_trace(
            go.Scatter(
                x=stock_df['date'],
                y=stock_df['bb_position'],
                mode='lines',
                name='BB Position',
                line=dict(color='#06A77D', width=2),
                fill='tonexty'
            ),
            row=3, col=1
        )
        
        fig.add_hline(y=100, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)
    
    fig.update_layout(height=700, showlegend=True)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=1, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1, range=[0, 100])
    
    return fig

def main():
    # Title
    st.title("🚀 Stock Analysis")
    st.markdown("**Deep-dive Individual Stock Analysis**")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("⚠️ No data found. Please download and analyze data from the home page.")
        if st.button("🏠 Go to Home"):
            st.switch_page("app.py")
        return
    
    # Get available dates and symbols
    unique_dates = sorted(df['date'].unique())
    latest_date = unique_dates[-1]
    symbols = sorted(df[df['date'] == latest_date]['symbol'].unique())
    
    # SIDEBAR
    with st.sidebar:
        st.header("🔍 Select Stock")
        
        # Stock search
        search_term = st.text_input(
            "Search Stock",
            placeholder="Type symbol name...",
            help="Search for stock symbol"
        )
        
        # Filter symbols based on search
        if search_term:
            filtered_symbols = [s for s in symbols if search_term.upper() in s.upper()]
        else:
            filtered_symbols = symbols
        
        selected_symbol = st.selectbox(
            "Stock Symbol",
            options=filtered_symbols,
            help="Select stock to analyze"
        )
        
        st.divider()
        
        # Date selector
        if len(unique_dates) > 1:
            selected_date = st.selectbox(
                "📅 Analysis Date",
                options=unique_dates,
                index=len(unique_dates)-1
            )
        else:
            selected_date = latest_date
            st.info(f"📅 Date: {selected_date}")
        
        st.divider()
        
        # Quick filters
        st.subheader("🔍 Quick Filters")
        
        show_peers = st.checkbox("Show Peer Comparison", value=True)
        show_technicals = st.checkbox("Show Technical Indicators", value=True)
        show_time_series = st.checkbox("Show Time Series", value=True)
    
    # Get stock data
    stock_df = df[df['symbol'] == selected_symbol].sort_values('date')
    latest_stock = stock_df[stock_df['date'] == selected_date]
    
    if len(latest_stock) == 0:
        st.error(f"❌ No data found for {selected_symbol} on {selected_date}")
        return
    
    latest_stock = latest_stock.iloc[0]
    
    # MAIN CONTENT
    
    # Header with stock name and signal
    signal_colors = {
        'STRONG_BUY': '🟢',
        'BUY': '🟡',
        'HOLD': '⚪',
        'SELL': '🔴'
    }
    signal_emoji = signal_colors.get(latest_stock['signal'], '⚪')
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header(f"{signal_emoji} {selected_symbol}")
        st.caption(f"Analysis Date: {selected_date}")
    
    with col2:
        # Signal badge
        signal_color_map = {
            'STRONG_BUY': 'green',
            'BUY': 'blue',
            'HOLD': 'orange',
            'SELL': 'red'
        }
        signal_color = signal_color_map.get(latest_stock['signal'], 'gray')
        st.markdown(f"""
        <div style='text-align: right; padding: 10px;'>
            <span style='background-color: {signal_color}; color: white; padding: 10px 20px; 
                         border-radius: 20px; font-size: 18px; font-weight: bold;'>
                {latest_stock['signal']}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Key Metrics Grid
    st.subheader("📊 Key Metrics")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Close Price", f"₹{latest_stock['close']:.2f}")
    
    with col2:
        day_return = latest_stock['day_return_pct']
        st.metric("Day Return", f"{day_return:+.2f}%", 
                 delta=f"{day_return:+.2f}%")
    
    with col3:
        st.metric("Momentum Score", f"{latest_stock['momentum_score']:.1f}",
                 delta=f"{(latest_stock['momentum_score'] - 60):+.1f}")
    
    with col4:
        st.metric("Risk Score", f"{latest_stock['risk_score']:.1f}",
                 delta=f"{(20 - latest_stock['risk_score']):+.1f}",
                 delta_color="inverse")
    
    with col5:
        st.metric("Delivery %", f"{latest_stock['delivery_pct']:.1f}%")
    
    with col6:
        st.metric("Volume", f"{latest_stock['volume']:,.0f}")
    
    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Open", f"₹{latest_stock['open']:.2f}")
        st.metric("High", f"₹{latest_stock['high']:.2f}")
    
    with col2:
        st.metric("Low", f"₹{latest_stock['low']:.2f}")
        if 'vwap' in latest_stock.index:
            st.metric("VWAP", f"₹{latest_stock['vwap']:.2f}")
    
    with col3:
        st.metric("Composite Score", f"{latest_stock['composite_score']:.1f}")
        if 'volume_ratio' in latest_stock.index:
            st.metric("Volume Ratio", f"{latest_stock['volume_ratio']:.2f}x")
    
    with col4:
        if 'validation_flags' in latest_stock.index and latest_stock['validation_flags']:
            st.warning(f"⚠️ Flags: {latest_stock['validation_flags']}")
        else:
            st.success("✅ No Risk Flags")
    
    st.divider()
    
    # TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Price Chart",
        "📊 Technical Indicators",
        "🔄 Peer Comparison",
        "📉 Time Series Analysis"
    ])
    
    # TAB 1: Price Chart
    with tab1:
        if len(stock_df) >= 2:
            st.subheader(f"💹 {selected_symbol} Price Action")
            
            fig = create_candlestick_chart(stock_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Price statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📈 Price Range**")
                st.write(f"52-Week High: ₹{stock_df['high'].max():.2f}")
                st.write(f"52-Week Low: ₹{stock_df['low'].min():.2f}")
                st.write(f"Avg Close: ₹{stock_df['close'].mean():.2f}")
            
            with col2:
                st.markdown("**📊 Volume Stats**")
                st.write(f"Avg Volume: {stock_df['volume'].mean():,.0f}")
                st.write(f"Max Volume: {stock_df['volume'].max():,.0f}")
                st.write(f"Today vs Avg: {(latest_stock['volume'] / stock_df['volume'].mean()):,.2f}x")
            
            with col3:
                st.markdown("**💰 Returns**")
                total_return = ((stock_df['close'].iloc[-1] / stock_df['close'].iloc[0]) - 1) * 100
                st.write(f"Period Return: {total_return:+.2f}%")
                st.write(f"Best Day: {stock_df['day_return_pct'].max():+.2f}%")
                st.write(f"Worst Day: {stock_df['day_return_pct'].min():+.2f}%")
        else:
            st.info("📅 Price chart requires multiple days of data. Download more historical data.")
    
    # TAB 2: Technical Indicators
    with tab2:
        if show_technicals and len(stock_df) >= 2:
            st.subheader(f"📊 {selected_symbol} Technical Indicators")
            
            fig = create_technical_indicators_chart(stock_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Indicator interpretations
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'rsi_14' in latest_stock.index:
                    rsi = latest_stock['rsi_14']
                    st.markdown("**📍 RSI Analysis**")
                    st.metric("RSI (14)", f"{rsi:.1f}")
                    
                    if rsi < 30:
                        st.success("🟢 Oversold - Potential Buy")
                    elif rsi > 70:
                        st.error("🔴 Overbought - Potential Sell")
                    else:
                        st.info("⚪ Neutral Zone")
            
            with col2:
                if 'macd' in latest_stock.index:
                    macd = latest_stock['macd']
                    signal = latest_stock.get('macd_signal', 0)
                    st.markdown("**📈 MACD Analysis**")
                    st.metric("MACD", f"{macd:.2f}")
                    
                    if macd > signal:
                        st.success("🟢 Bullish Crossover")
                    else:
                        st.error("🔴 Bearish Crossover")
            
            with col3:
                if 'bb_position' in latest_stock.index:
                    bb_pos = latest_stock['bb_position']
                    st.markdown("**📊 Bollinger Position**")
                    st.metric("BB %", f"{bb_pos:.1f}%")
                    
                    if bb_pos > 80:
                        st.warning("⚠️ Near Upper Band")
                    elif bb_pos < 20:
                        st.success("✅ Near Lower Band")
                    else:
                        st.info("⚪ Mid Range")
        else:
            st.info("📊 Enable 'Show Technical Indicators' in sidebar to view charts")
    
    # TAB 3: Peer Comparison
    with tab3:
        if show_peers:
            st.subheader(f"🔄 Stocks Similar to {selected_symbol}")
            
            peers = get_peer_comparison(df, selected_symbol, selected_date)
            
            if len(peers) > 0:
                st.write(f"Found {len(peers)} peer stocks with similar momentum and delivery characteristics")
                
                # Comparison table - FIX HERE
                # Convert latest_stock Series to DataFrame properly
                selected_stock_df = pd.DataFrame([latest_stock])
                
                # Combine with peers
                comparison_df = pd.concat([
                    selected_stock_df,
                    peers
                ], ignore_index=True)
                
                # Select comparison columns
                comparison_cols = ['symbol', 'close', 'momentum_score', 'risk_score', 
                                   'delivery_pct', 'day_return_pct', 'volume', 'signal']
                
                comparison_df = comparison_df[comparison_cols].copy()
                
                # Convert to regular pandas (in case of any special types)
                comparison_df = pd.DataFrame(comparison_df.to_dict('records'))
                
                # Ensure volume is numeric
                comparison_df['volume'] = pd.to_numeric(comparison_df['volume'], errors='coerce').fillna(0)
                
                # Highlight selected stock
                def highlight_selected(row):
                    if row['symbol'] == selected_symbol:
                        return ['background-color: #ffffcc'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    comparison_df.style.apply(highlight_selected, axis=1).format({
                        'close': '₹{:.2f}',
                        'momentum_score': '{:.1f}',
                        'risk_score': '{:.1f}',
                        'delivery_pct': '{:.1f}%',
                        'day_return_pct': '{:+.2f}%',
                        'volume': '{:,.0f}'
                    }),
                    use_container_width=True,
                    height=400
                )
                
                # Comparison chart
                fig = px.scatter(
                    comparison_df,
                    x='momentum_score',
                    y='risk_score',
                    size='volume',
                    color='delivery_pct',
                    text='symbol',
                    title="Peer Comparison: Risk vs Momentum",
                    color_continuous_scale='RdYlGn',
                    size_max=30
                )
                
                fig.update_traces(textposition='top center')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No similar peers found with current criteria")
        else:
            st.info("📊 Enable 'Show Peer Comparison' in sidebar")
    
    # TAB 4: Time Series
    with tab4:
        if show_time_series and len(stock_df) >= 2:
            st.subheader(f"📉 {selected_symbol} Historical Trends")
            
            # Score trends
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                subplot_titles=("Momentum Score", "Risk Score", "Delivery %"),
                vertical_spacing=0.08,
                row_heights=[0.33, 0.33, 0.34]
            )
            
            fig.add_trace(
                go.Scatter(
                    x=stock_df['date'],
                    y=stock_df['momentum_score'],
                    mode='lines+markers',
                    name='Momentum',
                    line=dict(color='#2E86AB', width=3),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=stock_df['date'],
                    y=stock_df['risk_score'],
                    mode='lines+markers',
                    name='Risk',
                    line=dict(color='#F77F00', width=3),
                    marker=dict(size=6)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=stock_df['date'],
                    y=stock_df['delivery_pct'],
                    mode='lines+markers',
                    name='Delivery %',
                    line=dict(color='#06A77D', width=3),
                    marker=dict(size=6),
                    fill='tonexty'
                ),
                row=3, col=1
            )
            
            fig.update_layout(height=700, showlegend=False)
            fig.update_xaxes(title_text="Date", row=3, col=1)
            fig.update_yaxes(title_text="Score", row=1, col=1)
            fig.update_yaxes(title_text="Score", row=2, col=1)
            fig.update_yaxes(title_text="%", row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Historical stats table
            st.subheader("📋 Historical Statistics")
            
            stats = stock_df[['momentum_score', 'risk_score', 'delivery_pct', 'day_return_pct']].describe()
            st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)
        else:
            st.info("📊 Enable 'Show Time Series' in sidebar or download more historical data")

if __name__ == "__main__":
    main()