"""
Market Overview Page - Zynapse Capital
Comprehensive market analysis with 2D visualizations
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
    page_title="Market Overview - Zynapse Capital",
    page_icon="📊",
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
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data():
    """Load latest analyzed data with error handling"""
    analytics_dir = Path('data/analytics')
    
    if not analytics_dir.exists():
        return None, None
    
    analyzed_files = list(analytics_dir.glob('analyzed_equity_full_*.csv'))
    
    if not analyzed_files:
        return None, None
    
    latest_file = max(analyzed_files, key=lambda x: x.stat().st_mtime)
    
    try:
        df = pd.read_csv(latest_file)
        return df, latest_file.name
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def clean_data(df):
    """Clean and prepare data"""
    # Fill NaN values in numeric columns with 0
    numeric_cols = ['momentum_score', 'risk_score', 'delivery_pct', 'day_return_pct', 'volume', 'composite_score']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Fill NaN in signal column
    if 'signal' in df.columns:
        df['signal'] = df['signal'].fillna('HOLD')
    
    return df

def main():
    # Title
    st.title("📊 Market Overview")
    st.markdown("**Comprehensive NSE Market Analysis**")
    
    # Load data
    df, filename = load_data()
    
    if df is None:
        st.error("⚠️ No data found. Please download and analyze data from the home page.")
        if st.button("🏠 Go to Home", use_container_width=True):
            st.switch_page("app.py")
        return
    
    # Clean data
    df = clean_data(df)
    
    # Get dates
    unique_dates = sorted(df['date'].unique())
    latest_date = unique_dates[-1]
    
    st.success(f"✅ Loaded: **{filename}** | {len(df):,} records | {len(unique_dates)} trading days")
    
    # SIDEBAR
    with st.sidebar:
        st.header("🔍 Filters")
        
        # Date selector
        if len(unique_dates) > 1:
            selected_date = st.selectbox(
                "📅 Select Date",
                options=unique_dates,
                index=len(unique_dates)-1,
                help="Choose trading date to analyze"
            )
        else:
            selected_date = latest_date
            st.info(f"📅 Date: {selected_date}")
        
        # Filter to selected date
        df_latest = df[df['date'] == selected_date].copy()
        
        st.metric("Stocks (Selected Date)", f"{len(df_latest):,}")
        
        st.divider()
        
        # Filters
        st.subheader("Filter Criteria")
        
        min_momentum = st.slider(
            "Min Momentum Score",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            help="Minimum momentum score threshold"
        )
        
        max_risk = st.slider(
            "Max Risk Score",
            min_value=0,
            max_value=100,
            value=100,
            step=5,
            help="Maximum risk score threshold"
        )
        
        min_delivery = st.slider(
            "Min Delivery %",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            help="Minimum delivery percentage"
        )
        
        st.divider()
        
        # Signal filter
        available_signals = ['BUY', 'STRONG_BUY', 'HOLD', 'SELL']
        selected_signals = st.multiselect(
            "Trading Signals",
            options=available_signals,
            default=available_signals,
            help="Select signals to include"
        )
        
        # Apply filters
        filtered_df = df_latest[
            (df_latest['momentum_score'] >= min_momentum) &
            (df_latest['risk_score'] <= max_risk) &
            (df_latest['delivery_pct'] >= min_delivery) &
            (df_latest['signal'].isin(selected_signals))
        ].copy()
        
        # Show filtered count
        filter_pct = (len(filtered_df) / len(df_latest) * 100) if len(df_latest) > 0 else 0
        
        st.divider()
        
        if filter_pct >= 80:
            st.success(f"✅ **{len(filtered_df):,}** stocks ({filter_pct:.1f}%)")
        elif filter_pct >= 50:
            st.info(f"ℹ️ **{len(filtered_df):,}** stocks ({filter_pct:.1f}%)")
        elif filter_pct >= 20:
            st.warning(f"⚠️ **{len(filtered_df):,}** stocks ({filter_pct:.1f}%)")
        else:
            st.error(f"❌ **{len(filtered_df):,}** stocks ({filter_pct:.1f}%)")
        
        # Quick stats
        st.divider()
        st.caption("**Filtered Data Stats:**")
        if len(filtered_df) > 0:
            st.caption(f"Avg Momentum: {filtered_df['momentum_score'].mean():.1f}")
            st.caption(f"Avg Risk: {filtered_df['risk_score'].mean():.1f}")
            st.caption(f"Avg Delivery: {filtered_df['delivery_pct'].mean():.1f}%")
    
    # MAIN CONTENT
    
    # Check if we have data to display
    if len(filtered_df) == 0:
        st.warning("⚠️ No stocks match your filter criteria. Try adjusting the filters.")
        return
    
    # Key Metrics
    st.header("📈 Market Metrics")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Stocks", f"{len(filtered_df):,}")
    
    with col2:
        avg_momentum = filtered_df['momentum_score'].mean()
        delta_momentum = avg_momentum - 60
        st.metric("Avg Momentum", f"{avg_momentum:.1f}", 
                 delta=f"{delta_momentum:+.1f}")
    
    with col3:
        avg_risk = filtered_df['risk_score'].mean()
        delta_risk = 20 - avg_risk
        st.metric("Avg Risk", f"{avg_risk:.1f}", 
                 delta=f"{delta_risk:+.1f}", delta_color="inverse")
    
    with col4:
        avg_delivery = filtered_df['delivery_pct'].mean()
        st.metric("Avg Delivery", f"{avg_delivery:.1f}%")
    
    with col5:
        gainers = len(filtered_df[filtered_df['day_return_pct'] > 0])
        losers = len(filtered_df[filtered_df['day_return_pct'] < 0])
        st.metric("Gainers", gainers, delta=f"{gainers - losers:+d}")
    
    with col6:
        quality = len(filtered_df[
            (filtered_df['delivery_pct'] > 70) & 
            (filtered_df['risk_score'] < 20)
        ])
        st.metric("Quality Stocks", quality)
    
    st.divider()
    
    # TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Distributions",
        "🎯 Scatter Analysis",
        "📡 Signals & Categories",
        "📈 Time Series"
    ])
    
    # TAB 1: Distributions
    with tab1:
        st.subheader("📊 Score Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Momentum histogram
            fig = px.histogram(
                filtered_df,
                x='momentum_score',
                nbins=40,
                title="Momentum Score Distribution",
                labels={'momentum_score': 'Momentum Score'},
                color_discrete_sequence=['#2E86AB']
            )
            fig.add_vline(x=60, line_dash="dash", line_color="red",
                         annotation_text="Baseline (60)")
            fig.add_vline(x=filtered_df['momentum_score'].mean(), 
                         line_dash="dot", line_color="green",
                         annotation_text=f"Mean ({filtered_df['momentum_score'].mean():.1f})")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk histogram
            fig = px.histogram(
                filtered_df,
                x='risk_score',
                nbins=40,
                title="Risk Score Distribution",
                labels={'risk_score': 'Risk Score'},
                color_discrete_sequence=['#F77F00']
            )
            fig.add_vline(x=30, line_dash="dash", line_color="red",
                         annotation_text="Threshold (30)")
            fig.add_vline(x=filtered_df['risk_score'].mean(),
                         line_dash="dot", line_color="green",
                         annotation_text=f"Mean ({filtered_df['risk_score'].mean():.1f})")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Delivery histogram
            fig = px.histogram(
                filtered_df,
                x='delivery_pct',
                nbins=40,
                title="Delivery Percentage Distribution",
                labels={'delivery_pct': 'Delivery %'},
                color_discrete_sequence=['#06A77D']
            )
            fig.add_vline(x=50, line_dash="dash", line_color="orange",
                         annotation_text="Mixed (50%)")
            fig.add_vline(x=70, line_dash="dash", line_color="green",
                         annotation_text="Quality (70%)")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Returns histogram
            fig = px.histogram(
                filtered_df,
                x='day_return_pct',
                nbins=40,
                title="Daily Returns Distribution",
                labels={'day_return_pct': 'Return %'},
                color_discrete_sequence=['#D62828']
            )
            fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=2)
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Scatter Analysis
    with tab2:
        st.subheader("🎯 Multi-Dimensional Analysis")
        
        # Main scatter: Risk vs Momentum
        fig = px.scatter(
            filtered_df,
            x='momentum_score',
            y='risk_score',
            size='volume',
            color='delivery_pct',
            hover_data=['symbol', 'close', 'day_return_pct', 'signal'],
            title="Risk vs Momentum (Size: Volume, Color: Delivery %)",
            labels={
                'momentum_score': 'Momentum Score →',
                'risk_score': 'Risk Score ↑',
                'delivery_pct': 'Delivery %'
            },
            color_continuous_scale='RdYlGn',
            size_max=30
        )
        
        # Add quadrant lines
        fig.add_hline(y=30, line_dash="dash", line_color="red", opacity=0.5, line_width=2)
        fig.add_vline(x=60, line_dash="dash", line_color="red", opacity=0.5, line_width=2)
        
        # Quadrant annotations
        fig.add_annotation(x=80, y=10, text="🎯 Sweet Spot<br>(High Mom, Low Risk)",
                          showarrow=False, font=dict(size=11, color="green", family="Arial Black"),
                          bgcolor="rgba(144, 238, 144, 0.3)", borderpad=4)
        fig.add_annotation(x=80, y=50, text="⚠️ High Risk<br>(High Mom, High Risk)",
                          showarrow=False, font=dict(size=11, color="orange", family="Arial Black"),
                          bgcolor="rgba(255, 165, 0, 0.2)", borderpad=4)
        fig.add_annotation(x=40, y=10, text="🐌 Low Opportunity<br>(Low Mom, Low Risk)",
                          showarrow=False, font=dict(size=11, color="gray", family="Arial Black"),
                          bgcolor="rgba(211, 211, 211, 0.3)", borderpad=4)
        fig.add_annotation(x=40, y=50, text="🚫 Avoid<br>(Low Mom, High Risk)",
                          showarrow=False, font=dict(size=11, color="red", family="Arial Black"),
                          bgcolor="rgba(255, 99, 71, 0.2)", borderpad=4)
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional scatter plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                filtered_df,
                x='delivery_pct',
                y='day_return_pct',
                color='momentum_score',
                size='volume',
                hover_data=['symbol', 'close', 'risk_score'],
                title="Delivery % vs Returns (Color: Momentum)",
                color_continuous_scale='Viridis',
                size_max=20
            )
            fig.add_hline(y=0, line_dash="solid", line_color="black")
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                filtered_df,
                x='composite_score',
                y='volume',
                color='signal',
                hover_data=['symbol', 'close', 'momentum_score', 'risk_score'],
                title="Composite Score vs Volume",
                color_discrete_map={
                    'STRONG_BUY': '#00cc00',
                    'BUY': '#66ff66',
                    'HOLD': '#ffcc00',
                    'SELL': '#ff6666'
                },
                log_y=True
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: Signals & Categories
    with tab3:
        st.subheader("📡 Trading Signals Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Signal pie chart
            signal_counts = filtered_df['signal'].value_counts()
            fig = px.pie(
                values=signal_counts.values,
                names=signal_counts.index,
                title="Trading Signal Distribution",
                color_discrete_map={
                    'STRONG_BUY': '#00cc00',
                    'BUY': '#66ff66',
                    'HOLD': '#ffcc00',
                    'SELL': '#ff6666'
                },
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Delivery categories
            def categorize_delivery(pct):
                if pct >= 70:
                    return 'Very Strong (>70%)'
                elif pct >= 50:
                    return 'Strong (50-70%)'
                elif pct >= 30:
                    return 'Mixed (30-50%)'
                else:
                    return 'Speculative (<30%)'
            
            filtered_df['delivery_category'] = filtered_df['delivery_pct'].apply(categorize_delivery)
            delivery_counts = filtered_df['delivery_category'].value_counts()
            
            fig = px.pie(
                values=delivery_counts.values,
                names=delivery_counts.index,
                title="Delivery Category Distribution",
                color_discrete_sequence=['#006400', '#228B22', '#90EE90', '#FFB6C1'],
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        # Box plots
        st.subheader("📦 Score Distribution by Signal")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(
                filtered_df,
                x='signal',
                y='momentum_score',
                color='signal',
                title="Momentum Score by Signal",
                color_discrete_map={
                    'STRONG_BUY': '#00cc00',
                    'BUY': '#66ff66',
                    'HOLD': '#ffcc00',
                    'SELL': '#ff6666'
                }
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                filtered_df,
                x='signal',
                y='risk_score',
                color='signal',
                title="Risk Score by Signal",
                color_discrete_map={
                    'STRONG_BUY': '#00cc00',
                    'BUY': '#66ff66',
                    'HOLD': '#ffcc00',
                    'SELL': '#ff6666'
                }
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top performers
        st.subheader("🏆 Top Performers")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**💎 Top Momentum**")
            top_momentum = filtered_df.nlargest(10, 'momentum_score')[
                ['symbol', 'close', 'momentum_score', 'risk_score', 'delivery_pct']
            ]
            st.dataframe(
                top_momentum.style.format({
                    'close': '₹{:.2f}',
                    'momentum_score': '{:.1f}',
                    'risk_score': '{:.1f}',
                    'delivery_pct': '{:.1f}%'
                }).background_gradient(subset=['momentum_score'], cmap='Greens'),
                use_container_width=True,
                height=400
            )
        
        with col2:
            st.markdown("**🛡️ Lowest Risk**")
            low_risk = filtered_df.nsmallest(10, 'risk_score')[
                ['symbol', 'close', 'momentum_score', 'risk_score', 'delivery_pct']
            ]
            st.dataframe(
                low_risk.style.format({
                    'close': '₹{:.2f}',
                    'momentum_score': '{:.1f}',
                    'risk_score': '{:.1f}',
                    'delivery_pct': '{:.1f}%'
                }).background_gradient(subset=['risk_score'], cmap='Greens_r'),
                use_container_width=True,
                height=400
            )
        
        with col3:
            st.markdown("**📈 Top Gainers**")
            top_gainers = filtered_df.nlargest(10, 'day_return_pct')[
                ['symbol', 'close', 'day_return_pct', 'momentum_score', 'delivery_pct']
            ]
            st.dataframe(
                top_gainers.style.format({
                    'close': '₹{:.2f}',
                    'day_return_pct': '{:+.2f}%',
                    'momentum_score': '{:.1f}',
                    'delivery_pct': '{:.1f}%'
                }).background_gradient(subset=['day_return_pct'], cmap='RdYlGn'),
                use_container_width=True,
                height=400
            )
    
    # TAB 4: Time Series
    with tab4:
        if len(unique_dates) < 2:
            st.info("📅 Time series analysis requires multiple trading days. Download more data to see trends over time.")
        else:
            st.subheader("📈 Market Trends Over Time")
            
            # Aggregate by date
            daily_stats = df.groupby('date').agg({
                'momentum_score': 'mean',
                'risk_score': 'mean',
                'delivery_pct': 'mean',
                'volume': 'sum'
            }).reset_index()
            
            # Momentum and Risk trends
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Average Momentum Over Time", "Average Risk Over Time"),
                vertical_spacing=0.12
            )
            
            fig.add_trace(
                go.Scatter(
                    x=daily_stats['date'],
                    y=daily_stats['momentum_score'],
                    mode='lines+markers',
                    name='Momentum',
                    line=dict(color='#2E86AB', width=3),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=daily_stats['date'],
                    y=daily_stats['risk_score'],
                    mode='lines+markers',
                    name='Risk',
                    line=dict(color='#F77F00', width=3),
                    marker=dict(size=8)
                ),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Momentum Score", row=1, col=1)
            fig.update_yaxes(title_text="Risk Score", row=2, col=1)
            fig.update_layout(height=600, showlegend=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Market breadth
            st.subheader("📊 Market Breadth (Gainers vs Losers)")
            
            breadth = df.groupby('date').apply(
                lambda x: pd.Series({
                    'gainers': (x['day_return_pct'] > 0).sum(),
                    'losers': (x['day_return_pct'] < 0).sum(),
                    'unchanged': (x['day_return_pct'] == 0).sum()
                })
            ).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=breadth['date'],
                y=breadth['gainers'],
                name='Gainers',
                marker_color='#00cc00'
            ))
            fig.add_trace(go.Bar(
                x=breadth['date'],
                y=breadth['losers'],
                name='Losers',
                marker_color='#ff6666'
            ))
            fig.update_layout(
                title="Daily Market Breadth",
                xaxis_title="Date",
                yaxis_title="Number of Stocks",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()