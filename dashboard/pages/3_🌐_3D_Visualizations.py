"""
3D Visualizations Page - Zynapse Capital
Interactive 3D scatter plots for multi-dimensional analysis
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Page config
st.set_page_config(
    page_title="3D Visualizations - Zynapse Capital",
    page_icon="🌐",
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
                    'volume', 'composite_score', 'close']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    if 'signal' in df.columns:
        df['signal'] = df['signal'].fillna('HOLD')
    
    return df

def create_3d_scatter_basic(df, x_col, y_col, z_col, color_col, title, size_col='volume'):
    """Create basic 3D scatter plot"""
    
    # Prepare data
    plot_df = df[[x_col, y_col, z_col, color_col, size_col, 'symbol']].copy()
    
    # Normalize size for better visualization
    if size_col in plot_df.columns:
        plot_df['size_normalized'] = np.log10(plot_df[size_col] + 1)
        plot_df['size_normalized'] = (plot_df['size_normalized'] - plot_df['size_normalized'].min()) / \
                                      (plot_df['size_normalized'].max() - plot_df['size_normalized'].min()) * 20 + 5
    else:
        plot_df['size_normalized'] = 10
    
    # Create figure
    fig = go.Figure()
    
    # Color mapping for signals
    if color_col == 'signal':
        color_map = {
            'STRONG_BUY': '#00cc00',
            'BUY': '#66ff66',
            'HOLD': '#ffcc00',
            'SELL': '#ff6666'
        }
        
        for signal, color in color_map.items():
            signal_df = plot_df[plot_df[color_col] == signal]
            if len(signal_df) > 0:
                fig.add_trace(go.Scatter3d(
                    x=signal_df[x_col],
                    y=signal_df[y_col],
                    z=signal_df[z_col],
                    mode='markers',
                    name=signal,
                    marker=dict(
                        size=signal_df['size_normalized'],
                        color=color,
                        opacity=0.7,
                        line=dict(color='white', width=0.5)
                    ),
                    text=signal_df['symbol'],
                    hovertemplate=(
                        '<b>%{text}</b><br>' +
                        f'{x_col}: %{{x:.1f}}<br>' +
                        f'{y_col}: %{{y:.1f}}<br>' +
                        f'{z_col}: %{{z:.1f}}<br>' +
                        '<extra></extra>'
                    )
                ))
    else:
        # Continuous color scale
        fig.add_trace(go.Scatter3d(
            x=plot_df[x_col],
            y=plot_df[y_col],
            z=plot_df[z_col],
            mode='markers',
            marker=dict(
                size=plot_df['size_normalized'],
                color=plot_df[color_col],
                colorscale='RdYlGn',
                opacity=0.7,
                colorbar=dict(title=color_col),
                line=dict(color='white', width=0.5)
            ),
            text=plot_df['symbol'],
            hovertemplate=(
                '<b>%{text}</b><br>' +
                f'{x_col}: %{{x:.1f}}<br>' +
                f'{y_col}: %{{y:.1f}}<br>' +
                f'{z_col}: %{{z:.1f}}<br>' +
                f'{color_col}: %{{marker.color:.1f}}<br>' +
                '<extra></extra>'
            )
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        height=700,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig

def create_time_series_3d(df):
    """Create 3D time series surface plot"""
    
    # Check if we have multiple dates
    unique_dates = sorted(df['date'].unique())
    
    if len(unique_dates) < 3:
        return None
    
    # Sample stocks (top 50 by volume)
    top_stocks = df.groupby('symbol')['volume'].mean().nlargest(50).index.tolist()
    df_subset = df[df['symbol'].isin(top_stocks)].copy()
    
    # Pivot for surface plot
    pivot_data = df_subset.pivot_table(
        values='close',
        index='date',
        columns='symbol',
        aggfunc='first'
    ).fillna(method='ffill')
    
    # Normalize prices for better visualization
    pivot_normalized = (pivot_data - pivot_data.mean()) / pivot_data.std()
    
    # Create figure
    fig = go.Figure(data=[go.Surface(
        z=pivot_normalized.values,
        x=list(range(len(pivot_data.columns))),
        y=list(range(len(pivot_data.index))),
        colorscale='Viridis',
        colorbar=dict(title="Normalized Price")
    )])
    
    fig.update_layout(
        title="Price Surface: Time vs Stocks (Top 50 by Volume)",
        scene=dict(
            xaxis_title="Stock Index",
            yaxis_title="Time Index",
            zaxis_title="Normalized Price",
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.3))
        ),
        height=700
    )
    
    return fig

def create_cluster_visualization(df):
    """Create 3D cluster visualization"""
    
    # Sample data for better performance
    if len(df) > 1000:
        df_sample = df.sample(n=1000, random_state=42)
    else:
        df_sample = df
    
    fig = go.Figure()
    
    # Define clusters based on risk and momentum
    def assign_cluster(row):
        if row['momentum_score'] > 70 and row['risk_score'] < 20:
            return 'Quality (High Mom, Low Risk)'
        elif row['momentum_score'] > 70 and row['risk_score'] > 30:
            return 'High Risk High Momentum'
        elif row['momentum_score'] < 50 and row['risk_score'] < 20:
            return 'Low Activity Low Risk'
        elif row['momentum_score'] < 50 and row['risk_score'] > 30:
            return 'Avoid (Low Mom, High Risk)'
        else:
            return 'Neutral'
    
    df_sample['cluster'] = df_sample.apply(assign_cluster, axis=1)
    
    # Color map for clusters
    cluster_colors = {
        'Quality (High Mom, Low Risk)': '#00cc00',
        'High Risk High Momentum': '#ff9900',
        'Low Activity Low Risk': '#3399ff',
        'Avoid (Low Mom, High Risk)': '#ff3333',
        'Neutral': '#999999'
    }
    
    for cluster, color in cluster_colors.items():
        cluster_df = df_sample[df_sample['cluster'] == cluster]
        if len(cluster_df) > 0:
            # Log scale for volume
            size_data = np.log10(cluster_df['volume'] + 1)
            size_normalized = ((size_data - size_data.min()) / (size_data.max() - size_data.min()) * 15 + 5)
            
            fig.add_trace(go.Scatter3d(
                x=cluster_df['momentum_score'],
                y=cluster_df['risk_score'],
                z=cluster_df['delivery_pct'],
                mode='markers',
                name=cluster,
                marker=dict(
                    size=size_normalized,
                    color=color,
                    opacity=0.7,
                    line=dict(color='white', width=0.5)
                ),
                text=cluster_df['symbol'],
                hovertemplate=(
                    '<b>%{text}</b><br>' +
                    'Momentum: %{x:.1f}<br>' +
                    'Risk: %{y:.1f}<br>' +
                    'Delivery: %{z:.1f}%<br>' +
                    f'Cluster: {cluster}<br>' +
                    '<extra></extra>'
                )
            ))
    
    fig.update_layout(
        title="Stock Clusters: Momentum vs Risk vs Delivery",
        scene=dict(
            xaxis_title="Momentum Score",
            yaxis_title="Risk Score",
            zaxis_title="Delivery %",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        height=700,
        showlegend=True
    )
    
    return fig

def main():
    # Title
    st.title("🌐 3D Visualizations")
    st.markdown("**Interactive Multi-Dimensional Analysis**")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("⚠️ No data found. Please download and analyze data from the home page.")
        if st.button("🏠 Go to Home"):
            st.switch_page("app.py")
        return
    
    # Get latest date
    unique_dates = sorted(df['date'].unique())
    latest_date = unique_dates[-1]
    
    # SIDEBAR
    with st.sidebar:
        st.header("🎛️ 3D Plot Controls")
        
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
        
        df_latest = df[df['date'] == selected_date].copy()
        
        st.metric("Stocks", f"{len(df_latest):,}")
        
        st.divider()
        
        # Axis selection
        st.subheader("📊 Customize Axes")
        
        available_numeric = ['momentum_score', 'risk_score', 'delivery_pct', 
                            'day_return_pct', 'volume', 'composite_score', 'close']
        
        x_axis = st.selectbox("X-Axis", available_numeric, index=0)
        y_axis = st.selectbox("Y-Axis", available_numeric, index=1)
        z_axis = st.selectbox("Z-Axis", available_numeric, index=2)
        
        st.divider()
        
        # Color and size
        color_by = st.selectbox(
            "Color By",
            ['signal', 'delivery_pct', 'momentum_score', 'risk_score', 'day_return_pct'],
            index=0
        )
        
        size_by = st.selectbox(
            "Size By",
            ['volume', 'composite_score', 'close'],
            index=0
        )
        
        st.divider()
        
        # Sample size
        sample_size = st.slider(
            "Sample Size",
            min_value=100,
            max_value=min(2000, len(df_latest)),
            value=min(500, len(df_latest)),
            step=100,
            help="Number of stocks to display (for performance)"
        )
        
        st.divider()
        
        # Filters
        st.subheader("🔍 Filters")
        
        min_momentum = st.slider("Min Momentum", 0, 100, 0)
        max_risk = st.slider("Max Risk", 0, 100, 100)
        min_delivery = st.slider("Min Delivery %", 0, 100, 0)
    
    # Apply filters
    filtered_df = df_latest[
        (df_latest['momentum_score'] >= min_momentum) &
        (df_latest['risk_score'] <= max_risk) &
        (df_latest['delivery_pct'] >= min_delivery)
    ].copy()
    
    # Sample for performance
    if len(filtered_df) > sample_size:
        display_df = filtered_df.sample(n=sample_size, random_state=42)
    else:
        display_df = filtered_df
    
    st.success(f"✅ Displaying {len(display_df):,} / {len(filtered_df):,} stocks")
    
    st.divider()
    
    # TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Custom 3D Plot",
        "📦 Cluster Analysis",
        "🌊 Time Series Surface",
        "💎 Preset Views"
    ])
    
    # TAB 1: Custom 3D Plot
    with tab1:
        st.subheader(f"🎯 Custom 3D Analysis: {x_axis} vs {y_axis} vs {z_axis}")
        
        with st.spinner("Generating 3D visualization..."):
            fig = create_3d_scatter_basic(
                display_df,
                x_axis,
                y_axis,
                z_axis,
                color_by,
                f"3D Analysis: {x_axis} vs {y_axis} vs {z_axis}",
                size_by
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Instructions
        with st.expander("💡 How to Interact with 3D Plot"):
            st.markdown("""
            ### 3D Plot Controls:
            - **Rotate**: Click and drag
            - **Zoom**: Scroll or pinch
            - **Pan**: Right-click and drag (or Shift + drag)
            - **Hover**: See stock details
            - **Double-click legend**: Isolate signal/category
            - **Single-click legend**: Toggle visibility
            
            ### Camera Reset:
            Click the 🏠 icon in the plot toolbar to reset view
            """)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"Avg {x_axis}", f"{display_df[x_axis].mean():.2f}")
        with col2:
            st.metric(f"Avg {y_axis}", f"{display_df[y_axis].mean():.2f}")
        with col3:
            st.metric(f"Avg {z_axis}", f"{display_df[z_axis].mean():.2f}")
    
    # TAB 2: Cluster Analysis
    with tab2:
        st.subheader("📦 Intelligent Stock Clustering")
        
        st.info("""
        **Cluster Definitions:**
        - 🟢 **Quality**: High Momentum (>70) + Low Risk (<20)
        - 🟠 **High Risk High Momentum**: Momentum >70, Risk >30
        - 🔵 **Low Activity Low Risk**: Momentum <50, Risk <20
        - 🔴 **Avoid**: Low Momentum (<50) + High Risk (>30)
        - ⚪ **Neutral**: Everything else
        """)
        
        with st.spinner("Generating cluster visualization..."):
            fig = create_cluster_visualization(filtered_df)
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster statistics
        st.subheader("📊 Cluster Statistics")
        
        def assign_cluster(row):
            if row['momentum_score'] > 70 and row['risk_score'] < 20:
                return 'Quality'
            elif row['momentum_score'] > 70 and row['risk_score'] > 30:
                return 'High Risk'
            elif row['momentum_score'] < 50 and row['risk_score'] < 20:
                return 'Low Activity'
            elif row['momentum_score'] < 50 and row['risk_score'] > 30:
                return 'Avoid'
            else:
                return 'Neutral'
        
        filtered_df['cluster'] = filtered_df.apply(assign_cluster, axis=1)
        cluster_stats = filtered_df.groupby('cluster').agg({
            'symbol': 'count',
            'momentum_score': 'mean',
            'risk_score': 'mean',
            'delivery_pct': 'mean',
            'day_return_pct': 'mean'
        }).round(2)
        cluster_stats.columns = ['Count', 'Avg Momentum', 'Avg Risk', 'Avg Delivery %', 'Avg Return %']
        
        st.dataframe(cluster_stats, use_container_width=True)
    
    # TAB 3: Time Series Surface
    with tab3:
        st.subheader("🌊 Time Series Surface Plot")
        
        if len(unique_dates) < 3:
            st.warning("""
            ⚠️ **Insufficient Time Series Data**
            
            Time series surface plots require at least 3 trading days.
            
            Current data: **{} day(s)**
            
            Download more historical data from the home page to unlock this visualization.
            """.format(len(unique_dates)))
            
            if st.button("🏠 Go to Home"):
                st.switch_page("app.py")
        else:
            st.info(f"📅 Analyzing {len(unique_dates)} trading days")
            
            with st.spinner("Generating 3D surface plot..."):
                fig = create_time_series_3d(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption("""
                    **Surface Plot Explanation:**
                    - Each point represents a stock's normalized price
                    - X-axis: Stock index (top 50 by volume)
                    - Y-axis: Time progression
                    - Z-axis: Normalized price (standardized)
                    - Color: Price intensity
                    """)
                else:
                    st.error("Could not generate surface plot. Check data quality.")
    
    # TAB 4: Preset Views
    with tab4:
        st.subheader("💎 Preset 3D Views")
        
        view_option = st.radio(
            "Select Preset View",
            [
                "🎯 Risk-Momentum-Delivery (Classic)",
                "💰 Price-Volume-Return",
                "📊 Composite-Momentum-Risk",
                "🎪 Delivery-Return-Volume"
            ]
        )
        
        if "Classic" in view_option:
            fig = create_3d_scatter_basic(
                display_df, 'momentum_score', 'risk_score', 'delivery_pct',
                'signal', 'Classic View: Risk vs Momentum vs Delivery'
            )
        elif "Price-Volume" in view_option:
            fig = create_3d_scatter_basic(
                display_df, 'close', 'volume', 'day_return_pct',
                'momentum_score', 'Price vs Volume vs Return'
            )
        elif "Composite" in view_option:
            fig = create_3d_scatter_basic(
                display_df, 'composite_score', 'momentum_score', 'risk_score',
                'delivery_pct', 'Composite vs Momentum vs Risk'
            )
        else:  # Delivery-Return-Volume
            fig = create_3d_scatter_basic(
                display_df, 'delivery_pct', 'day_return_pct', 'volume',
                'signal', 'Delivery % vs Return vs Volume'
            )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()