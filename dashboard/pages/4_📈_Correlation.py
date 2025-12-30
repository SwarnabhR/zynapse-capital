"""
Correlation Analysis Page - Zynapse Capital
Correlation matrices, heatmaps, and stock similarity analysis
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Page config
st.set_page_config(
    page_title="Correlation Analysis - Zynapse Capital",
    page_icon="📈",
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
                    'volume', 'composite_score', 'close', 'open', 'high', 'low']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return df

def create_correlation_matrix(df, columns):
    """Create correlation matrix heatmap"""
    
    # Calculate correlation
    corr_matrix = df[columns].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation"),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Correlation Matrix",
        xaxis_title="",
        yaxis_title="",
        height=600,
        width=600
    )
    
    return fig, corr_matrix

def create_stock_similarity_matrix(df, top_n=50):
    """Create stock similarity matrix based on features"""
    
    # Select features for similarity
    feature_cols = ['momentum_score', 'risk_score', 'delivery_pct', 
                    'day_return_pct', 'composite_score']
    
    # Get top N stocks by volume
    top_stocks = df.nlargest(top_n, 'volume')['symbol'].tolist()
    df_subset = df[df['symbol'].isin(top_stocks)].copy()
    
    # Create feature matrix
    feature_matrix = df_subset[feature_cols].values
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    
    # Calculate similarity (1 - distance)
    distances = pdist(feature_matrix_scaled, metric='euclidean')
    distance_matrix = squareform(distances)
    
    # Convert to similarity (inverse of distance)
    max_dist = distance_matrix.max()
    similarity_matrix = 1 - (distance_matrix / max_dist)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=df_subset['symbol'].tolist(),
        y=df_subset['symbol'].tolist(),
        colorscale='Viridis',
        colorbar=dict(title="Similarity"),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f"Stock Similarity Matrix (Top {top_n} by Volume)",
        height=800,
        width=800,
        xaxis={'side': 'bottom'},
        yaxis={'side': 'left'}
    )
    
    return fig, similarity_matrix, df_subset['symbol'].tolist()

def create_calendar_heatmap(df):
    """Create calendar heatmap for time series data"""
    
    # Check if we have multiple dates
    unique_dates = sorted(df['date'].unique())
    
    if len(unique_dates) < 2:
        return None
    
    # Aggregate daily metrics
    daily_metrics = df.groupby('date').agg({
        'momentum_score': 'mean',
        'risk_score': 'mean',
        'delivery_pct': 'mean',
        'day_return_pct': 'mean'
    }).reset_index()
    
    # Convert date to datetime
    daily_metrics['date'] = pd.to_datetime(daily_metrics['date'])
    daily_metrics['weekday'] = daily_metrics['date'].dt.day_name()
    daily_metrics['week'] = daily_metrics['date'].dt.isocalendar().week
    
    # Create subplots for each metric
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Avg Momentum Score', 'Avg Risk Score', 
                       'Avg Delivery %', 'Avg Daily Return %')
    )
    
    metrics = ['momentum_score', 'risk_score', 'delivery_pct', 'day_return_pct']
    positions = [(1,1), (1,2), (2,1), (2,2)]
    
    for metric, (row, col) in zip(metrics, positions):
        fig.add_trace(
            go.Bar(
                x=daily_metrics['date'],
                y=daily_metrics[metric],
                name=metric,
                marker_color=daily_metrics[metric],
                marker_colorscale='RdYlGn' if 'return' in metric or 'momentum' in metric else 'RdYlGn_r',
                showlegend=False,
                hovertemplate='Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
    
    fig.update_layout(height=700, showlegend=False, title_text="Daily Market Metrics Calendar")
    
    return fig

def create_pairplot_matrix(df, columns):
    """Create pair plot matrix (scatter plot matrix)"""
    
    # Sample data for performance
    if len(df) > 500:
        df_sample = df.sample(n=500, random_state=42)
    else:
        df_sample = df
    
    # Create scatter matrix using plotly
    fig = px.scatter_matrix(
        df_sample,
        dimensions=columns,
        color='signal',
        color_discrete_map={
            'STRONG_BUY': '#00cc00',
            'BUY': '#66ff66',
            'HOLD': '#ffcc00',
            'SELL': '#ff6666'
        },
        title="Pair Plot Matrix"
    )
    
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    fig.update_layout(height=800, width=800)
    
    return fig

def find_similar_stocks(df, symbol, top_n=10):
    """Find stocks similar to given symbol"""
    
    feature_cols = ['momentum_score', 'risk_score', 'delivery_pct', 
                    'day_return_pct', 'composite_score']
    
    # Get target stock features
    target = df[df['symbol'] == symbol][feature_cols].iloc[0].values
    
    # Calculate similarity to all other stocks
    similarities = []
    for idx, row in df.iterrows():
        if row['symbol'] != symbol:
            features = row[feature_cols].values
            # Euclidean distance
            distance = np.linalg.norm(target - features)
            similarities.append({
                'symbol': row['symbol'],
                'distance': distance,
                'momentum_score': row['momentum_score'],
                'risk_score': row['risk_score'],
                'delivery_pct': row['delivery_pct'],
                'signal': row['signal']
            })
    
    # Sort by distance (lower is more similar)
    similarities_df = pd.DataFrame(similarities).nsmallest(top_n, 'distance')
    
    return similarities_df

def main():
    # Title
    st.title("📈 Correlation Analysis")
    st.markdown("**Discover Relationships and Patterns in Market Data**")
    
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
        st.header("⚙️ Analysis Settings")
        
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
        
        # Feature selection for correlation
        st.subheader("📊 Select Features")
        
        available_features = [
            'momentum_score', 'risk_score', 'delivery_pct', 
            'day_return_pct', 'volume', 'composite_score',
            'close', 'open', 'high', 'low'
        ]
        
        # Filter only available columns
        available_features = [f for f in available_features if f in df_latest.columns]
        
        selected_features = st.multiselect(
            "Features for Correlation",
            options=available_features,
            default=['momentum_score', 'risk_score', 'delivery_pct', 'day_return_pct'],
            help="Select metrics to analyze correlations"
        )
        
        st.divider()
        
        # Similarity settings
        st.subheader("🔍 Similarity Analysis")
        
        top_n_stocks = st.slider(
            "Number of Stocks",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            help="Top N stocks by volume for similarity matrix"
        )
    
    # Main content
    if len(selected_features) < 2:
        st.warning("⚠️ Please select at least 2 features for correlation analysis")
        return
    
    # TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔢 Correlation Matrix",
        "🎯 Stock Similarity",
        "📊 Pair Plots",
        "📅 Calendar Heatmap",
        "🔍 Find Similar Stocks"
    ])
    
    # TAB 1: Correlation Matrix
    with tab1:
        st.subheader("🔢 Feature Correlation Matrix")
        
        st.info("""
        **Interpretation:**
        - Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation)
        - Values near 0 indicate no correlation
        - Red: Negative correlation | Blue: Positive correlation
        """)
        
        with st.spinner("Calculating correlations..."):
            fig, corr_matrix = create_correlation_matrix(df_latest, selected_features)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 Key Insights")
                
                # Find strongest correlations
                corr_flat = corr_matrix.abs().unstack()
                corr_flat = corr_flat[corr_flat < 1]  # Remove self-correlations
                top_corr = corr_flat.nlargest(5)
                
                st.markdown("**Strongest Correlations:**")
                for (var1, var2), value in top_corr.items():
                    original_corr = corr_matrix.loc[var1, var2]
                    direction = "positive" if original_corr > 0 else "negative"
                    st.write(f"• {var1} ↔ {var2}: {original_corr:.3f} ({direction})")
        
        # Correlation table
        st.subheader("📋 Correlation Table")
        st.dataframe(
            corr_matrix.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1)
            .format("{:.3f}"),
            use_container_width=True
        )
        
        # Download correlation matrix
        csv = corr_matrix.to_csv()
        st.download_button(
            label="📥 Download Correlation Matrix",
            data=csv,
            file_name=f"correlation_matrix_{selected_date}.csv",
            mime="text/csv"
        )
    
    # TAB 2: Stock Similarity
    with tab2:
        st.subheader("🎯 Stock Similarity Matrix")
        
        st.info(f"""
        **Similarity Analysis:**
        - Analyzing top {top_n_stocks} stocks by volume
        - Similarity based on: momentum, risk, delivery, returns, composite score
        - Darker colors = More similar stocks
        - Hover to see exact similarity scores
        """)
        
        with st.spinner(f"Calculating similarity for top {top_n_stocks} stocks..."):
            try:
                fig, sim_matrix, stock_symbols = create_stock_similarity_matrix(df_latest, top_n_stocks)
                st.plotly_chart(fig, use_container_width=True)
                
                # Find most similar pairs
                st.subheader("💎 Most Similar Stock Pairs")
                
                # Get upper triangle of similarity matrix
                mask = np.triu(np.ones_like(sim_matrix), k=1).astype(bool)
                sim_flat = sim_matrix.copy()
                sim_flat[~mask] = 0
                
                # Find top 10 similar pairs
                top_pairs = []
                for i in range(len(stock_symbols)):
                    for j in range(i+1, len(stock_symbols)):
                        top_pairs.append({
                            'Stock 1': stock_symbols[i],
                            'Stock 2': stock_symbols[j],
                            'Similarity': sim_matrix[i, j]
                        })
                
                pairs_df = pd.DataFrame(top_pairs).nlargest(10, 'Similarity')
                
                st.dataframe(
                    pairs_df.style.background_gradient(subset=['Similarity'], cmap='Greens')
                    .format({'Similarity': '{:.3f}'}),
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Error calculating similarity: {e}")
                st.info("Try reducing the number of stocks or check if sklearn is installed: `pip install scikit-learn`")
    
    # TAB 3: Pair Plots
    with tab3:
        st.subheader("📊 Pair Plot Matrix (Scatter Plot Matrix)")
        
        st.info("""
        **Pair Plots:**
        - Shows relationships between all selected features
        - Each cell = scatter plot of two features
        - Color-coded by trading signal
        - Useful for identifying non-linear relationships
        """)
        
        # Limit features for pair plot
        if len(selected_features) > 6:
            st.warning("⚠️ Too many features selected. Using first 6 for pair plot.")
            pairplot_features = selected_features[:6]
        else:
            pairplot_features = selected_features
        
        with st.spinner("Generating pair plots..."):
            fig = create_pairplot_matrix(df_latest, pairplot_features)
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: Calendar Heatmap
    with tab4:
        st.subheader("📅 Calendar Heatmap (Time Series)")
        
        if len(unique_dates) < 2:
            st.warning("""
            ⚠️ **Insufficient Time Series Data**
            
            Calendar heatmap requires multiple trading days.
            
            Current data: **1 day**
            
            Download more historical data from the home page to unlock this visualization.
            """)
            
            if st.button("🏠 Go to Home"):
                st.switch_page("app.py")
        else:
            st.info(f"📅 Analyzing {len(unique_dates)} trading days")
            
            with st.spinner("Generating calendar heatmap..."):
                fig = create_calendar_heatmap(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Daily statistics table
                    st.subheader("📊 Daily Statistics")
                    
                    daily_stats = df.groupby('date').agg({
                        'momentum_score': ['mean', 'std'],
                        'risk_score': ['mean', 'std'],
                        'delivery_pct': ['mean', 'std'],
                        'day_return_pct': ['mean', 'std']
                    }).round(2)
                    
                    st.dataframe(daily_stats, use_container_width=True)
    
    # TAB 5: Find Similar Stocks
    with tab5:
        st.subheader("🔍 Find Stocks Similar to...")
        
        # Stock selector
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_symbol = st.text_input(
                "Search Stock Symbol",
                placeholder="Type symbol...",
                help="Enter stock symbol to find similar stocks"
            )
        
        with col2:
            symbols = sorted(df_latest['symbol'].unique())
            
            if search_symbol:
                filtered_symbols = [s for s in symbols if search_symbol.upper() in s.upper()]
            else:
                filtered_symbols = symbols
            
            selected_symbol = st.selectbox(
                "Or Select Symbol",
                options=filtered_symbols
            )
        
        if selected_symbol:
            st.divider()
            
            # Show selected stock info
            stock_info = df_latest[df_latest['symbol'] == selected_symbol].iloc[0]
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Momentum", f"{stock_info['momentum_score']:.1f}")
            with col2:
                st.metric("Risk", f"{stock_info['risk_score']:.1f}")
            with col3:
                st.metric("Delivery %", f"{stock_info['delivery_pct']:.1f}%")
            with col4:
                st.metric("Return", f"{stock_info['day_return_pct']:+.2f}%")
            with col5:
                st.metric("Signal", stock_info['signal'])
            
            st.divider()
            
            # Find similar stocks
            n_similar = st.slider("Number of Similar Stocks", 5, 20, 10)
            
            with st.spinner(f"Finding {n_similar} most similar stocks..."):
                similar_stocks = find_similar_stocks(df_latest, selected_symbol, n_similar)
                
                st.subheader(f"📊 Top {n_similar} Stocks Similar to {selected_symbol}")
                
                # Format display
                display_df = similar_stocks[['symbol', 'momentum_score', 'risk_score', 
                                            'delivery_pct', 'signal']].copy()
                display_df['similarity_rank'] = range(1, len(display_df) + 1)
                display_df = display_df[['similarity_rank', 'symbol', 'momentum_score', 
                                        'risk_score', 'delivery_pct', 'signal']]
                
                st.dataframe(
                    display_df.style.format({
                        'momentum_score': '{:.1f}',
                        'risk_score': '{:.1f}',
                        'delivery_pct': '{:.1f}%'
                    }),
                    use_container_width=True,
                    height=400
                )
                
                # Visualization
                fig = px.scatter(
                    similar_stocks,
                    x='momentum_score',
                    y='risk_score',
                    size='delivery_pct',
                    color='signal',
                    text='symbol',
                    title=f"Stocks Similar to {selected_symbol}",
                    color_discrete_map={
                        'STRONG_BUY': '#00cc00',
                        'BUY': '#66ff66',
                        'HOLD': '#ffcc00',
                        'SELL': '#ff6666'
                    }
                )
                
                # Add the original stock
                fig.add_trace(go.Scatter(
                    x=[stock_info['momentum_score']],
                    y=[stock_info['risk_score']],
                    mode='markers+text',
                    marker=dict(size=20, color='red', symbol='star', line=dict(color='white', width=2)),
                    text=[selected_symbol],
                    name=f'{selected_symbol} (Target)',
                    textposition='top center',
                    textfont=dict(size=14, color='red', family='Arial Black')
                ))
                
                fig.update_traces(textposition='top center')
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()