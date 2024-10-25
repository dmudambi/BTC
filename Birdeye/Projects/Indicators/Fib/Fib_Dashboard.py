
#### Importing Libraries ####

import os
import sys
import json
from datetime import datetime
import subprocess
import pandas as pd
import plotly.graph_objects as go
current_dir = os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
sys.path.append(root_dir)
import Birdeye.Basics.Master_Functions as Master_Functions
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Specify the path to the Python file you want to run
python_file_path = 'Dataset_Generator_New_Tokens.py'

# Run the Python file
subprocess.run(['python', python_file_path])





#### Getting the most recent OHLCV Data ####

# Get the most recent folder
base_path = 'Data/New_Token_Data'
current_date = datetime.now().strftime('%Y_%m_%d')
date_folder = os.path.join(base_path, current_date)
ohlcv_folder = os.path.join(date_folder, 'OHLCV_Data')
ohlcv_datetime_folder = Master_Functions.get_most_recent_folder(ohlcv_folder)

# Print the name of the folder being used for import
print(f"Importing OHLCV data from folder: {ohlcv_datetime_folder}")

# Import the OHLCV data
imported_ohlcv_data = Master_Functions.import_ohlcv_data(ohlcv_datetime_folder)






#### Plotting the Fibonacci Retracement Levels ####

def plot_price_and_fib_levels(imported_ohlcv_data, fib_levels, initial_timeframe='5m'):
    """
    Plot price action with Fibonacci levels using a dark theme and enhanced display features.
    Mobile-optimized with clean display and improved visuals.
    
    Args:
        imported_ohlcv_data (dict): Dictionary containing OHLCV data for different tokens
        fib_levels (list): List of Fibonacci retracement levels
        initial_timeframe (str): Initial timeframe to display
        
    Returns:
        list: List of tuples containing (plotly figure, token_address)
    """
    
    def calculate_fib_proximity(df, level=0.786):
        """Calculate how close current price is to a Fibonacci level"""
        if df.empty:
            return float('inf'), 0, 0
            
        ath_close = df['close'].max()
        atl = df['low'].min()
        current_price = df['close'].iloc[-1]
        fib_price = ath_close - (ath_close - atl) * level
        proximity = abs((current_price - fib_price) / fib_price * 100)
        return proximity, fib_price, current_price
    
    def create_timeframe_annotations(timeframe, market_cap, volume):
        """Create annotations for the chart"""
        formatted_volume = Master_Functions.format_number(volume)
        return [
            dict(
                text=f"<b>Timeframe:</b> {timeframe}",
                x=0.2,
                y=-0.07,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color=TITLE_COLOR),
                bgcolor=BACKGROUND_COLOR,
                bordercolor=GRID_COLOR,
                borderwidth=1,
                borderpad=8
            ),
            dict(
                text=f"<b>Current MC:</b> {market_cap}",
                x=0.5,
                y=-0.07,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color=TITLE_COLOR),
                bgcolor=BACKGROUND_COLOR,
                bordercolor=GRID_COLOR,
                borderwidth=1,
                borderpad=8
            ),
            dict(
                text=f"<b>Prior Candle Vol:</b> {formatted_volume}",
                x=0.8,
                y=-0.07,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color=TITLE_COLOR),
                bgcolor=BACKGROUND_COLOR,
                bordercolor=GRID_COLOR,
                borderwidth=1,
                borderpad=8
            )
        ]
    
    def has_outliers(df, threshold_multiplier=15.0):
        """Check if the dataset contains outliers"""
        if df.empty:
            return True
            
        median_price = df['close'].median()
        high_outliers = df['high'] > (median_price * threshold_multiplier)
        
        if not high_outliers.any():
            return False
            
        outlier_indices = df.index[high_outliers]
        for idx in outlier_indices:
            start_idx = max(0, df.index.get_loc(idx) - 3)
            end_idx = min(len(df), df.index.get_loc(idx) + 4)
            window = df.iloc[start_idx:end_idx]
            
            window_median = window['close'].median()
            if df.loc[idx, 'high'] > window_median * threshold_multiplier:
                return True
                
        return False
    
    # Define color scheme
    BACKGROUND_COLOR = '#1e1e1e'
    TEXT_COLOR = '#ffffff'
    GRID_COLOR = 'rgba(51, 51, 51, 0.3)'
    TITLE_COLOR = '#00cc00'
    FIB_COLORS = ['#0077BE', '#ff9100', '#FFD700', '#00FF7F', '#FF00FF']
    ATH_COLOR = '#FFD700'
    
    available_timeframes = ['5m', '15m', '1H', '4H']
    token_data = []
    plot_data = []  # List to store tuples of (figure, token_address)
    
    # Process market cap data
    base_path = 'Data/New_Token_Data'
    date_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and f.startswith('2024_')]
    most_recent_date = max(date_folders)
    date_folder = os.path.join(base_path, most_recent_date)
    
    # Get market cap data
    summary_folder = os.path.join(date_folder, 'Token_Summary')
    datetime_folders = [f for f in os.listdir(summary_folder) if os.path.isdir(os.path.join(summary_folder, f))]
    most_recent_datetime = max(datetime_folders)
    summary_datetime_folder = os.path.join(summary_folder, most_recent_datetime)
    
    mc_files = [f for f in os.listdir(summary_datetime_folder) if f.startswith('new_tokens_mc') and f.endswith('.csv')]
    latest_mc_file = os.path.join(summary_datetime_folder, mc_files[0])
    market_caps = pd.read_csv(latest_mc_file, index_col=0)['Market Cap']
    
    # Calculate and sort tokens
    for token_address, timeframes in imported_ohlcv_data.items():
        if initial_timeframe not in timeframes:
            continue
            
        df_initial = timeframes[initial_timeframe]
        if has_outliers(df_initial):
            continue
            
        proximity, fib_price, current_price = calculate_fib_proximity(df_initial)
        token_data.append({
            'address': token_address,
            'proximity': proximity,
            'fib_price': fib_price,
            'current_price': current_price
        })
    
    sorted_tokens = sorted(token_data, key=lambda x: x['proximity'])
    
    # Create plots for each token
    for token_data in sorted_tokens:
        token_address = token_data['address']
        dexscreener_link = f"https://dexscreener.com/solana/{token_address}"
        print(f"\nProcessing: {dexscreener_link}")
        
        timeframes = imported_ohlcv_data[token_address]
        market_cap = Master_Functions.format_number(market_caps.get(token_address, 0))
        
        fig = go.Figure(layout_xaxis_rangeslider_visible=False)
        timeframe_traces = {}
        timeframe_volumes = {}
        
        # Process each timeframe
        for tf in available_timeframes:
            if tf not in timeframes:
                continue
                
            df = timeframes[tf]
            if df.empty:
                continue
                
            traces = []
            timeframe_volumes[tf] = df['volume'].iloc[-1]
            
            # Calculate ATH close and ATL
            ath_close = df['close'].max()
            ath_idx = df['close'].idxmax()
            atl = df['low'].min()
            
            # Add candlestick chart
            traces.append(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price",
                showlegend=False,
                increasing_line_color='#00ff9d',
                decreasing_line_color='#ff005b',
                visible=(tf == initial_timeframe)
            ))
            
            # Add ATH star
            traces.append(go.Scatter(
                x=[ath_idx],
                y=[ath_close],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=24,
                    color=ATH_COLOR,
                    line=dict(
                        color=ATH_COLOR,
                        width=2
                    )
                ),
                name='ATH Close',
                showlegend=False,
                visible=(tf == initial_timeframe)
            ))
            
            # Add Fibonacci level lines
            for i, level in enumerate(fib_levels):
                fib_price = ath_close - (ath_close - atl) * level
                traces.append(go.Scatter(
                    x=[df.index[0], df.index[-1]],
                    y=[fib_price, fib_price],
                    mode='lines',
                    line=dict(color=FIB_COLORS[i], width=3),
                    showlegend=False,
                    visible=(tf == initial_timeframe)
                ))
            
            timeframe_traces[tf] = traces
        
        # Create timeframe buttons
        buttons = []
        for tf in available_timeframes:
            if tf not in timeframe_traces:
                continue
                
            visible_traces = []
            for other_tf in timeframe_traces:
                visible_traces.extend([tf == other_tf] * len(timeframe_traces[other_tf]))
            
            tf_annotations = create_timeframe_annotations(
                tf,
                market_cap,
                timeframe_volumes[tf]
            )
            
            buttons.append(dict(
                label=tf,
                method="update",
                args=[
                    {"visible": visible_traces},
                    {"annotations": tf_annotations}
                ]
            ))
        
        # Add all traces to the figure
        for traces in timeframe_traces.values():
            fig.add_traces(traces)
        
        # Create initial annotations
        initial_annotations = create_timeframe_annotations(
            initial_timeframe,
            market_cap,
            timeframe_volumes[initial_timeframe]
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            paper_bgcolor=BACKGROUND_COLOR,
            plot_bgcolor=BACKGROUND_COLOR,
            margin=dict(r=50, t=80, l=50, b=50),
            showlegend=False,
            hovermode='x unified',
            annotations=initial_annotations,
            updatemenus=[dict(
                type="buttons",
                direction="right",
                x=0.5,
                y=1.05,
                xanchor="center",
                yanchor="bottom",
                showactive=True,
                active=available_timeframes.index(initial_timeframe),
                buttons=buttons,
                bgcolor=BACKGROUND_COLOR,
                bordercolor="#555555",
                borderwidth=2,
                font=dict(size=16, color=TITLE_COLOR),
                pad=dict(r=20, t=10, b=10, l=20)
            )],
            xaxis=dict(
                showticklabels=False,
                showgrid=True,
                gridcolor=GRID_COLOR,
                showline=True,
                linecolor=GRID_COLOR,
                rangeslider=dict(visible=False)
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=GRID_COLOR,
                showticklabels=False,
                showline=False,
                zeroline=False,
                fixedrange=True
            )
        )
        
        # Append figure and token address to plot_data
        plot_data.append((fig, token_address))
    
    # Return list of tuples containing figures and token addresses
    return plot_data