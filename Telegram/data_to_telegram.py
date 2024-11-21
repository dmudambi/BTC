import os
import sys
import json
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import logging
from typing import Dict, Optional, List, Any, Set, Tuple
import requests
import time
import numpy as np
import pickle

# =============================================================================
# Configuration Parameters
# =============================================================================

# Time Intervals
CHECK_INTERVAL = 60        # Time between token checks (seconds)
RATE_LIMIT_DELAY = 2.0    # Delay between API calls to prevent rate limiting (seconds)

# API Settings
MAX_RETRIES = 3           # Maximum number of API call retry attempts

# Token Monitoring
NUM_RECENT_TOKENS = 3     # Number of most recent tokens to monitor
LOOKBACK_HOURS = 24       # Hours to look back for token history

# Timeframe Settings
VALID_TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1H', '2H', '4H', '1D']
DEFAULT_TIMEFRAMES = ['15m', '1H']  # Default timeframes to monitor

# Chart Parameters
VOLUME_THRESHOLD = 4      # Minimum volume multiple to consider significant
VOLUME_MA_PERIOD = 4      # Period for volume moving average
MIN_BREAKOUT_PERCENT = 50 # Minimum percentage for breakout detection
DRAWDOWN_PERCENT = 30     # Maximum drawdown percentage to monitor
MAX_HOURS_SEPARATION = 24 # Maximum hours between price points
LOOKBACK_CANDLES = 200    # Number of candles to analyze for patterns

# Folder Settings
REQUIRED_FOLDERS = ['ATH_Price', 'PUMP_FDV_Surge', 'Solana_FDV_Surge']
DEFAULT_MONITORED_FOLDERS = ['ATH_Price']

# Add to configuration section at top
LATEST_TOKEN_FILE = 'latest_token.json'  # File to store latest successful token info
SAVED_PLOTS_FILE = 'saved_plots.pkl'  # File to track saved plots

# Plot Settings
PLOT_HEIGHT = 800
MA_PERIODS = [20, 50, 200]  # Moving average periods
PRICE_DECIMALS = 8
CHANGE_DECIMALS = 2  # Decimal places for percentage changes

# Color Settings
BACKGROUND_COLOR = '#1e1e1e'
TEXT_COLOR = '#ffffff'
GRID_COLOR = 'rgba(51, 51, 51, 0.3)'
TITLE_COLOR = '#00cc00'
INCREASING_COLOR = '#00ff9d'
DECREASING_COLOR = '#ff005b'
LEGEND_TEXT_COLOR = '#ffffff'
SEPARATOR_COLOR = '#333333'

# =============================================================================
# Directory Setup
# =============================================================================

# Add root directory to path for imports
current_dir = os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

# Import custom modules
import Birdeye.Basics.Master_Functions as Master_Functions
import Birdeye.Basics.dontshare as d

# Create plots directory if it doesn't exist
PLOTS_DIR = os.path.join(current_dir, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Folder mappings
folder_mapping = {
    'ATH_Price': 'ath_price',
    'PUMP_FDV_Surge': 'pump_fdv',
    'Solana_FDV_Surge': 'solana_fdv'
}

# Data folder mappings
data_folder_mapping = {
    'ath_price': ('ATH_Price', 'Telegram_ATH_Price_Data'),
    'pump_fdv': ('PUMP_FDV_Surge', 'Telegram_PUMP_FDV_Surge_Data'),
    'solana_fdv': ('Solana_FDV_Surge', 'Telegram_Solana_FDV_Surge_Data')
}

def get_latest_file(folder_path: str, prefix: str) -> Optional[str]:
    """Get the most recent file with given prefix from specified folder."""
    files = [f for f in os.listdir(folder_path) if f.startswith(prefix) and f.endswith('.csv')]
    if not files:
        return None
    return max(files)  # This works because the timestamp is at the end of filename

def import_telegram_data() -> Dict[str, pd.DataFrame]:
    """Import the most recent data files from each Telegram data subfolder."""
    base_path = os.path.join(current_dir, 'data')
    data_dict = {}
    
    for key, (folder_name, file_prefix) in data_folder_mapping.items():
        folder_path = os.path.join(base_path, folder_name)
        latest_file = get_latest_file(folder_path, file_prefix)
        
        if latest_file:
            file_path = os.path.join(folder_path, latest_file)
            print(f"\nReading file: {file_path}")
            df = pd.read_csv(file_path)
            print(f"File contents for {key}:")
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print("First few rows:")
            print(df[['row_index', 'name', 'token']].head())
            data_dict[key] = df
        else:
            logging.warning(f"No CSV files found in {folder_path}")
    
    return data_dict

# Usage example:
telegram_data = import_telegram_data()
print(telegram_data)

def load_processed_tokens() -> Dict[str, datetime]:
    """Load the set of processed token addresses from a pickle file."""
    processed_tokens_file = 'processed_tokens.pkl'
    if os.path.exists(processed_tokens_file):
        with open(processed_tokens_file, 'rb') as f:
            return pickle.load(f)
    return {}

def save_processed_tokens(processed_tokens: Dict[str, datetime]) -> None:
    """Save the set of processed token addresses to a pickle file."""
    with open('processed_tokens.pkl', 'wb') as f:
        pickle.dump(processed_tokens, f)

def print_token_processing_history() -> None:
    """Print the history of processed tokens with their timestamps."""
    processed_tokens = load_processed_tokens()
    if not processed_tokens:
        print("\nNo tokens have been processed yet.")
        return
    
    print("\nToken Processing History:")
    print("-" * 80)
    print(f"{'Token':<50} {'Processed At':<30}")
    print("-" * 80)
    
    # Sort by timestamp, most recent first
    sorted_tokens = sorted(processed_tokens.items(), key=lambda x: x[1], reverse=True)
    for token, timestamp in sorted_tokens:
        print(f"{token:<50} {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    print(f"Total tokens processed: {len(processed_tokens)}")

def get_token_data(
    selected_folders: List[str],
    timeframes: List[str],
    num_recent_tokens: int = 3,
    lookback_hours: int = 24
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], bool, List[str]]:
    """Get token data and track new tokens."""
    processed_tokens = load_processed_tokens()
    current_time = datetime.now()
    result_data = {}
    
    # Check if this is the first run
    is_first_run = len(processed_tokens) == 0
    
    # Import telegram data
    telegram_data = import_telegram_data()
    print(f"\nImported telegram data keys: {telegram_data.keys()}")
    
    for folder in selected_folders:
        print(f"\nProcessing folder: {folder}")
        try:
            data_key = folder_mapping.get(folder)
            if not data_key or data_key not in telegram_data:
                continue
                
            df = telegram_data[data_key]
            if df.empty:
                continue
                
            print(f"Available columns: {df.columns.tolist()}")
            print(f"Initial DataFrame shape: {df.shape}")
            
            # Debug print raw data
            print("\nFirst few rows of raw data:")
            print(df[['row_index', 'name', 'token']].head())
            
            # Ensure row_index is numeric
            df['row_index'] = pd.to_numeric(df['row_index'], errors='coerce')
            
            # Sort by row_index in descending order (most recent first)
            df = df.sort_values('row_index', ascending=False)
            
            # Take top N rows
            recent_df = df.head(num_recent_tokens)
            
            print("\nAfter sorting (top rows):")
            print(recent_df[['row_index', 'name', 'token']])
            
            tokens_to_process = []
            seen_tokens = set()
            
            # Process the top N tokens
            for _, row in recent_df.iterrows():
                token = row['token']
                if token not in seen_tokens:
                    if is_first_run or token not in processed_tokens:
                        tokens_to_process.append(token)
                        print(f"\nAdding token to process: {token} ({row['name']})")
                    seen_tokens.add(token)
            
            if not tokens_to_process:
                print("\nNo new tokens to process")
                continue
            
            result_data[folder] = {}
            
            # Get OHLCV data for all tokens at once
            time.sleep(RATE_LIMIT_DELAY)
            ohlcv_data = Master_Functions.get_ohlcv_data_multi(
                tokens_to_process,
                Master_Functions.API_Key,
                timeframes
            )
            
            if not ohlcv_data:
                print("No OHLCV data returned from API")
                continue
            
            # Process each timeframe
            for timeframe in timeframes:
                result_data[folder][timeframe] = pd.DataFrame()
                dfs_for_timeframe = []
                
                # Convert timeframe to minutes for timestamp generation
                minutes_map = {'15m': 15, '1H': 60, '4H': 240, '1D': 1440}
                interval_minutes = minutes_map.get(timeframe, 15)  # default to 15 if unknown
                
                for token in tokens_to_process:
                    try:
                        if token not in ohlcv_data:
                            print(f"No data available for token: {token}")
                            continue
                            
                        if timeframe not in ohlcv_data[token]:
                            print(f"No {timeframe} data for token: {token}")
                            continue
                        
                        token_df = ohlcv_data[token][timeframe]
                        if token_df is None or token_df.empty:
                            print(f"Empty data for {token} - {timeframe}")
                            continue
                        
                        # Debug print
                        print(f"\nData structure for {token} - {timeframe}:")
                        print(f"Columns: {token_df.columns.tolist()}")
                        print(f"First row: {token_df.iloc[0].to_dict()}")
                        
                        # Generate timestamps
                        current_time = datetime.now()
                        num_periods = len(token_df)
                        timestamps = [current_time - timedelta(minutes=interval_minutes * i) 
                                    for i in range(num_periods)]
                        timestamps.reverse()  # Oldest to newest
                        
                        # Create new DataFrame with proper structure
                        new_df = pd.DataFrame({
                            'timestamp': timestamps,
                            'open': token_df['o'],
                            'high': token_df['h'],
                            'low': token_df['l'],
                            'close': token_df['c'],
                            'volume': token_df['v'],
                            'token': token
                        })
                        
                        # Convert price columns to float
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
                        
                        # Sort by timestamp
                        new_df = new_df.sort_values('timestamp')
                        
                        # Debug print after processing
                        print(f"\nProcessed data for {token} - {timeframe}:")
                        print(f"Columns: {new_df.columns.tolist()}")
                        print(f"Data types: {new_df.dtypes}")
                        print(f"First row: {new_df.iloc[0].to_dict()}")
                        
                        dfs_for_timeframe.append(new_df)
                        print(f"Successfully processed {timeframe} data for {token}")
                        
                    except Exception as e:
                        print(f"Error processing {token} - {timeframe}: {str(e)}")
                        print("Full error details:", e.__class__.__name__)
                        continue
                
                # Combine all DataFrames for this timeframe
                if dfs_for_timeframe:
                    result_data[folder][timeframe] = pd.concat(dfs_for_timeframe, ignore_index=True)
                    
                    # Final validation of combined data
                    print(f"\nFinal combined data for {timeframe}:")
                    print(f"Shape: {result_data[folder][timeframe].shape}")
                    print(f"Columns: {result_data[folder][timeframe].columns.tolist()}")
                    print(f"Data types: {result_data[folder][timeframe].dtypes}")
            
            # Update processed tokens only after successful data retrieval
            for token in tokens_to_process:
                processed_tokens[token] = current_time
                print(f"Marked as processed: {token}")
            
            save_processed_tokens(processed_tokens)
            
        except Exception as e:
            logging.error(f"Error processing folder {folder}: {str(e)}")
            continue
    
    return result_data, bool(tokens_to_process), tokens_to_process

def validate_data(result_data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Validate and print information about the processed data."""
    for folder, timeframe_data in result_data.items():
        for timeframe, df in timeframe_data.items():
            if not df.empty:
                print(f"\nValidation for {folder} - {timeframe}:")
                print(f"Number of rows: {len(df)}")
                print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                print(f"Number of unique tokens: {df['token'].nunique()}")
            else:
                print(f"\nNo data available for {folder} - {timeframe}")

def save_latest_token(token_address: str, token_name: str, timestamp: str):
    """Save the latest successfully processed token information."""
    token_info = {
        'address': token_address,
        'name': token_name,
        'timestamp': timestamp
    }
    with open(LATEST_TOKEN_FILE, 'w') as f:
        json.dump(token_info, f, indent=4)
    print(f"\nLatest successful token plot saved:")
    print(f"Token: {token_name} ({token_address})")
    print(f"Time: {timestamp}")

def check_recent_ath(df, drawdown_percent, max_hours_separation, lookback_candles, min_breakout_percent):
    """Check for recent ATH breakout pattern."""
    if len(df) <= lookback_candles + 10:
        return False, None, None, None, None
        
    # Get data excluding recent lookback period
    historical_df = df.iloc[:-lookback_candles]
    
    # Find the highest peak before lookback period
    previous_high = historical_df['high'].max()
    previous_high_idx = historical_df['high'].idxmax()
    
    # Get recent candles
    recent_df = df.iloc[-lookback_candles:]
    recent_high = recent_df['high'].max()
    recent_high_idx = recent_df['high'].idxmax()
    
    # Calculate breakout percentage
    breakout_percent = ((recent_high - previous_high) / previous_high) * 100
    
    if breakout_percent < min_breakout_percent:
        return False, None, None, None, None
        
    # Find lowest point between highs
    between_highs_df = df.loc[previous_high_idx:recent_high_idx]
    lowest_close = between_highs_df['close'].min()
    lowest_close_idx = between_highs_df['close'].idxmin()
    
    # Calculate drawdown
    drawdown = ((previous_high - lowest_close) / previous_high) * 100
    
    # Calculate time separation
    time_diff = (recent_high_idx - previous_high_idx).total_seconds() / 3600
    
    if drawdown < drawdown_percent or time_diff > max_hours_separation:
        return False, None, None, None, None
        
    return True, (previous_high, previous_high_idx), (lowest_close, lowest_close_idx), (recent_high, recent_high_idx), breakout_percent

def calculate_volume_increase(df, ma_period):
    """Calculate volume percentage increase relative to prior MA."""
    volume_ma = df['volume'].rolling(window=ma_period).mean().shift(1)
    volume_increase = df['volume'] / volume_ma
    return volume_increase, volume_ma

def create_token_plot(df: pd.DataFrame, timeframe: str, token_name: str, lookback_candles: int = 4) -> go.Figure:
    """Create a plotly figure with smooth price and volume data."""
    
    # Add color dictionaries at the top
    VOLUME_COLORS = {
        'up': 'rgba(38, 166, 154, 0.3)',  # Green
        'down': 'rgba(239, 83, 80, 0.3)'   # Red
    }
    
    MA_COLORS = {
        '20': '#2962ff'  # Blue
    }
    
    # First create the subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": False}],
               [{"secondary_y": True}]]
    )

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color=INCREASING_COLOR,
            decreasing_line_color=DECREASING_COLOR,
            increasing_fillcolor=INCREASING_COLOR,
            decreasing_fillcolor=DECREASING_COLOR,
            line=dict(width=1),
            whiskerwidth=0.5,
            hoverlabel=dict(
                bgcolor='rgba(0,0,0,0.8)',
                font=dict(color='white')
            ),
            hoverinfo='x+y',  # Show time and price values
            text=[f"Open: {o:.8f}<br>High: {h:.8f}<br>Low: {l:.8f}<br>Close: {c:.8f}"
                  for o, h, l, c in zip(df['open'], df['high'], df['low'], df['close'])]
        ),
        row=1, col=1
    )

    # Now check for ATH pattern and add markers
    has_ath, previous_high_data, lowest_close_data, recent_high_data, breakout_percent = check_recent_ath(
        df, 
        drawdown_percent=30,
        max_hours_separation=24,
        lookback_candles=lookback_candles,
        min_breakout_percent=50
    )

    if has_ath and all(x is not None for x in [previous_high_data, lowest_close_data, recent_high_data]):
        prev_high, prev_idx = previous_high_data
        low_close, low_idx = lowest_close_data
        new_high, new_idx = recent_high_data
        
        # Add connecting lines between markers
        fig.add_trace(
            go.Scatter(
                x=[df['timestamp'].iloc[prev_idx], df['timestamp'].iloc[low_idx], df['timestamp'].iloc[new_idx]],
                y=[prev_high, low_close, new_high],
                mode='lines',
                line=dict(color='rgba(255, 165, 0, 0.3)', width=1, dash='dot'),
                showlegend=False
            ),
            row=1, col=1
        )

        # Previous high marker (red triangle)
        fig.add_trace(
            go.Scatter(
                x=[df['timestamp'].iloc[prev_idx]],
                y=[prev_high],
                mode='markers+text',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='#ff4444'
                ),
                text=f'{prev_high:.8f}',
                textposition='top center',
                name='Previous High',
                showlegend=True
            ),
            row=1, col=1
        )

        # Lowest close marker (yellow triangle)
        fig.add_trace(
            go.Scatter(
                x=[df['timestamp'].iloc[low_idx]],
                y=[low_close],
                mode='markers+text',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='#ffbb33'
                ),
                text=f'{low_close:.8f}',
                textposition='bottom center',
                name='Lowest Close',
                showlegend=True
            ),
            row=1, col=1
        )

        # New high marker (green triangle)
        fig.add_trace(
            go.Scatter(
                x=[df['timestamp'].iloc[new_idx]],
                y=[new_high],
                mode='markers+text',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='#00C851'
                ),
                text=f'{new_high:.8f}',
                textposition='top center',
                name='New High',
                showlegend=True
            ),
            row=1, col=1
        )

    # Calculate volume MA
    volume_ma = df['volume'].rolling(window=20).mean()

    # Add volume bars
    colors = [VOLUME_COLORS['up'] if df['close'].iloc[i] >= df['open'].iloc[i] 
              else VOLUME_COLORS['down'] for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            marker_line_width=0,
            opacity=0.8
        ),
        row=2, col=1
    )

    # Add volume MA line
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=volume_ma,
            name='Volume MA20',
            line=dict(color='#9B59B6', width=1),  # Purple color
            opacity=0.8
        ),
        row=2, col=1
    )

    # Calculate volume multiplier
    volume_ma_4 = df['volume'].rolling(window=4).mean().shift(1)
    volume_multiplier = df['volume'] / volume_ma_4
    volume_multiplier = volume_multiplier.replace([np.inf, -np.inf], np.nan)

    # Add volume multiplier line on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=volume_multiplier,
            name='Volume Multiplier',
            line=dict(color='#F39C12', width=1),
            opacity=0.8
        ),
        row=2, col=1,
        secondary_y=True
    )

    # Add MA20 only
    ma = df['close'].rolling(window=20).mean()
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=ma,
            name='MA20',
            line=dict(color=MA_COLORS['20'], width=1),
            opacity=0.8
        ),
        row=1, col=1
    )

    # Add volume spikes markers with dynamic coloring
    volume_ma = df['volume'].rolling(window=4).mean().shift(1)
    volume_increase = df['volume'] / volume_ma
    volume_increase = volume_increase.replace([np.inf, -np.inf], np.nan)
    volume_spikes = volume_increase[
        (volume_increase >= 4) & 
        (volume_increase.notna()) & 
        (volume_increase < 1000)
    ]
    
    if not volume_spikes.empty:
        for idx in volume_spikes.index:
            spike_value = volume_increase[idx]
            price_change = (df.loc[idx, 'close'] - df.loc[idx, 'open']) / df.loc[idx, 'open'] * 100
            
            # Enhanced color intensity calculation
            intensity = min((spike_value - 4) / 16, 1)  # Normalize from 4x to 20x
            base_intensity = 0.3
            final_intensity = base_intensity + (1 - base_intensity) * intensity
            
            color = (f'rgba(0, {int(255 * final_intensity)}, {int(157 * final_intensity)}, 1)' 
                    if price_change >= 0 else 
                    f'rgba({int(255 * final_intensity)}, 0, {int(91 * final_intensity)}, 1)')
            
            fig.add_trace(
                go.Scatter(
                    x=[df.loc[idx, 'timestamp']],
                    y=[spike_value],  # Use the actual multiplier value for y-position
                    mode='markers+text',
                    marker=dict(
                        symbol='circle',
                        size=12,
                        color=color,
                        line=dict(color=color, width=2)
                    ),
                    text=f"{int(spike_value)}x",
                    textposition="top center",
                    textfont=dict(color=color, size=10),
                    showlegend=False
                ),
                row=2, col=1,
                secondary_y=True  # Plot on volume multiplier axis
            )

    # Add a single legend entry for volume spikes
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                symbol='circle',
                size=12,
                color='#00ff9d'
            ),
            name="Volume Spikes (≥3x)",
            showlegend=True
        ),
        row=2, col=1
    )

    # Update layout with market stats in title
    duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
    drawdown = ((df['high'].max() - df['close'].iloc[-1]) / df['high'].max() * 100)
    market_cap = df['close'].iloc[-1] * df['volume'].iloc[-1]  # Simplified market cap calculation
    
    title_text = (f"{token_name} - {timeframe} | "
                 f"Duration: {int(duration)}h | "
                 f"Drawdown: {drawdown:.1f}% | "
                 f"Market Cap: {market_cap/1000:.0f}K")

    if has_ath:
        title_text += f" | Breakout: {breakout_percent:.1f}%"

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=14, color=TITLE_COLOR),
            x=0.5,
            y=0.98
        ),
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        height=800,
        width=1200,
        showlegend=True,
        legend=dict(
            bgcolor=BACKGROUND_COLOR,
            font=dict(color=TEXT_COLOR),
            bordercolor=GRID_COLOR,
            borderwidth=1
        ),
        xaxis_rangeslider_visible=False,
        # Add axis titles
        yaxis_title="Price",
        yaxis2_title="Volume",
        # Add secondary y-axis for volume multiplier
        yaxis3=dict(
            title="Volume Multiplier",
            side="right",
            overlaying="y2",
            showgrid=False,
            tickfont=dict(color='#F39C12'),
            titlefont=dict(color='#F39C12')
        )
    )

    # Update axes
    for row in [1, 2]:
        fig.update_xaxes(
            showgrid=True,
            gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_COLOR, size=12),
            tickformat='%m-%d %H:%M',  # Show both date and time
            dtick='1H',  # Show hourly ticks
            tickangle=45,  # Angle the labels for better readability
            type='date',
            row=row, col=1
        )

    # Update y-axes with titles
    fig.update_yaxes(
        row=1, col=1,
        showgrid=True,
        gridcolor=GRID_COLOR,
        tickfont=dict(color=TEXT_COLOR, size=12),
        tickformat='.8f',
        title="Price",
        title_font=dict(color=TEXT_COLOR),
        zerolinecolor=GRID_COLOR
    )

    fig.update_yaxes(
        row=2, col=1,
        showgrid=True,
        gridcolor=GRID_COLOR,
        tickfont=dict(color=TEXT_COLOR, size=12),
        tickformat=',.0f',
        title="Volume",
        title_font=dict(color=TEXT_COLOR),
        zerolinecolor=GRID_COLOR
    )

    # Update secondary y-axis for volume multiplier
    fig.update_yaxes(
        title_text="Volume Multiple",
        title_font=dict(color='#F39C12'),
        tickfont=dict(color='#F39C12'),
        showgrid=False,
        row=2, col=1,
        secondary_y=True,
        range=[0, max(max(volume_multiplier.fillna(0)) * 1.1, 20)],  # Ensure range shows up to at least 20x
        tickformat='.0f',
        dtick=5  # Show ticks every 5x
    )

    return fig

# Add retry decorator
def retry_on_exception(retries=3, delay=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:  # Last attempt
                        raise  # Re-raise the last exception
                    logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(delay * (attempt + 1))  # Exponential backoff
            return None
        return wrapper
    return decorator

@retry_on_exception(retries=MAX_RETRIES, delay=RATE_LIMIT_DELAY)
def get_ohlcv_data_safe(token: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Safely get OHLCV data with retries and rate limit handling."""
    max_retries = 3
    base_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            data = Master_Functions.get_ohlcv_data_multi(
                [token], 
                Master_Functions.API_Key, 
                [timeframe]
            )
            return data.get(token, {}).get(timeframe)
        except Exception as e:
            if "rate limit" in str(e).lower():
                delay = base_delay * (attempt + 1)
                print(f"Rate limit hit, waiting {delay} seconds...")
                time.sleep(delay)
                continue
            raise
    return None

def load_saved_plots() -> Set[str]:
    """Load the set of already saved plot combinations."""
    if os.path.exists(SAVED_PLOTS_FILE):
        with open(SAVED_PLOTS_FILE, 'rb') as f:
            return pickle.load(f)
    return set()

def save_saved_plots(saved_plots: Set[str]) -> None:
    """Save the set of plot combinations."""
    with open(SAVED_PLOTS_FILE, 'wb') as f:
        pickle.dump(saved_plots, f)

def plot_ATH_data(
    token_data: Dict[str, Dict[str, pd.DataFrame]],
    initial_timeframe: str = '15m',
    drawdown_percent: float = 30,
    max_hours_separation: int = 24,
    lookback_candles: int = 200,
    volume_threshold: float = 4,
    volume_ma_period: int = 4,
    min_breakout_percent: float = 50
) -> None:
    """Plot token data with ATH analysis."""
    
    # Load saved plot combinations
    saved_plots = load_saved_plots()
    newly_saved_plots = set()
    
    telegram_data = import_telegram_data()
    
    for folder, timeframe_data in token_data.items():
        print(f"\nProcessing {folder} - {initial_timeframe}")
        
        # Get token names from the DataFrame
        token_names = {}
        if 'ath_price' in telegram_data:
            for timeframe, df in timeframe_data.items():
                for token in df['token'].unique():
                    if token not in token_names:
                        token_row = telegram_data['ath_price'][
                            telegram_data['ath_price']['token'] == token
                        ]
                        token_names[token] = token_row['name'].iloc[0] if not token_row.empty else token
        
        for timeframe in sorted(timeframe_data.keys()):
            try:
                df = timeframe_data[timeframe]
                if df.empty:
                    logging.warning(f"Empty DataFrame for {timeframe}")
                    continue
                    
                print(f"DataFrame columns: {df.columns.tolist()}")
                print(f"DataFrame shape: {df.shape}")
                
                # Process each token
                for token in df['token'].unique():
                    try:
                        # Check if this token-timeframe combination has already been saved
                        plot_key = f"{token}_{timeframe}"
                        if plot_key in saved_plots:
                            print(f"Plot already exists for {token} ({timeframe})")
                            continue
                            
                        token_df = df[df['token'] == token].copy()
                        
                        # Generate plot filename with timestamp
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        plot_filename = f"{token}_{timeframe}_{timestamp}.jpg"
                        plot_path = os.path.join(PLOTS_DIR, plot_filename)
                        
                        # Create and save plot
                        fig = create_token_plot(token_df, timeframe, token_names.get(token, 'Unknown'), lookback_candles=4)
                        fig.write_image(plot_path, scale=2)
                        print(f"Saved plot for {token} ({timeframe}) to {plot_filename}")
                        
                        # Add to saved plots set
                        newly_saved_plots.add(plot_key)
                        
                        # Print analysis
                        print(f"\nAnalysis for {token} ({timeframe}):")
                        print(f"Token Name: {token_names.get(token, 'Unknown')}")
                        print(f"Time range: {token_df['timestamp'].min()} to {token_df['timestamp'].max()}")
                        print(f"Number of candles: {len(token_df)}")
                        print(f"Current price: {token_df['close'].iloc[-1]:.8f}")
                        
                        if timeframe == initial_timeframe:
                            save_latest_token(
                                token_address=token,
                                token_name=token_names.get(token, 'Unknown'),
                                timestamp=timestamp
                            )
                        
                        # Clear the figure from memory
                        fig = None
                        
                    except Exception as e:
                        logging.error(f"Error processing token {token}: {str(e)}")
                        continue
                        
            except Exception as e:
                logging.error(f"Error processing timeframe {timeframe}: {str(e)}")
                continue
    
    # Update saved plots set with new additions
    saved_plots.update(newly_saved_plots)
    save_saved_plots(saved_plots)

def validate_token_data(token_data: Dict[str, Dict[str, pd.DataFrame]]) -> bool:
    """Validate the structure and content of token data."""
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'token']
    
    if not token_data:
        logging.error("Empty token data")
        return False
        
    print("\nValidating token data structure:")
    for folder, timeframe_data in token_data.items():
        print(f"\nFolder: {folder}")
        if not timeframe_data:
            logging.error(f"No timeframe data for folder {folder}")
            return False
            
        for timeframe, df in timeframe_data.items():
            print(f"Timeframe: {timeframe}")
            
            # Check if DataFrame is empty
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                logging.error(f"No data for {folder} - {timeframe}")
                return False
                
            if not isinstance(df, pd.DataFrame):
                logging.error(f"Invalid data type for {folder} - {timeframe}: {type(df)}")
                return False
            
            print(f"Columns: {df.columns.tolist()}")
            print(f"Shape: {df.shape}")
            
            # Check for required columns
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                logging.error(f"Missing required columns in {folder} - {timeframe}: {missing_cols}")
                return False
            
            # Validate data types
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                logging.error(f"Invalid timestamp type in {folder} - {timeframe}")
                return False
                
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    logging.error(f"Invalid numeric type for {col} in {folder} - {timeframe}")
                    return False
    
    return True

def cleanup_old_plots(max_age_hours: int = 24) -> None:
    """Remove plot files older than specified hours."""
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    
    for filename in os.listdir(PLOTS_DIR):
        if not filename.endswith('.jpg'):
            continue
            
        file_path = os.path.join(PLOTS_DIR, filename)
        file_time = datetime.fromtimestamp(os.path.getctime(file_path))
        
        if file_time < cutoff_time:
            try:
                os.remove(file_path)
                print(f"Removed old plot: {filename}")
            except Exception as e:
                print(f"Error removing {filename}: {e}")

def main():
    # Add directory validation
    data_path = os.path.join(current_dir, 'data')
    
    # Create data directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)
    
    # Create required subfolders
    for folder in REQUIRED_FOLDERS:
        folder_path = os.path.join(data_path, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            logging.warning(f"Created missing directory: {folder_path}")
    
    # Force reset processed tokens and saved plots
    if os.path.exists('processed_tokens.pkl'):
        os.remove('processed_tokens.pkl')
        print("Reset processed tokens history")
    
    if os.path.exists(SAVED_PLOTS_FILE):
        os.remove(SAVED_PLOTS_FILE)
        print("Reset saved plots history")
    
    # Use configuration parameters
    timeframes = [tf for tf in DEFAULT_TIMEFRAMES if tf in VALID_TIMEFRAMES]
    
    if not timeframes:
        logging.error("No valid timeframes specified")
        return
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\nStarting token monitoring...")
    print(f"Checking for new tokens every {CHECK_INTERVAL/60} minutes")
    print(f"Looking for {NUM_RECENT_TOKENS} most recent tokens")
    print(f"Monitoring folders: {', '.join(DEFAULT_MONITORED_FOLDERS)}")
    
    while True:
        try:
            # Clean up old plots
            cleanup_old_plots(max_age_hours=24)
            
            token_data, new_tokens_found, new_token_addresses = get_token_data(
                selected_folders=DEFAULT_MONITORED_FOLDERS,
                timeframes=timeframes,
                num_recent_tokens=NUM_RECENT_TOKENS,
                lookback_hours=LOOKBACK_HOURS
            )

            if new_tokens_found:
                print(f"\nProcessing new tokens: {new_token_addresses}")
                if validate_token_data(token_data):
                    plot_ATH_data(
                        token_data=token_data,
                        initial_timeframe=DEFAULT_TIMEFRAMES[0],
                        drawdown_percent=DRAWDOWN_PERCENT,
                        max_hours_separation=MAX_HOURS_SEPARATION,
                        lookback_candles=LOOKBACK_CANDLES,
                        volume_threshold=VOLUME_THRESHOLD,
                        volume_ma_period=VOLUME_MA_PERIOD,
                        min_breakout_percent=MIN_BREAKOUT_PERCENT
                    )
                else:
                    logging.error("Invalid token data structure")
            else:
                print("\nNo new tokens found. Waiting before next check...")
                print(f"Next check in {CHECK_INTERVAL/60} minutes...")
            
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\nScript terminated by user.")
            break
        except Exception as e:
            logging.error(f"Error in main loop: {str(e)}")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()