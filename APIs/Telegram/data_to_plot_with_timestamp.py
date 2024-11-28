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
CHECK_INTERVAL = 600        # Time between token checks (seconds)
RATE_LIMIT_DELAY = 0.25    # Delay between API calls to prevent rate limiting (seconds)

# API Settings
MAX_RETRIES = 3           # Maximum number of API call retry attempts

# Token Monitoring
NUM_RECENT_TOKENS = 50     # Number of most recent tokens to monitor

# Timeframe Settings
VALID_TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1H', '2H', '4H', '1D']
DEFAULT_TIMEFRAMES = ['5m', '15m', '1H']  # Default timeframes to monitor

DATA_HRS_PRIOR_TIMESTAMP_NOTIFICATION = 48  # Hours of data to fetch before notification timestamp

# Chart Parameters
VOLUME_MOVING_AVERAGE_PERIOD = 4 # Orange line on Plot
VOLUME_THRESHOLD = 15      # Minimum volume multiple to consider significant
VOLUME_MA_PERIOD = 4      # Period for volume moving average

# Folder Settings
REQUIRED_FOLDERS = ['ATH_Price', 'PUMP_FDV_Surge', 'Solana_FDV_Surge']
DEFAULT_MONITORED_FOLDERS = ['Solana_FDV_Surge']

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
                minutes_map = {
                    '1m': 1,
                    '5m': 5,
                    '15m': 15,
                    '1H': 60,
                    '4H': 240,
                    '1D': 1440
                }
                interval_minutes = minutes_map.get(timeframe, 15)  # default to 15 if unknown
                
                for token in tokens_to_process:
                    try:
                        if token not in ohlcv_data:
                            print(f"No data available for token: {token}")
                            continue
                            
                        if timeframe not in ohlcv_data[token]:
                            print(f"No {timeframe} data for token: {token}")
                            continue
                        
                        # Get message timestamp for this token
                        message_time = pd.to_datetime(df[df['token'] == token]['message_timestamp'].iloc[0])
                        start_time = message_time - timedelta(hours=DATA_HRS_PRIOR_TIMESTAMP_NOTIFICATION)
                        
                        # Fetch OHLCV data from start_time to current time
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
                            'token': token,
                            'message_timestamp': df[df['token'] == token]['message_timestamp'].iloc[0]
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

def calculate_volume_increase(df, ma_period):
    """Calculate volume percentage increase relative to prior MA."""
    volume_ma = df['volume'].rolling(window=ma_period).mean().shift(1)
    volume_increase = df['volume'] / volume_ma
    return volume_increase, volume_ma

def create_folder_structure(token_name: str, token: str, row_index: str, timeframe: str, folder: str) -> str:
    """Create nested folder structure and return the path for saving plots."""
    # Get current date in the format DD_Mon_YY
    current_date = datetime.now().strftime('%d_%b_%y')
    
    # Create token folder name using name_token_rowindex format
    token_folder = f"{token_name}_{token}_{row_index}"
    
    # Create path structure: plots/folder/date/token_info/timeframe
    complete_path = os.path.join(
        PLOTS_DIR,
        folder,
        current_date,
        token_folder,
        timeframe
    )
    
    # Create all necessary directories
    os.makedirs(complete_path, exist_ok=True)
    
    return complete_path

def create_token_plot(df: pd.DataFrame, timeframe: str, token_name: str, token_info: dict) -> go.Figure:
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
            hoverinfo='x+y',
            text=[f"Open: {o:.8f}<br>High: {h:.8f}<br>Low: {l:.8f}<br>Close: {c:.8f}"
                  for o, h, l, c in zip(df['open'], df['high'], df['low'], df['close'])]
        ),
        row=1, col=1
    )

    # Add notification marker based on message_timestamp
    try:
        if 'message_timestamp' in df.columns and not df['message_timestamp'].isna().all():
            # Convert message_timestamp to datetime if it's not already
            if isinstance(df['message_timestamp'].iloc[0], str):
                message_time = pd.to_datetime(df['message_timestamp'].iloc[0])
            else:
                message_time = df['message_timestamp'].iloc[0]
            
            # Ensure timestamp column is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Find the candle closest to the message timestamp
            df['time_diff'] = abs(df['timestamp'] - message_time)
            notification_idx = df['time_diff'].idxmin()
            notification_price = df.loc[notification_idx, 'high']
            notification_time = df.loc[notification_idx, 'timestamp']
            
            print(f"Adding notification marker at {notification_time} with price {notification_price}")
            
            # Add notification marker
            fig.add_trace(
                go.Scatter(
                    x=[notification_time],
                    y=[notification_price],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='#ffff00',  # Yellow color
                        line=dict(
                            color='#ffff00',
                            width=1
                        )
                    ),
                    name='Notified',
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Clean up temporary column
            df.drop('time_diff', axis=1, inplace=True)
    except Exception as e:
        print(f"Error adding notification marker: {str(e)}")

    # After adding the notification marker and before the ATH pattern check, add this code:
    try:
        if 'message_timestamp' in df.columns and not df['message_timestamp'].isna().all():
            # Get the notification price (using the existing notification_price variable)
            multiples = [2, 3, 5, 10]
            max_price = df['high'].max()
            
            for multiple in multiples:
                multiple_price = notification_price * multiple
                
                # Only show the line if price crosses it at some point
                if df['high'].max() >= multiple_price:
                    # Find the first timestamp where price crosses this multiple
                    cross_idx = df[df['high'] >= multiple_price].index[0]
                    cross_time = df.loc[cross_idx, 'timestamp']
                    
                    # Calculate position for the text (1/3 of the way through the chart)
                    text_position = df['timestamp'].iloc[len(df)//3]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=[multiple_price] * len(df),
                            mode='lines+text',
                            line=dict(
                                color='rgba(255, 165, 0, 0.5)',  # Orange with 0.5 opacity
                                dash='dash',
                                width=1
                            ),
                            text=[f"{multiple}x"],  # Text will appear once
                            textposition="middle right",
                            textfont=dict(
                                color='rgba(255, 165, 0, 0.8)',
                                size=10
                            ),
                            hoverinfo='y',
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    
                    # Add marker at crossing point
                    fig.add_trace(
                        go.Scatter(
                            x=[cross_time],
                            y=[multiple_price],
                            mode='markers',
                            marker=dict(
                                symbol='circle',
                                size=6,
                                color='rgba(255, 165, 0, 0.8)',
                                line=dict(width=1, color='rgba(255, 165, 0, 1)')
                            ),
                            showlegend=False
                        ),
                        row=1, col=1
                    )
    except Exception as e:
        print(f"Error adding multiple lines: {str(e)}")

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

    # Add volume spikes markers
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
            
            intensity = min((spike_value - 4) / 16, 1)
            base_intensity = 0.3
            final_intensity = base_intensity + (1 - base_intensity) * intensity
            
            color = (f'rgba(0, {int(255 * final_intensity)}, {int(157 * final_intensity)}, 1)' 
                    if price_change >= 0 else 
                    f'rgba({int(255 * final_intensity)}, 0, {int(91 * final_intensity)}, 1)')
            
            fig.add_trace(
                go.Scatter(
                    x=[df.loc[idx, 'timestamp']],
                    y=[spike_value],
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
                secondary_y=True
            )

    # Add volume spikes legend entry
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
    
    # Get market cap and 5m data from token_info
    mcp = token_info.get('mcp', 'N/A')
    transactions_5m = token_info.get('5m_transactions', 'N/A')
    volume_5m = token_info.get('5m_volume', 'N/A')
    
    # Convert K/M values to integers
    def convert_to_int(value):
        if isinstance(value, str):
            try:
                if 'K' in value:
                    return int(float(value.replace('K', '')) * 1000)
                elif 'M' in value:
                    return int(float(value.replace('M', '')) * 1000000)
                return int(float(value))
            except:
                return 'N/A'
        return value

    transactions_5m = convert_to_int(transactions_5m)
    volume_5m = convert_to_int(volume_5m)
    
    title_text = (f"{token_name} | {timeframe} | "
                 f"Duration: {int(duration)}h | "
                 f"Drawdown: {drawdown:.1f}% | "
                 f"Market Cap: {mcp} | "
                 f"5m Txns: {transactions_5m:,.0f} | "
                 f"5m Vol: ${volume_5m:,.0f}")

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
        yaxis_title="Price",
        yaxis2_title="Volume",
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
            tickformat='%m-%d %H:%M',
            dtick='1H',
            tickangle=45,
            type='date',
            row=row, col=1
        )

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

    fig.update_yaxes(
        title_text="Volume Multiple",
        title_font=dict(color='#F39C12'),
        tickfont=dict(color='#F39C12'),
        showgrid=False,
        row=2, col=1,
        secondary_y=True,
        range=[0, max(max(volume_multiplier.fillna(0)) * 1.1, 20)],
        tickformat='.0f',
        dtick=5
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
    volume_threshold: float = 4,
    volume_ma_period: int = 4,
) -> None:
    """Plot token data with ATH analysis."""
    
    saved_plots = load_saved_plots()
    newly_saved_plots = set()
    telegram_data = import_telegram_data()
    
    for folder, timeframe_data in token_data.items():
        print(f"\nProcessing {folder} - {initial_timeframe}")
        
        # Get token information including market cap and volume data
        token_info = {}
        if folder_mapping[folder] in telegram_data:
            df_source = telegram_data[folder_mapping[folder]]
            for _, row in df_source.iterrows():
                token_info[row['token']] = {
                    'name': row['name'],
                    'row_index': str(row['row_index']),
                    'mcp': row['mcp'],
                    '5m_transactions': row['5m_transactions'],
                    '5m_volume': row['5m_volume'],
                    'message_timestamp': row['message_timestamp']
                }
        
        for timeframe in sorted(timeframe_data.keys()):
            try:
                df = timeframe_data[timeframe]
                if df.empty:
                    continue
                
                for token in df['token'].unique():
                    try:
                        # Get token info
                        current_token_info = token_info.get(token, {
                            'name': 'Unknown',
                            'row_index': '0',
                            'mcp': 'N/A',
                            '5m_transactions': 'N/A',
                            '5m_volume': 'N/A'
                        })
                        
                        token_name = current_token_info['name']
                        row_index = current_token_info['row_index']
                        
                        # Create plot path
                        plot_path = create_folder_structure(
                            token_name=token_name,
                            token=token,
                            row_index=row_index,
                            timeframe=timeframe,
                            folder=folder
                        )
                        
                        # Check if plot already exists
                        plot_key = f"{token}_{timeframe}"
                        if plot_key in saved_plots:
                            continue
                        
                        token_df = df[df['token'] == token].copy()
                        
                        # Generate plot filename
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        plot_filename = f"{token}_{timeframe}_{timestamp}.jpg"
                        full_plot_path = os.path.join(plot_path, plot_filename)
                        
                        # Create and save plot with updated information
                        fig = create_token_plot(
                            token_df,
                            timeframe,
                            token_name,
                            current_token_info
                        )
                        fig.write_image(full_plot_path, scale=2)
                        
                        newly_saved_plots.add(plot_key)
                        
                        # Print analysis and clear figure
                        print_token_analysis(token_df, token_name, current_token_info)
                        fig = None
                        
                    except Exception as e:
                        logging.error(f"Error processing token {token}: {str(e)}")
                        continue
                        
            except Exception as e:
                logging.error(f"Error processing timeframe {timeframe}: {str(e)}")
                continue
    
    # Update saved plots
    saved_plots.update(newly_saved_plots)
    save_saved_plots(saved_plots)

def print_token_analysis(token_df: pd.DataFrame, token_name: str, token_info: dict) -> None:
    """Print analysis information for a token."""
    print(f"\nAnalysis for {token_name}:")
    print(f"Market Cap: ${token_info.get('mcp', 'N/A')}")
    print(f"5m Transactions: {token_info.get('5m_transactions', 'N/A')}")
    print(f"5m Volume: ${token_info.get('5m_volume', 'N/A')}")
    print(f"Time range: {token_df['timestamp'].min()} to {token_df['timestamp'].max()}")
    print(f"Number of candles: {len(token_df)}")
    print(f"Current price: {token_df['close'].iloc[-1]:.8f}")

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
            )

            if new_tokens_found:
                print(f"\nProcessing new tokens: {new_token_addresses}")
                if validate_token_data(token_data):
                    plot_ATH_data(
                        token_data=token_data,
                        initial_timeframe=DEFAULT_TIMEFRAMES[0],
                        volume_threshold=VOLUME_THRESHOLD,
                        volume_ma_period=VOLUME_MA_PERIOD,
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