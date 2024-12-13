import os
import sys
import asyncio
import subprocess
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import pytz
import json
import glob
from pathlib import Path
import traceback
import shutil
import aiohttp
from typing import List, Dict

MAX_CLUSTER_PERCENTAGE = 6
MAX_RISK_SCORE = 10000

def setup_logging(log_dir):
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Generate timestamp for log filename
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    log_file = os.path.join(log_dir, f'token_finder_log_{timestamp}.log')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Temporarily set to DEBUG
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create a formatter that matches terminal output
    formatter = logging.Formatter('%(message)s')
    
    # File handler with write mode
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    
    # Stream handler for console output
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    root_logger.addHandler(stream_handler)
    
    # Configure all other loggers to use the same settings
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.handlers = []  # Clear any existing handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)
    
    # Log the file location
    logging.info(f"Logging to file: {log_file}")

    return log_file

# Add this right after the imports
def configure_all_loggers():
    # Get all existing loggers
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    
    # Configure each logger to propagate to root
    for logger in loggers:
        logger.propagate = True
        logger.setLevel(logging.INFO)

# Initialize logging right after imports
current_dir = os.getcwd()
log_dir = os.path.join(current_dir, 'logs')
log_file = setup_logging(log_dir)
configure_all_loggers()  # Configure all existing loggers after setting up the root logger

current_dir = os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(os.path.join(root_dir, 'APIs', 'Rugcheck'))
from rugcheck_monitor import get_token_risk_report, MAX_RISK_SCORE, get_token_risk_report_async, check_multiple_tokens_async

# Add a rate limit filter to the main logger
class RateLimitFilter(logging.Filter):
    def filter(self, record):
        return "Rate limited" not in record.getMessage()

# Initialize paths and load environment variables early
current_dir = os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(root_dir)

# Update env_path to point to the Telegram_PirbView directory
env_path = os.path.join(root_dir, 'APIs', 'Telegram_PirbView', '.env')
load_dotenv(env_path)

# Add API path to system path
api_path = os.path.join(root_dir, 'APIs', 'Telegram_PirbView')
sys.path.append(api_path)

# Import required modules
import Birdeye.Basics.Master_Functions_Asynco as Master_Functions
from APIs.Telegram_PirbView.tel_bubblemap import PirbViewBot

# Add this after imports at the top of the file
QUALIFIED_TOKENS_FILE = os.path.join(current_dir, 'qualified_tokens.txt')

# Add this function after imports
def load_qualified_tokens():
    """Load previously qualified tokens from file"""
    if os.path.exists(QUALIFIED_TOKENS_FILE):
        with open(QUALIFIED_TOKENS_FILE, 'r') as f:
            return set(line.strip() for line in f)
    return set()

def save_qualified_token(token_address):
    """Save a newly qualified token to file"""
    with open(QUALIFIED_TOKENS_FILE, 'a') as f:
        f.write(f"{token_address}\n")

def plot_price_and_fib_levels(imported_ohlcv_data, fib_levels, initial_timeframe='15m', 
                             target_fib_level=0.786, collect_tokens=None, display_plots=True):
    """
    Analyze price data and Fibonacci levels for multiple tokens.
    """
    if collect_tokens is None:
        collect_tokens = []
    
    # Get the next lower fib level compared to target_fib_level
    fib_levels_sorted = sorted(fib_levels)
    target_idx = fib_levels_sorted.index(target_fib_level)
    higher_fib_level = fib_levels_sorted[target_idx - 1] if target_idx > 0 else None
    
    logging.info(f"Debug - Target Fib Level: {target_fib_level}")
    logging.info(f"Debug - Higher Fib Level: {higher_fib_level}")

    # Create plots directory structure if display_plots is True
    if display_plots:
        current_date = datetime.now().strftime('%Y_%m_%d')
        current_datetime = datetime.now().strftime('%Y_%b_%d_%I%M%p')  # Format: YYYY_Mon_DD_HHMM(AM/PM)
        plots_base_dir = os.path.join(current_dir, '..', 'Plots')
        date_folder = os.path.join(plots_base_dir, current_date)
        datetime_folder = os.path.join(date_folder, current_datetime)
        
        # Create directories if they don't exist
        os.makedirs(datetime_folder, exist_ok=True)

    for token_address, token_data in imported_ohlcv_data.items():
        try:
            logging.info(f"\n{'='*50}")
            logging.info(f"Analyzing Token: {token_address}")
            logging.info(f"{'='*50}")
            
            # Extract price data
            df = token_data[initial_timeframe].copy()
            
            # Find ATH Close and its index
            ath_close = df['close'].max()
            ath_idx = df['close'].idxmax()
            
            # Get first candle open (for 5x multiple check only)
            first_open = df['open'].iloc[0]
            if first_open == 0:
                first_open = df[['open', 'high', 'low', 'close']].replace(0, np.nan).min().min()
            
            # Calculate price range and Fibonacci levels using 0 as starting point
            price_range = ath_close - 0  # Changed to use 0 instead of first_open
            fib_price_levels = {level: ath_close - (price_range * level) 
                              for level in fib_levels}
            
            # Get current price (last close)
            current_price = df['close'].iloc[-1]
            
            # Debug logging for initial values
            logging.info("\nInitial Values:")
            logging.info(f"First Open: {first_open}")
            logging.info(f"ATH Close: {ath_close}")
            logging.info(f"Current Price: {current_price}")
            logging.info(f"ATH Multiple from First Open: {ath_close/first_open:.2f}x")
            
            logging.info("\nFibonacci Price Levels:")
            for level, price in fib_price_levels.items():
                logging.info(f"Fib {level}: {price}")
            
            # Check conditions with detailed logging:
            
            # 1. Check for 5x multiple from first open to ATH close
            logging.info("\nCondition 1 - 5x Multiple Check:")
            multiple = ath_close / first_open
            logging.info(f"Required: ATH ({ath_close}) >= First Open * 5 ({first_open * 5})")
            logging.info(f"Actual Multiple: {multiple:.2f}x")
            if ath_close < (first_open * 5):
                logging.info("❌ FAILED - Less than 5x from first open to ATH")
                continue
            logging.info("✅ PASSED - 5x multiple condition")
                
            # 2. Check if current price is between higher_fib_level and target_fib_level
            logging.info("\nCondition 2 - Price in Fib Range Check:")
            # Get the fib level 2 steps above target_fib_level
            target_idx = fib_levels_sorted.index(target_fib_level)
            higher_fib_level = fib_levels_sorted[target_idx - 2] if target_idx > 1 else fib_levels_sorted[0]
            logging.info(f"Required: {fib_price_levels[higher_fib_level]} >= {current_price} >= {fib_price_levels[target_fib_level]}")
            if not (fib_price_levels[higher_fib_level] >= current_price >= fib_price_levels[target_fib_level]):
                logging.info("❌ FAILED - Price not in target range")
                continue
            logging.info("✅ PASSED - Price in target range")
                
            # 3. Check if price has not gone below target_fib_level after ATH
            logging.info("\nCondition 3 - Post-ATH Price Check:")
            post_ath_df = df.loc[ath_idx:]
            min_post_ath = post_ath_df['low'].min()
            logging.info(f"Lowest post-ATH price: {min_post_ath}")
            logging.info(f"Target Fib level price: {fib_price_levels[target_fib_level]}")
            if (post_ath_df['low'] < fib_price_levels[target_fib_level]).any():
                logging.info("❌ FAILED - Price went below target fib level after ATH")
                continue
            logging.info("✅ PASSED - Price maintained above target fib level")
                
            # 4. Check for large price drops post-ATH (>50% in a single candle)
            logging.info("\nCondition 4 - Check for Large Price Drops:")
            post_ath_df = df.loc[ath_idx:]
            max_drop_percentage = 50
            
            # Calculate percentage difference between open and close for each candle
            post_ath_df['price_drop_percentage'] = ((post_ath_df['open'] - post_ath_df['close']) / post_ath_df['open']) * 100
            max_drop = post_ath_df['price_drop_percentage'].max()
            
            logging.info(f"Maximum price drop in a single candle: {max_drop:.2f}%")
            logging.info(f"Maximum allowed drop: {max_drop_percentage}%")
            
            if max_drop > max_drop_percentage:
                logging.info(f"❌ FAILED - Found price drop of {max_drop:.2f}% exceeding {max_drop_percentage}% threshold")
                continue
            logging.info("✅ PASSED - No excessive price drops found")
            
            # If all conditions pass, add to qualifying tokens
            logging.info("\nTOKEN QUALIFIED - Passed all conditions!")
            collect_tokens.append({
                'token_address': token_address,
                'current_price': current_price,
                'ath_close': ath_close,
                'first_open': first_open,
                'market_cap': df['market_cap'].iloc[-1] if 'market_cap' in df.columns else None
            })

        except Exception as e:
            logging.error(f"Error analyzing token {token_address}: {str(e)}")
            continue

    return collect_tokens

async def analyze_qualifying_tokens(qualifying_tokens, max_cluster_percentage=5):
    """
    Analyze qualifying tokens using bubblemap data and save results to files.
    """
    results = []
    csv_file = 'qualified_tokens.csv'
    
    # Define paths relative to root directory
    telegram_dir = os.path.join(root_dir, 'APIs', 'Telegram_PirbView')
    session_file = os.path.join(telegram_dir, 'session.txt')
    
    for token in qualifying_tokens:
        try:
            logging.info(f"\nAnalyzing bubblemap for token: {token['token_address']}")
            
            # Initialize PirbViewBot with correct paths
            bot = PirbViewBot()
            bot.session_file = session_file
            
            # Initialize client
            if not await bot.initialize_client():
                logging.error("Failed to initialize Telegram client")
                continue
                
            # Get bubblemap data
            bubble_data = await bot.get_token_info(token['token_address'])
            
            if bubble_data and 'clusters' in bubble_data:
                logging.info("Received bubble data:")
                logging.info(f"Token Name: {bubble_data.get('token_name')}")
                
                # Check if any individual cluster exceeds max percentage
                max_cluster = max((cluster.get('cluster_total_percentage', 0) for cluster in bubble_data['clusters']), default=0)
                logging.info(f"Largest individual cluster: {max_cluster}%")
                logging.info(f"Maximum allowed cluster: {max_cluster_percentage}%")
                
                if max_cluster > max_cluster_percentage:
                    logging.info(f"❌ REJECTED - Individual cluster {max_cluster}% exceeds maximum {max_cluster_percentage}%")
                    continue
                
                logging.info("✅ PASSED - No clusters exceed maximum percentage")
                
                # If we get here, the token passed all criteria
                logging.info("🎯 Token passed all bubblemap criteria!")
                results.append(token)
                
            else:
                logging.error("Failed to get bubble data or invalid data structure")
            
            await bot.disconnect()
            
        except Exception as e:
            logging.error(f"Error in bubblemap analysis for {token['token_address']}: {str(e)}")
            continue
            
    return results

def get_most_recent_folder(base_path):
    """Helper function to get the most recent folder"""
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base path does not exist: {base_path}")

    # Get all folders in the directory
    folders = [
        f for f in glob.glob(os.path.join(base_path, '*'))
        if os.path.isdir(f)
    ]

    if not folders:
        raise FileNotFoundError(f"No folders found in: {base_path}")

    # Get the most recent folder
    most_recent = max(folders, key=os.path.getctime)
    
    # Check if the folder contains any data
    if not any(os.path.isdir(os.path.join(most_recent, d)) for d in os.listdir(most_recent)):
        raise FileNotFoundError(f"Most recent folder is empty: {most_recent}")
        
    return most_recent

async def run_analysis_cycle():
    try:
        # Update paths to match desired structure
        current_date = datetime.now().strftime('%Y_%m_%d')
        current_time = datetime.now().strftime('%Y_%b_%d_%I%M%p')
        
        # Define base paths
        base_path = os.path.join(root_dir, 'Bots', 'Fib', 'Data', 'New_Token_Data')
        date_folder = os.path.join(base_path, current_date)
        
        # Create folder structure
        ohlcv_folder = os.path.join(date_folder, 'OHLCV_Data', current_time)
        token_summary_folder = os.path.join(date_folder, 'Token_Summary', current_time)
        results_folder = os.path.join(date_folder, 'Results', current_time)
        logs_folder = os.path.join(date_folder, 'Logs', current_time)
        
        # Create all directories
        for folder in [ohlcv_folder, token_summary_folder, results_folder, logs_folder]:
            os.makedirs(folder, exist_ok=True)
        
        # Update log file path
        log_file = setup_logging(logs_folder)

        logging.info("Starting new analysis cycle")

        # Step 1: Run Token Generator
        logging.info("Running Token Generator...")
        python_file_path = os.path.join(root_dir, 'Bots', 'Fib', 'Execution_Algo', 'token_generator.py')

        # Add debug logging
        logging.info(f"Token Generator path: {python_file_path}")
        logging.info(f"File exists: {os.path.exists(python_file_path)}")

        # Set up environment with correct PYTHONPATH
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{root_dir}:{os.path.join(root_dir, 'Bots', 'Fib')}"
        logging.info(f"PYTHONPATH set to: {env['PYTHONPATH']}")

        try:
            # Run process from the correct directory
            original_dir = os.getcwd()
            token_generator_dir = os.path.dirname(python_file_path)
            os.chdir(token_generator_dir)
            logging.info(f"Changed directory to: {token_generator_dir}")

            process = subprocess.Popen(
                ['python', '-u', os.path.basename(python_file_path)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=token_generator_dir,
                universal_newlines=True
            )

            # Create async tasks to read both stdout and stderr
            async def read_stream(stream, is_error=False):
                while True:
                    line = stream.readline()
                    if not line:
                        break
                    # Remove the "Token Generator Error: " prefix and just log the message
                    message = line.strip()
                    if is_error:
                        message = message.replace("Token Generator Error: ", "")
                    logging.info(message)

            # Create tasks for both streams
            stdout_task = asyncio.create_task(read_stream(process.stdout))
            stderr_task = asyncio.create_task(read_stream(process.stderr, True))

            # Wait for both streams to complete
            await asyncio.gather(stdout_task, stderr_task)

            # Wait for process to complete
            process.wait()

            if process.returncode != 0:
                logging.error(f"Token Generator failed with return code: {process.returncode}")
                return

            logging.info("Token Generator completed successfully")

        finally:
            os.chdir(original_dir)
            logging.info(f"Returned to directory: {original_dir}")

        # Step 2: Import token data from token_data.csv
        logging.info("Importing token data from token_data.csv...")
        token_data_path = os.path.join(token_generator_dir, 'token_data.csv')

        if not os.path.exists(token_data_path):
            logging.error(f"token_data.csv not found at {token_data_path}")
            return

        token_data_df = pd.read_csv(token_data_path)
        
        # Ensure the 'Address' column exists
        if 'Address' not in token_data_df.columns:
            logging.error("token_data.csv does not have an 'Address' column")
            return

        # Get token addresses
        token_addresses = token_data_df['Address'].tolist()
        logging.info(f"Found {len(token_addresses)} token addresses in token_data.csv")

        # New Step: Perform Rugcheck Analysis
        logging.info("Performing Rugcheck Analysis on token addresses...")
        rugcheck_results = await check_multiple_tokens_async(token_addresses)
        
        # Filter out high-risk tokens
        low_risk_tokens = [address for address, data in rugcheck_results.items() if data and data.get('score', 0) <= MAX_RISK_SCORE]
        logging.info(f"Low risk tokens after rugcheck: {len(low_risk_tokens)}")

        # Step 3: Import latest OHLCV data - Updated logging
        logging.info("\nSetting up OHLCV data paths...")
        
        # Define the exact folder structure
        base_path = os.path.join(root_dir, 'Bots', 'Fib', 'Data', 'New_Token_Data')
        current_date = datetime.now().strftime('%Y_%m_%d')
        current_time = datetime.now().strftime('%Y_%b_%d_%I%M%p')
        
        # Create folder structure
        date_folder = os.path.join(base_path, current_date)
        ohlcv_folder = os.path.join(date_folder, 'OHLCV_Data')
        ohlcv_datetime_folder = os.path.join(ohlcv_folder, current_time)
        token_summary_folder = os.path.join(date_folder, 'Token_Summary', current_time)
        
        # Create all necessary directories
        os.makedirs(ohlcv_datetime_folder, exist_ok=True)
        os.makedirs(token_summary_folder, exist_ok=True)
        
        # Detailed logging of folder structure
        logging.info("OHLCV Data Structure:")
        logging.info(f"├── Base Path: {base_path}")
        logging.info(f"├── Date Folder: {current_date}")
        logging.info(f"│   ├── OHLCV_Data")
        logging.info(f"│   │   └── {current_time}")
        logging.info(f"│   └── Token_Summary")
        logging.info(f"│       └── {current_time}")
        
        # Save token_data.csv to the Token_Summary folder
        token_data_summary_path = os.path.join(token_summary_folder, 'token_data.csv')
        token_data_df.to_csv(token_data_summary_path, index=False)
        logging.info(f"\nSaved token data summary to:")
        logging.info(f"── {token_data_summary_path}")

        # Add clear separation before OHLCV processing
        logging.info("\n" + "="*50)
        logging.info("Starting OHLCV Data Collection")
        logging.info("="*50)
        
        # Explicitly collect OHLCV data
        logging.info(f"OHLCV data will be saved to:")
        logging.info(f"└── {ohlcv_datetime_folder}")
        
        # Pass the low_risk_tokens to process_all_tokens_ohlcv
        await process_all_tokens_ohlcv(low_risk_tokens, ohlcv_datetime_folder)

        # After OHLCV data is saved successfully...
        
        # Step 3: Import OHLCV data for qualified tokens
        logging.info("\n" + "="*50)
        logging.info("Importing OHLCV data for qualified tokens")
        logging.info("="*50)
        
        try:
            # Get the path to the most recent OHLCV data folder
            ohlcv_datetime_folder = os.path.join(ohlcv_folder, current_time)
            
            # Import OHLCV data for the qualified tokens
            imported_ohlcv_data = {}
            for token_address in low_risk_tokens:
                token_folder = os.path.join(ohlcv_datetime_folder, token_address)
                if os.path.exists(token_folder):
                    try:
                        # Import the CSV data
                        csv_path = os.path.join(token_folder, '15m.csv')
                        if os.path.exists(csv_path):
                            df = pd.read_csv(csv_path)
                            
                            # Rename columns to match expected format
                            column_mapping = {
                                'c': 'close',
                                'h': 'high',
                                'l': 'low',
                                'o': 'open',
                                'v': 'volume'
                            }
                            df = df.rename(columns=column_mapping)
                            
                            # Store the processed dataframe
                            imported_ohlcv_data[token_address] = {'15m': df}
                            
                            logging.info(f"✅ Successfully imported OHLCV data for {token_address}")
                            logging.info(f"   Available timeframes: ['15m']")
                        else:
                            logging.warning(f"⚠️ No 15m.csv found for {token_address}")
                    except Exception as e:
                        logging.error(f"Error importing data for {token_address}: {str(e)}")
                else:
                    logging.warning(f"⚠️ No folder found for {token_address}")
            
            logging.info(f"\nSuccessfully imported OHLCV data for {len(imported_ohlcv_data)} tokens")

            # Step 4: Conduct Fibonacci analysis
            logging.info("\n" + "="*50)
            logging.info("Starting Fibonacci Analysis")
            logging.info("="*50)
            
            # We know we have 15m timeframe
            analysis_timeframe = '15m'
            logging.info(f"Selected timeframe for analysis: {analysis_timeframe}")
            
            qualifying_tokens = []
            fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786, 0.886]
            
            qualifying_tokens = plot_price_and_fib_levels(
                imported_ohlcv_data=imported_ohlcv_data,
                fib_levels=fib_levels,
                initial_timeframe=analysis_timeframe,
                target_fib_level=0.786,
                collect_tokens=[],
                display_plots=True
            )
            
            logging.info(f"\nFound {len(qualifying_tokens)} tokens passing Fibonacci criteria")

            # Before bubblemap analysis, check for existing tokens
            qualified_tokens_path = os.path.join(results_folder, 'qualified_tokens.csv')
            existing_tokens = set()
            
            if os.path.exists(qualified_tokens_path):
                try:
                    existing_df = pd.read_csv(qualified_tokens_path)
                    existing_tokens = set(existing_df['token_address'].unique())
                    logging.info(f"Found {len(existing_tokens)} existing qualified tokens")
                except Exception as e:
                    logging.error(f"Error reading existing qualified tokens: {str(e)}")
                    existing_tokens = set()

            # Filter out already qualified tokens
            new_qualifying_tokens = [
                token for token in qualifying_tokens 
                if token['token_address'] not in existing_tokens
            ]
            
            logging.info(f"Found {len(new_qualifying_tokens)} new tokens for bubblemap analysis")

            # Only perform bubblemap analysis on new tokens
            if new_qualifying_tokens:
                final_qualifying_tokens = await analyze_qualifying_tokens(new_qualifying_tokens, MAX_CLUSTER_PERCENTAGE)
                
                # Save final qualifying tokens
                if final_qualifying_tokens:
                    results_data = []
                    for token in final_qualifying_tokens:
                        token_data = {
                            'token_address': token['token_address'],
                            'qualification_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'current_price': token.get('current_price', 'N/A'),
                            'market_cap': token.get('market_cap', 'N/A')
                        }
                        results_data.append(token_data)
                    
                    # Create DataFrame for new tokens
                    new_results_df = pd.DataFrame(results_data)
                    
                    # Combine with existing data if present
                    if os.path.exists(qualified_tokens_path):
                        existing_df = pd.read_csv(qualified_tokens_path)
                        combined_df = pd.concat([existing_df, new_results_df], ignore_index=True)
                        # Drop duplicates based on token_address, keeping the most recent
                        combined_df = combined_df.sort_values('qualification_date').drop_duplicates(
                            'token_address', keep='last'
                        )
                    else:
                        combined_df = new_results_df
                    
                    # Save to CSV
                    combined_df.to_csv(qualified_tokens_path, index=False)
                    logging.info(f"Saved {len(final_qualifying_tokens)} new qualifying tokens to {qualified_tokens_path}")
            else:
                logging.info("No new tokens to analyze")

        except Exception as e:
            logging.error(f"Error during final analysis steps: {str(e)}")
            traceback.print_exc()

        logging.info("\nAnalysis cycle completed")

    except Exception as e:
        logging.error(f"Error in analysis cycle: {str(e)}")
        traceback.print_exc()

async def get_ohlcv_async(session, address, timeframe, api_key):
    try:
        logging.info(f"Requesting OHLCV data for token {address} on {timeframe} timeframe...")
        result = await Master_Functions.get_ohlcv_data_async(session, address, timeframe, api_key)
        if result is not None and not result.empty:
            logging.info(f"Successfully retrieved OHLCV data for {address} ({timeframe}) - {len(result)} rows")
            return result
        else:
            logging.warning(f"No OHLCV data returned for {address} ({timeframe})")
            return None
    except Exception as e:
        logging.error(f"Error getting OHLCV data for {address} ({timeframe}): {str(e)}")
        return None

async def process_token_ohlcv(address, token_folder):
    try:
        logging.info(f"\n{'='*50}")
        logging.info(f"Processing token: {address}")
        logging.info(f"{'='*50}")
        
        async with aiohttp.ClientSession() as session:
            for timeframe in Master_Functions.timeframes:
                logging.info(f"\nFetching {timeframe} data...")
                
                # Add delay for rate limiting
                await asyncio.sleep(0.08)
                
                try:
                    result = await get_ohlcv_async(session, address, timeframe, Master_Functions.API_Key)
                    
                    if result is not None and not result.empty:
                        filename = f"{timeframe}.csv"
                        file_path = os.path.join(token_folder, filename)
                        result.to_csv(file_path)
                        
                        # Log successful save with file details
                        file_size = os.path.getsize(file_path)
                        logging.info(f"✅ {timeframe} data saved:")
                        logging.info(f"   └── Path: {file_path}")
                        logging.info(f"   └── Rows: {len(result)}")
                        logging.info(f"   └── Size: {file_size/1024:.2f} KB")
                    else:
                        logging.warning(f"⚠️ No {timeframe} data available for {address}")
                        
                except Exception as e:
                    logging.error(f"❌ Error processing {timeframe} data for {address}: {str(e)}")
                    continue
            
            # Verify final token folder contents
            if os.path.exists(token_folder):
                files = os.listdir(token_folder)
                logging.info(f"\nToken {address} processing complete:")
                logging.info(f"└── Total timeframes saved: {len(files)}")
                logging.info(f"└── Folder: {token_folder}")
                return len(files) > 0
            else:
                logging.error(f"❌ Token folder not created: {token_folder}")
                return False
                
    except Exception as e:
        logging.error(f"❌ Critical error processing {address}: {str(e)}")
        logging.error(traceback.format_exc())
        return False

async def process_all_tokens_ohlcv(token_addresses, ohlcv_datetime_folder):
    logging.info(f"\nProcessing OHLCV data for {len(token_addresses)} tokens")
    
    successful_tokens = 0
    failed_tokens = 0
    
    for i, address in enumerate(token_addresses, 1):
        logging.info(f"\nToken {i}/{len(token_addresses)}: {address}")
        
        # Create token folder
        token_folder = os.path.join(ohlcv_datetime_folder, address)
        os.makedirs(token_folder, exist_ok=True)
        
        # Process token
        if await process_token_ohlcv(address, token_folder):
            successful_tokens += 1
        else:
            failed_tokens += 1
        
        # Add delay between tokens
        await asyncio.sleep(1)
    
    # Final summary
    logging.info("\n" + "="*50)
    logging.info("OHLCV Processing Summary")
    logging.info("="*50)
    logging.info(f"Total tokens processed: {len(token_addresses)}")
    logging.info(f"✅ Successful: {successful_tokens}")
    logging.info(f"❌ Failed: {failed_tokens}")
    
    # Verify final structure
    if os.path.exists(ohlcv_datetime_folder):
        total_files = sum(len(files) for _, _, files in os.walk(ohlcv_datetime_folder))
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk(ohlcv_datetime_folder)
            for filename in filenames
        )
        logging.info(f"\nFinal OHLCV Data Structure:")
        logging.info(f"└── Folder: {ohlcv_datetime_folder}")
        logging.info(f"└── Total files: {total_files}")
        logging.info(f"└── Total size: {total_size/1024/1024:.2f} MB")

async def check_multiple_tokens_async(token_addresses: List[str], batch_size: int = 2, max_retries: int = 5) -> Dict[str, Dict]:
    """Check multiple tokens concurrently with robust retry mechanism"""
    results = {}
    timeout = aiohttp.ClientTimeout(total=60)  # Increased timeout
    
    async def fetch_with_retry(address: str, session: aiohttp.ClientSession, attempt: int = 1) -> Dict:
        try:
            if attempt > 1:
                await asyncio.sleep(5 * (2 ** (attempt - 1)))
            result = await get_token_risk_report_async(address, session=session)
            if result and 'score' in result:
                logging.info(f"Token: {address}, Score: {result['score']}")
                return result
            raise Exception("Invalid response format")
        except Exception as e:
            if attempt < max_retries:
                logging.info(f"Retry {attempt}/{max_retries} for {address}")
                return await fetch_with_retry(address, session, attempt + 1)
            await asyncio.sleep(90)  # 90 second cooldown
            try:
                result = await get_token_risk_report_async(address, session=session)
                if result and 'score' in result:
                    logging.info(f"Token: {address}, Score: {result['score']}")
                    return result
            except Exception as final_e:
                logging.warning(f"Final attempt failed for {address}: {str(final_e)}")
            return None
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for i in range(0, len(token_addresses), batch_size):
                batch = token_addresses[i:i + batch_size]
                tasks = []
                
                for address in batch:
                    tasks.append(fetch_with_retry(address, session))
                    await asyncio.sleep(1)  # 1s between requests within batch
                
                batch_results = await asyncio.gather(*tasks)
                
                for address, result in zip(batch, batch_results):
                    if result and 'score' in result:
                        results[address] = result
                        logging.info(f"Token: {address}, Score: {result['score']}")
                    else:
                        await asyncio.sleep(30)  # 30s cooldown
                        try:
                            final_result = await get_token_risk_report_async(address, session=session)
                            if final_result and 'score' in final_result:
                                results[address] = final_result
                                logging.info(f"Token: {address}, Score: {final_result['score']} (final attempt)")
                            else:
                                results[address] = {'score': 99999}
                                logging.info(f"Token: {address}, Score: 99999 (failed all attempts)")
                        except Exception:
                            results[address] = {'score': 99999}
                            logging.info(f"Token: {address}, Score: 99999 (failed all attempts)")
                
                await asyncio.sleep(3)  # 3s between batches
    
    except Exception as e:
        logging.error(f"Critical error in check_multiple_tokens_async: {e}")
    
    # Verify coverage and add any missing tokens
    missing_tokens = set(token_addresses) - set(results.keys())
    if missing_tokens:
        logging.warning(f"Missing rugcheck scores for {len(missing_tokens)} tokens")
        for address in missing_tokens:
            results[address] = {'score': 99999}
            logging.info(f"Token: {address}, Score: 99999 (missing)")
    
    # Log summary
    total_tokens = len(token_addresses)
    successful_checks = sum(1 for r in results.values() if r.get('score', 99999) != 99999)
    logging.info(f"\nRugcheck Summary:")
    logging.info(f"Total tokens checked: {total_tokens}")
    logging.info(f"Successful checks: {successful_checks} ({(successful_checks/total_tokens)*100:.1f}%)")
    logging.info(f"Failed checks: {total_tokens - successful_checks} ({((total_tokens-successful_checks)/total_tokens)*100:.1f}%)")
    
    return results

async def main():
    try:
        # Verify API credentials
        if not all([os.getenv("TELEGRAM_API_ID"), os.getenv("TELEGRAM_API_HASH")]):
            logging.error("Telegram API credentials not found. Please check your .env file")
            logging.error(f"Current .env path: {env_path}")
            return

        logging.info("Starting main loop with credentials:")
        logging.info(f"API_ID exists: {bool(os.getenv('TELEGRAM_API_ID'))}")
        logging.info(f"API_HASH exists: {bool(os.getenv('TELEGRAM_API_HASH'))}")

        while True:
            try:
                await run_analysis_cycle()
                
                # Wait for 10 seconds before next cycle
                logging.info("Waiting 10 seconds before next cycle...")
                await asyncio.sleep(10)  # 10 seconds
                
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}", exc_info=True)
                await asyncio.sleep(60)  # Wait 1 minute before retrying if there's an error
    
    except Exception as e:
        logging.error(f"Critical error in main: {str(e)}", exc_info=True)

if __name__ == "__main__":
    # Add better startup logging
    logging.info("="*50)
    logging.info("Starting Token Finder Script")
    logging.info(f"Current Directory: {current_dir}")
    logging.info(f"Root Directory: {root_dir}")
    logging.info(f"Environment File Path: {env_path}")
    logging.info("="*50)
    
    asyncio.run(main())