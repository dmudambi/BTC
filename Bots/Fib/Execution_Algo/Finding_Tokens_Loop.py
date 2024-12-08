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

MAX_CLUSTER_PERCENTAGE = 6

# Add a rate limit filter to the main logger
class RateLimitFilter(logging.Filter):
    def filter(self, record):
        return "Rate limited" not in record.getMessage()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('token_finder.log'),
        logging.StreamHandler()
    ]
)

# Add filter to all handlers
for handler in logging.getLogger().handlers:
    handler.addFilter(RateLimitFilter())

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
        logging.info("Starting new analysis cycle")
        
        # Step 1: Run Dataset Generator
        logging.info("Running Dataset Generator...")
        python_file_path = os.path.join(root_dir, 'Bots', 'Fib', 'Dataset_Generator_Rugcheck_Automated.py')
        
        # Add debug logging
        logging.info(f"Dataset Generator path: {python_file_path}")
        logging.info(f"File exists: {os.path.exists(python_file_path)}")
        
        # Set up environment with correct PYTHONPATH
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{root_dir}:{os.path.join(root_dir, 'Bots', 'Fib')}"
        logging.info(f"PYTHONPATH set to: {env['PYTHONPATH']}")
        
        # Get current time and calculate start time for filtering
        end_time = datetime.now(pytz.UTC)
        start_time = end_time - timedelta(
            days=Master_Functions.days_back,
            hours=Master_Functions.hours_back,
            minutes=Master_Functions.minutes_back
        )
        
        # Log the start and end times in human-readable format
        logging.info(f"Filtering data from: {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')} to {end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        try:
            # Run process from the correct directory
            original_dir = os.getcwd()
            dataset_generator_dir = os.path.dirname(python_file_path)
            os.chdir(dataset_generator_dir)
            logging.info(f"Changed directory to: {dataset_generator_dir}")
            
            # Pass parameters to the subprocess
            process_args = [
                'python', '-u', os.path.basename(python_file_path),
                '--days_back', str(Master_Functions.days_back),
                '--hours_back', str(Master_Functions.hours_back),
                '--minutes_back', str(Master_Functions.minutes_back)
            ]

            process = subprocess.Popen(
                process_args,  # Use modified arguments
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=dataset_generator_dir,
                universal_newlines=True
            )

            # Create async tasks to read both stdout and stderr
            async def read_stream(stream, is_error=False):
                while True:
                    line = stream.readline()
                    if not line:
                        break
                    if is_error:
                        logging.error(f"Dataset Generator Error: {line.strip()}")
                    else:
                        logging.info(f"Dataset Generator: {line.strip()}")

            # Create tasks for both streams
            import asyncio
            stdout_task = asyncio.create_task(read_stream(process.stdout))
            stderr_task = asyncio.create_task(read_stream(process.stderr, True))

            # Wait for both streams to complete
            await asyncio.gather(stdout_task, stderr_task)

            # Wait for process to complete
            process.wait()

            if process.returncode != 0:
                logging.error(f"Dataset Generator failed with return code: {process.returncode}")
                return

            logging.info("Dataset Generator completed successfully")

        finally:
            os.chdir(original_dir)
            logging.info(f"Returned to directory: {original_dir}")
            
        # Step 2: Import latest data
        logging.info("Importing latest OHLCV data...")
        
        # Use the correct path relative to the Dataset Generator output
        base_path = os.path.join(root_dir, 'Bots', 'Fib', 'Data', 'New_Token_Data')
        current_date = datetime.now().strftime('%Y_%m_%d')
        date_folder = os.path.join(base_path, current_date)
        ohlcv_folder = os.path.join(date_folder, 'OHLCV_Data')
        
        logging.info(f"Looking for OHLCV data in: {ohlcv_folder}")
        
        # Add retry logic for finding the OHLCV folder
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                if not os.path.exists(ohlcv_folder):
                    raise FileNotFoundError(f"OHLCV folder does not exist: {ohlcv_folder}")
                
                ohlcv_datetime_folder = get_most_recent_folder(ohlcv_folder)
                logging.info(f"Using OHLCV folder: {ohlcv_datetime_folder}")
                
                # Verify that we have data to process
                token_folders = [f for f in os.listdir(ohlcv_datetime_folder) 
                               if os.path.isdir(os.path.join(ohlcv_datetime_folder, f))]
                
                if not token_folders:
                    raise FileNotFoundError(f"No token data found in: {ohlcv_datetime_folder}")
                
                logging.info(f"Found {len(token_folders)} token folders to process")
                
                # Import OHLCV data
                imported_ohlcv_data = Master_Functions.import_ohlcv_data(ohlcv_datetime_folder)
                
                if not imported_ohlcv_data:
                    raise ValueError("No OHLCV data was imported")
                    
                logging.info(f"Successfully imported OHLCV data for {len(imported_ohlcv_data)} tokens")
                break  # Success - exit retry loop
                
            except (FileNotFoundError, ValueError) as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                    logging.info(f"Waiting {retry_delay} seconds before retry...")
                    await asyncio.sleep(retry_delay)
                else:
                    logging.error(f"Failed to import OHLCV data after {max_retries} attempts: {str(e)}")
                    return  # Exit the analysis cycle
        
        # Continue with the rest of the analysis cycle
        # Step 3: Initialize analysis parameters
        qualifying_tokens = []
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786, 0.886]
        
        # Step 4: Run Fibonacci analysis
        logging.info("Running Fibonacci analysis...")
        plot_price_and_fib_levels(
            imported_ohlcv_data=imported_ohlcv_data,
            fib_levels=fib_levels,
            initial_timeframe='15m',
            target_fib_level=0.786,
            collect_tokens=qualifying_tokens,
            display_plots=False  # Disable plots for automated running
        )
        
        logging.info(f"Found {len(qualifying_tokens)} qualifying tokens")
        
        # After finding qualifying tokens but before bubblemap analysis:
        if qualifying_tokens:
            # Load previously qualified tokens
            previously_qualified = load_qualified_tokens()
            
            # Filter out previously qualified tokens by comparing addresses only
            new_qualifying_tokens = [
                token for token in qualifying_tokens 
                if token['token_address'] not in previously_qualified
            ]
            
            if not new_qualifying_tokens:
                logging.info("All qualifying tokens have been previously processed")
                return
                
            logging.info(f"Found {len(new_qualifying_tokens)} new qualifying tokens")
            
            # Run bubblemap analysis only on new tokens
            logging.info("Running bubblemap analysis...")
            results = await analyze_qualifying_tokens(new_qualifying_tokens, max_cluster_percentage=MAX_CLUSTER_PERCENTAGE)
            
            if results:
                logging.info(f"Found {len(results)} tokens passing all criteria")
                for result in results:
                    token_address = result['token_address']
                    logging.info(f"Token found: {token_address}")
                    # Save newly qualified tokens
                    save_qualified_token(token_address)
            else:
                logging.info("No tokens passed all criteria")
        
        logging.info("Analysis cycle completed")
        
    except Exception as e:
        logging.error(f"Error in analysis cycle: {str(e)}")
        traceback.print_exc()  # Add stack trace for debugging

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