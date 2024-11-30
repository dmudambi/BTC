import os
import sys
import asyncio
import subprocess
from datetime import datetime
import time
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('token_finder.log'),
        logging.StreamHandler()
    ]
)

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

def plot_price_and_fib_levels(imported_ohlcv_data, fib_levels, initial_timeframe='15m', 
                             target_fib_level=0.786, min_market_cap=200000, 
                             collect_tokens=None, display_plots=False):
    """
    Analyze price data and Fibonacci levels for multiple tokens.
    """
    if collect_tokens is None:
        collect_tokens = []

    for token_address, token_data in imported_ohlcv_data.items():
        try:
            # Extract price data
            df = token_data[initial_timeframe]
            
            # Check market cap
            if 'market_cap' in df.columns and df['market_cap'].iloc[-1] < min_market_cap:
                continue

            # Find highest and lowest points
            highest_price = df['high'].max()
            lowest_price = df['low'].min()
            price_range = highest_price - lowest_price

            # Calculate Fibonacci levels
            fib_price_levels = {level: highest_price - (price_range * level) 
                              for level in fib_levels}

            # Get current price
            current_price = df['close'].iloc[-1]

            # Check if price is near target Fibonacci level
            target_price = fib_price_levels[target_fib_level]
            price_deviation = 0.02  # 2% deviation allowed

            if (target_price * (1 - price_deviation) <= current_price <= 
                target_price * (1 + price_deviation)):
                collect_tokens.append({
                    'token_address': token_address,
                    'current_price': current_price,
                    'target_price': target_price,
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
    
    # Create CSV file with headers if it doesn't exist
    if not os.path.exists(csv_file):
        with open(csv_file, 'w') as f:
            f.write('timestamp,token_address,current_price,target_price,market_cap,wallet_concentration\n')
    
    for token in qualifying_tokens:
        try:
            # Initialize PirbViewBot for bubblemap analysis
            bot = PirbViewBot()
            
            # Get bubblemap data
            bubblemap_data = await bot.get_bubblemap_data(token['token_address'])
            
            if not bubblemap_data:
                continue

            # Analyze wallet concentration
            total_holders = sum(holder['value'] for holder in bubblemap_data)
            largest_holder = max(holder['value'] for holder in bubblemap_data)
            
            # Calculate concentration percentage
            concentration = (largest_holder / total_holders) * 100
            
            # Check if concentration is below threshold
            if concentration <= max_cluster_percentage:
                token['wallet_concentration'] = concentration
                results.append(token)
                
                timestamp = datetime.now()
                
                # Save to txt file (keeping existing functionality)
                with open('tokens_bubblemap_passed.txt', 'a') as f:
                    f.write(f"{timestamp}, {token['token_address']}, "
                           f"Concentration: {concentration}%\n")
                
                # Save to CSV file
                with open(csv_file, 'a') as f:
                    f.write(f"{timestamp},"
                           f"{token['token_address']},"
                           f"{token['current_price']},"
                           f"{token['target_price']},"
                           f"{token['market_cap']},"
                           f"{concentration}\n")
                
                logging.info(f"Token {token['token_address']} saved to both txt and csv files")
                
        except Exception as e:
            logging.error(f"Error in bubblemap analysis for {token['token_address']}: {str(e)}")
            continue
            
    return results

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
        
        try:
            # Run process from the correct directory
            original_dir = os.getcwd()
            dataset_generator_dir = os.path.dirname(python_file_path)
            os.chdir(dataset_generator_dir)
            logging.info(f"Changed directory to: {dataset_generator_dir}")
            
            # Modified subprocess execution
            process = subprocess.Popen(
                ['python', '-u', os.path.basename(python_file_path)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
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
        base_path = os.path.join(current_dir, 'Data', 'New_Token_Data')
        current_date = datetime.now().strftime('%Y_%m_%d')
        date_folder = os.path.join(base_path, current_date)
        ohlcv_folder = os.path.join(date_folder, 'OHLCV_Data')
        
        # Add debug logging for paths
        logging.info(f"Looking for OHLCV data in: {ohlcv_folder}")
        logging.info(f"Folder exists: {os.path.exists(ohlcv_folder)}")
        
        ohlcv_datetime_folder = Master_Functions.get_most_recent_folder(ohlcv_folder)
        logging.info(f"Most recent OHLCV folder: {ohlcv_datetime_folder}")
        
        imported_ohlcv_data = Master_Functions.import_ohlcv_data(ohlcv_datetime_folder)

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
            min_market_cap=200000,
            collect_tokens=qualifying_tokens,
            display_plots=False  # Disable plots for automated running
        )
        
        logging.info(f"Found {len(qualifying_tokens)} qualifying tokens")
        
        # Step 5: Run bubblemap analysis
        if qualifying_tokens:
            logging.info("Running bubblemap analysis...")
            MAX_CLUSTER_PERCENTAGE = 5
            results = await analyze_qualifying_tokens(qualifying_tokens, max_cluster_percentage=MAX_CLUSTER_PERCENTAGE)
            
            if results:
                logging.info(f"Found {len(results)} tokens passing all criteria")
                for result in results:
                    logging.info(f"Token found: {result['token_address']}")
            else:
                logging.info("No tokens passed all criteria")
        
        logging.info("Analysis cycle completed")
        
    except Exception as e:
        logging.error(f"Error in analysis cycle: {str(e)}", exc_info=True)

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
                
                # Wait for 5 minutes before next cycle
                logging.info("Waiting 5 minutes before next cycle...")
                await asyncio.sleep(300)  # 5 minutes
                
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
