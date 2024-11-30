#### Imports ####

import sys
import os
from datetime import datetime, timedelta

# Setup proper path resolution
current_dir = os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))  # Remove one level since we're already in Bots/Fib
sys.path.append(root_dir)  # Add root directory to Python path

# Now we can import our local modules
import Birdeye.Basics.Master_Functions_Asynco as Master_Functions

# Add rugcheck import with corrected path
sys.path.append(os.path.join(root_dir, 'APIs', 'Rugcheck'))
from rugcheck_monitor import get_token_risk_report, MAX_RISK_SCORE, get_token_risk_report_async

import pandas as pd
import time
import concurrent.futures
import aiohttp
import asyncio
import sys
sys.stdout.reconfigure(line_buffering=True)  # Force line buffering for stdout

# Add debug prints to verify paths
print(f"Current Directory: {current_dir}")
print(f"Root Directory: {root_dir}")
print(f"Python Path: {sys.path}")

# Force immediate flushing of print statements
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Configure logging to show everything
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    force=True
)

#### Prints ####   

print("\n--------------------------------\n")
print(f"Running on: {Master_Functions.chain}")
print("\n--------------------------------\n")
print("NEW TOKEN FILTERS:")
print(f"\nMin Liquidity: {Master_Functions.format_number(Master_Functions.new_token_min_liquidity)}")
print(f"Max Liquidity: {Master_Functions.format_number(Master_Functions.new_token_max_liquidity)}")
print(f"Min Market Cap: {Master_Functions.format_number(Master_Functions.new_token_min_market_cap)}")
print(f"Max Market Cap: {Master_Functions.format_number(Master_Functions.new_token_max_market_cap)}")
print(f"Filtering New Tokens for: {Master_Functions.days_back} days, {Master_Functions.hours_back} hours, {Master_Functions.minutes_back} minutes back")
print(f"Token Launch Liquidity Filter: {Master_Functions.format_number(Master_Functions.new_token_liquidity_filter)}")
print(f"RugCheck Risk Score Threshold: {MAX_RISK_SCORE}")
print("\n--------------------------------\n \n Collecting New Token Listings...\n \n--------------------------------\n")



#### New Token Listings Summary ####

# Create the optimized folder structure
data_folder = 'Data'
new_token_data_folder = os.path.join(data_folder, 'New_Token_Data')
current_date = datetime.now().strftime('%Y_%m_%d')  # Format: YYYY_MM_DD
date_folder = os.path.join(new_token_data_folder, current_date)

# Create subfolders for different data types
ohlcv_folder = os.path.join(date_folder, 'OHLCV_Data')
token_summary_folder = os.path.join(date_folder, 'Token_Summary')
live_trades_folder = os.path.join(date_folder, 'Live_Trades')
backtesting_results_folder = os.path.join(date_folder, 'Backtesting_Results')

# Create a single timestamp for both Token_Summary and OHLCV_Data
current_datetime = datetime.now().strftime('%Y_%b_%d_%I%M%p')  # Format: YYYY_Mon_DD_HHMM(AM/PM)
ohlcv_datetime_folder = os.path.join(ohlcv_folder, current_datetime)
token_summary_datetime_folder = os.path.join(token_summary_folder, current_datetime)

# Create the folders if they don't exist
for folder in [data_folder, new_token_data_folder, date_folder, ohlcv_datetime_folder, token_summary_datetime_folder, live_trades_folder, backtesting_results_folder]:
    os.makedirs(folder, exist_ok=True)

# Adjust days_back for the function call
adjusted_days_back = Master_Functions.days_back + 1
# Example usage:
new_tokens = Master_Functions.get_new_listings(adjusted_days_back, Master_Functions.hours_back, Master_Functions.minutes_back, Master_Functions.API_Key, Master_Functions.new_token_liquidity_filter)
new_tokens_filtered = new_tokens[1].rename(columns={
    'address': 'Address'
})

# Get List of New Tokens Addresses
new_tokens_address = new_tokens_filtered['Address'].tolist()

# Set Index
new_tokens_filtered = new_tokens_filtered.set_index('Address')

# Modify get_token_trade_data_multi to use concurrent processing
async def get_token_trade_data_async(session, address, api_key):
    try:
        return await Master_Functions.get_token_trade_data_async(session, address, api_key)
    except Exception as e:
        print(f"Error getting trade data for {address}: {e}")
        return None

async def get_all_token_trade_data(addresses, api_key):
    async with aiohttp.ClientSession() as session:
        tasks = [get_token_trade_data_async(session, address, api_key) for address in addresses]
        results = await asyncio.gather(*tasks)
        return {addr: result for addr, result in zip(addresses, results) if result is not None}

# Replace the synchronous call with async version
token_overview_data = asyncio.run(get_all_token_trade_data(new_tokens_address, Master_Functions.API_Key))

# Modify the attributes processing
all_attributes = set()
for token, df in token_overview_data.items():
    if isinstance(df, pd.DataFrame) and not df.empty:
        # Convert all attributes to strings before adding to set
        all_attributes.update(str(attr) for attr in df['Attribute'].unique())

# Convert the set to a sorted list
columns_array = sorted(list(all_attributes))

# Create the master DataFrame
new_tokens_master_overview = pd.DataFrame(index=token_overview_data.keys(), columns=columns_array)

# Fill the DataFrame with values
for token, df in token_overview_data.items():
    if isinstance(df, pd.DataFrame) and not df.empty:
        for _, row in df.iterrows():
            new_tokens_master_overview.at[token, row['Attribute']] = row['Value']

# Set the index name to 'Address'
new_tokens_master_overview.index.name = 'Address'

# Filter out rows where 'Error' column is not NaN, if the column exists
if 'Error' in new_tokens_master_overview.columns:
    new_tokens_master_overview = new_tokens_master_overview[new_tokens_master_overview['Error'].isna()]
new_tokens_master_overview["liquidityAddedAt"] = new_tokens_filtered["liquidityAddedAt"]
new_tokens_filtered_overview = new_tokens_master_overview

# Get the list of addresses from new_tokens_master_overview
new_tokens_filtered_overview_address = new_tokens_filtered_overview.index.tolist()

# Modify market data retrieval to use concurrent processing
async def get_market_data_async(session, address, api_key):
    try:
        return await Master_Functions.get_token_market_data_async(session, address, api_key)
    except Exception as e:
        print(f"Error getting market data for {address}: {e}")
        return None

async def get_all_market_data(addresses, api_key):
    async with aiohttp.ClientSession() as session:
        tasks = [get_market_data_async(session, address, api_key) for address in addresses]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Create a DataFrame with all addresses as index
        market_data_df = pd.DataFrame(index=addresses)
        
        for address, result in zip(addresses, results):
            if isinstance(result, Exception):
                print(f"Error processing {address}: {result}")
                continue
            if result is not None and not result.empty:
                # Ensure we're using the renamed columns from get_token_market_data_async
                for col in result.columns:
                    # Add suffix '_market' to avoid column name conflicts
                    market_data_df.at[address, f"{col}_market"] = result.iloc[0][col]
        
        return market_data_df

# First, define the rugcheck functions (move these up before line 150)
async def check_token_risk_async(session, address):
    try:
        result = await get_token_risk_report_async(address)
        if result:
            return {
                'score': result.get('score', 0),
                'risks': result.get('risks', [])
            }
    except Exception as e:
        print(f"Error checking {address}: {e}")
    return None

async def check_all_tokens_risk(addresses):
    results = {}
    for address in addresses:
        try:
            result = await get_token_risk_report_async(address)
            if result:
                results[address] = result
            await asyncio.sleep(0.08)  # Rate limiting
        except Exception as e:
            print(f"Error checking {address}: {e}")
            results[address] = None
    return results

def add_rugcheck_data(df):
    print("\nPerforming RugCheck Assessment...")
    results = asyncio.run(check_all_tokens_risk(df.index))
    
    risk_scores = []
    risk_status = []
    
    for address in df.index:
        result = results.get(address)
        if result:
            score = result.get('score', 0)
            risk_scores.append(score)
            risk_status.append('LOW RISK' if score <= MAX_RISK_SCORE else 'HIGH RISK')
        else:
            risk_scores.append(None)
            risk_status.append('UNKNOWN')
    
    df['RugCheck_Score'] = risk_scores
    df['Risk_Status'] = risk_status
    
    return df

# Then use the functions (around line 150)
market_data_df = asyncio.run(get_all_market_data(new_tokens_filtered_overview_address, Master_Functions.API_Key))
new_tokens_mc_added = new_tokens_filtered_overview.join(market_data_df, how='left')

# After joining, you might want to rename some columns to remove the suffix
columns_to_rename = {
    'Price_market': 'Price',
    'Liquidity_market': 'Liquidity',
    'Market Cap_market': 'Market Cap',
    'Total Supply_market': 'Total Supply',
    'Circulating Supply_market': 'Circulating Supply',
    'Circulating Market Cap_market': 'Circulating Market Cap'
}
new_tokens_mc_added = new_tokens_mc_added.rename(columns=columns_to_rename)

# Now add rugcheck data
new_tokens_mc_added = add_rugcheck_data(new_tokens_mc_added)

# Then print the debug information
print("\nBefore filtering:")
print(f"Total tokens: {len(new_tokens_mc_added)}")
print("\nSample of data:")
print(new_tokens_mc_added[['Liquidity', 'Market Cap', 'RugCheck_Score', 'Risk_Status']].head())

# Then update the filtering section to include rugcheck criteria
filtered_tokens = new_tokens_mc_added[
    new_tokens_mc_added['Liquidity'].notna() &
    new_tokens_mc_added['Market Cap'].notna() &
    (new_tokens_mc_added['Liquidity'].astype(float) >= Master_Functions.new_token_min_liquidity) &
    (new_tokens_mc_added['Liquidity'].astype(float) <= Master_Functions.new_token_max_liquidity) &
    (new_tokens_mc_added['Market Cap'].astype(float) >= Master_Functions.new_token_min_market_cap) &
    (new_tokens_mc_added['Market Cap'].astype(float) <= Master_Functions.new_token_max_market_cap) &
    (new_tokens_mc_added['Risk_Status'] == 'LOW RISK')  # Add rugcheck filter
].copy()

# More debug prints
print("\nAfter filtering:")
print(f"Remaining tokens: {len(filtered_tokens)}")
print("\nFiltered tokens data:")
for idx, row in filtered_tokens.iterrows():
    print(f"\nToken: {idx}")
    print(f"Liquidity: {row['Liquidity']}")
    print(f"Market Cap: {row['Market Cap']}")
    print(f"RugCheck Score: {row['RugCheck_Score']}")

# Save the filtered data with market cap range and time range in the filename
mc_range = f"{Master_Functions.new_token_min_market_cap/1e6:.1f}M-{Master_Functions.new_token_max_market_cap/1e6:.1f}M"
time_range = f"{Master_Functions.days_back}d_{Master_Functions.hours_back}h_{Master_Functions.minutes_back}m"
filename = f"new_tokens_mc_added_filtered_{mc_range}_{time_range}.csv"
file_path = os.path.join(token_summary_datetime_folder, filename)
filtered_tokens.to_csv(file_path)

# Sort the DataFrame by 'Market Cap' in descending order
filtered_tokens_sorted = filtered_tokens.sort_values(by='Market Cap', ascending=False)

# Print each address with its corresponding Market Cap in descending order
for address in filtered_tokens_sorted.index:
    market_cap = filtered_tokens_sorted.at[address, 'Market Cap']
    print(f"https://dexscreener.com/solana/{address} - Market Cap: {Master_Functions.format_number(market_cap)}")
print(f"\nToken summary data saved to: {file_path}")

# Print the number of items in the "Address" column
num_addresses = filtered_tokens.index.size
print(f"\nNumber of Tokens Filtered: {num_addresses}")
print(f"\n--------------------------------\n \nAll Token summary data saved to: {file_path}\n \n--------------------------------\n")




#### OHLCV Data Retrieval and Storage ####

# Print the timeframes being used for OHLCV data
print(f"Timeframes being used to retrieve OHLCV data: {', '.join(Master_Functions.timeframes)}")
print(f"\n--------------------------------\n")

# Get the list of token addresses
token_addresses = filtered_tokens.index.tolist()

# Modify OHLCV data retrieval to use concurrent processing
async def get_ohlcv_async(session, address, timeframe, api_key):
    try:
        return await Master_Functions.get_ohlcv_data_async(session, address, timeframe, api_key)
    except Exception as e:
        print(f"Error getting OHLCV data for {address} ({timeframe}): {e}")
        return None

async def process_token_ohlcv(address, token_folder):
    try:
        async with aiohttp.ClientSession() as session:
            tasks = []
            for timeframe in Master_Functions.timeframes:
                if tasks:  # If not the first request
                    await asyncio.sleep(0.08)  # 80ms between timeframes
                tasks.append(get_ohlcv_async(session, address, timeframe, Master_Functions.API_Key))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success = False
            for timeframe, result in zip(Master_Functions.timeframes, results):
                if isinstance(result, Exception):
                    print(f"Error for {address} ({timeframe}): {result}")
                    continue
                    
                if result is not None and not result.empty:
                    filename = f"{timeframe}.csv"
                    file_path = os.path.join(token_folder, filename)
                    result.to_csv(file_path)
                    success = True
                    print(f"OHLCV data saved for {address} ({timeframe}).")
            
            return success
    except Exception as e:
        print(f"Critical error processing {address}: {e}")
        return False

# Replace the OHLCV processing loop
async def process_all_tokens_ohlcv():
    tasks = []
    for address in token_addresses:
        token_folder = os.path.join(ohlcv_datetime_folder, address)
        os.makedirs(token_folder, exist_ok=True)
        tasks.append(process_token_ohlcv(address, token_folder))
    
    # Add error handling for the gather
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for address, result in zip(token_addresses, results):
            if isinstance(result, Exception):
                print(f"Error processing OHLCV data for {address}: {result}")
    except Exception as e:
        print(f"Error in OHLCV processing: {e}")

asyncio.run(process_all_tokens_ohlcv())

print("\n--------------------------------\nData Processing Complete\n--------------------------------\n")

# Add batch processing for tokens
BATCH_SIZE = 1  # Process 5 tokens at a time

async def process_tokens_in_batches(tokens):
    all_results = []
    
    for i in range(0, len(tokens), BATCH_SIZE):
        batch = tokens[i:i + BATCH_SIZE]
        
        async with aiohttp.ClientSession() as session:
            tasks = [get_token_risk_report_async(token, session) for token in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            all_results.extend(batch_results)
            
        # Add delay between batches
        await asyncio.sleep(0.08)  # 80ms delay between batches
        
    return all_results

# Replace your existing token processing code with:
results = asyncio.run(process_tokens_in_batches(token_addresses))