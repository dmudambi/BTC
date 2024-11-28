#### Imports ####

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
current_dir = os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(root_dir)
import Birdeye.Basics.Master_Functions as Master_Functions
# Add rugcheck import
sys.path.append(os.path.join(root_dir, 'APIs', 'Rugcheck'))
from rugcheck_monitor import get_token_risk_report, MAX_RISK_SCORE
import time



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

# Get Token Overview Data
token_overview_data = Master_Functions.get_token_trade_data_multi(new_tokens_address, Master_Functions.API_Key)

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

# Get market data for all tokens in new_tokens_filtered_overview_address
market_data_list = []
for address in new_tokens_filtered_overview_address:
    market_data_df = Master_Functions.get_token_market_data(address, Master_Functions.API_Key)
    if not market_data_df.empty:
        # Convert numeric columns to float
        for col in ['Liquidity', 'Market Cap']:
            if col in market_data_df.columns:
                market_data_df[col] = market_data_df[col].astype(float)
        market_data_df.index = [address]  # Set index to address
        market_data_list.append(market_data_df)

# Combine all market data into a single DataFrame
if market_data_list:
    all_market_data_df = pd.concat(market_data_list)
else:
    all_market_data_df = pd.DataFrame(index=new_tokens_filtered_overview_address)

# Add rugcheck assessment function with rate limiting
def add_rugcheck_data(df):
    """Add rugcheck risk assessment data to the dataframe with rate limiting"""
    risk_scores = []
    risk_status = []
    
    print("\nPerforming RugCheck Assessment...")
    for address in df.index:
        print(f"Checking {address}...")
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                report = get_token_risk_report(address)
                if report:
                    score = report.get('score', 0)
                    risk_scores.append(score)
                    risk_status.append('LOW RISK' if score <= MAX_RISK_SCORE else 'HIGH RISK')
                else:
                    risk_scores.append(None)
                    risk_status.append('UNKNOWN')
                time.sleep(1)  # Add 1 second delay between requests
                break
            except Exception as e:
                if 'Too Many Requests' in str(e) and attempt < max_retries - 1:
                    print(f"Rate limited, waiting {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Error checking {address}: {e}")
                    risk_scores.append(None)
                    risk_status.append('UNKNOWN')
                    break
    
    df['RugCheck_Score'] = risk_scores
    df['Risk_Status'] = risk_status
    return df

# Add rugcheck data before applying market cap filters
new_tokens_mc_added = add_rugcheck_data(new_tokens_filtered_overview)

# Join market data after rugcheck assessment
new_tokens_mc_added = new_tokens_mc_added.join(
    all_market_data_df,
    how='left'
)

# Debug prints
print("\nBefore filtering:")
print(f"Total tokens: {len(new_tokens_mc_added)}")
print("\nSample of data:")
print(new_tokens_mc_added[['Liquidity', 'Market Cap', 'RugCheck_Score', 'Risk_Status']].head())

# Update filtering with proper error handling
filtered_tokens = new_tokens_mc_added[
    (new_tokens_mc_added['Liquidity'].astype(float) >= Master_Functions.new_token_min_liquidity) &
    (new_tokens_mc_added['Liquidity'].astype(float) <= Master_Functions.new_token_max_liquidity) &
    (new_tokens_mc_added['Market Cap'].astype(float) >= Master_Functions.new_token_min_market_cap) &
    (new_tokens_mc_added['Market Cap'].astype(float) <= Master_Functions.new_token_max_market_cap) &
    (new_tokens_mc_added['Risk_Status'] == 'LOW RISK')
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

# Retrieve and save OHLCV data for each token and timeframe
for address in token_addresses:
    # Create a folder for the token
    token_folder = os.path.join(ohlcv_datetime_folder, address)
    os.makedirs(token_folder, exist_ok=True)
    # Retrieve OHLCV data for all timeframes
    ohlcv_data = Master_Functions.get_ohlcv_data_multi([address], Master_Functions.API_Key, Master_Functions.timeframes)
    # Save OHLCV data for each timeframe
    for timeframe, df in ohlcv_data[address].items():
        if not df.empty:
            filename = f"{timeframe}.csv"
            file_path = os.path.join(token_folder, filename)
            df.to_csv(file_path)
            #print(f"OHLCV data for {address} ({timeframe}) saved to: {file_path}")
            print(f"OHLCV data retrieval and storage completed for {address} ({timeframe}).")
        else:
            print(f"No OHLCV data available for {address} ({timeframe})")

print("\n--------------------------------\nData Processing Complete\n--------------------------------\n")