import requests
import sys
import json
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction 
from solana.rpc.api import Client
from solana.rpc.types import TxOpts
from pprint import pprint
from functools import lru_cache
from datetime import datetime, timedelta
import pandas as pd
import json
import pandas as pd
from pprint import pprint 
from IPython.display import display, HTML
import os
from datetime import datetime
import glob
import pytz
import time

current_dir = os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(root_dir)
import Birdeye.Basics.dontshare as d


#### API Keys and Wallets ####
API_Key = d.birdeye
wallet = d.sol_wallet
chain = "solana" 
# Choose between: solana,ethereum,arbitrum,avalanche,bsc,optimism,polygon,base,zksync
multichain = "solana,ethereum,bsc"

#### PRE-COLLECTION FILTERS ####
days_back = 2 # 0 for last 24 hours, 1 for 24-48 hours ago, 2 for 48-72 hours ago (max 2)
hours_back = 12  # 0-23 hours back within the selected day
minutes_back = 0  # 0-59 minutes back within the selected hour
new_token_liquidity_filter = 30000 # Minimum liquidity in USD for new tokens

#### POST-COLLECTION FILTERS ####
new_token_min_liquidity = 30000  # Minimum liquidity in USD for new tokens
new_token_max_liquidity = 1000000  # Maximum liquidity in USD for new tokens
new_token_min_market_cap = 300000  # Minimum market cap in USD for new tokens
new_token_max_market_cap = 2500000  # Maximum market cap in USD for new tokens

#### OHLCV DATA #### 
timeframes = ['1m', '5m', '15m', '1H'] 
# ['1m', '3m', '5m', '15m', '30m', '1H', '2H', '4H', '6H', '8H', '12H', '1D']



### DEFI 

# Data Gathering Functions
# Function to get the most recent folder
def get_most_recent_folder(base_path):
    folders = glob.glob(os.path.join(base_path, '*'))
    return max(folders, key=os.path.getctime)

# Import OHLCV data for analysis
def import_ohlcv_data(ohlcv_datetime_folder):
    ohlcv_data = {}
    
    # Get all token folders
    token_folders = glob.glob(os.path.join(ohlcv_datetime_folder, '*'))
    
    for token_folder in token_folders:
        token_address = os.path.basename(token_folder)
        ohlcv_data[token_address] = {}
        
        # Get all CSV files in the token folder
        csv_files = glob.glob(os.path.join(token_folder, '*.csv'))
        
        for csv_file in csv_files:
            timeframe = os.path.splitext(os.path.basename(csv_file))[0]
            df = pd.read_csv(csv_file)
            
            # Rename columns to standard OHLCV names
            df = df.rename(columns={
                'datetime': 'timestamp',
                'c': 'close',
                'h': 'high',
                'l': 'low',
                'o': 'open',
                'v': 'volume'
            })
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            ohlcv_data[token_address][timeframe] = df
    
    return ohlcv_data

# Function to format numbers into human-readable format
def format_number(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}Bil"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}Mil"
    elif num >= 1_000:
        return f"{num / 1_000:.0f}K"
    else:
        return str(num)
# OHLCV Data

def get_ohlcv_data_multi(tokens, API_Key, timeframes=None):
    """
    Fetch OHLCV data for multiple tokens and specified timeframes.

    Args:
    tokens (list): List of token addresses to fetch data for.
    API_Key (str): Your Birdeye API key.
    timeframes (list): List of timeframes to fetch data for. Default is ['15m'].

    Returns:
    dict: A nested dictionary with tokens and timeframes as keys and DataFrames as values.
    """
    if timeframes is None:
        timeframes = ['15m']

    timeframes_data = {
        '1m': 1 * 24 * 60,    # 1 day of 1-minute data
        '3m': 3 * 24 * 20,    # 3 days of 3-minute data
        '5m': 7 * 24 * 12,    # 7 days of 5-minute data
        '15m': 25 * 24 * 4,   # 25 days of 15-minute data
        '30m': 50 * 24 * 2,   # 50 days of 30-minute data
        '1H': 100 * 24,       # 100 days of 1-hour data
        '2H': 200 * 12,       # 200 days of 2-hour data
        '4H': 400 * 6,        # 400 days of 4-hour data
        '6H': 600 * 4,        # 600 days of 6-hour data
        '8H': 800 * 3,        # 800 days of 8-hour data
        '12H': 1200 * 2,      # 1200 days of 12-hour data
        '1D': 5 * 365         # 5 years of daily data
    }

    results = {}

    for token in tokens:
        results[token] = {}
        for timeframe in timeframes:
            if timeframe not in timeframes_data:
                print(f"Invalid timeframe: {timeframe}. Skipping.")
                continue

            # Calculate start and end times
            end_time = datetime.now(pytz.UTC)
            start_time = end_time - timedelta(hours=timeframes_data[timeframe])

            # Convert to Unix timestamps
            time_from = int(start_time.timestamp())
            time_to = int(end_time.timestamp())

            url = f"https://public-api.birdeye.so/defi/ohlcv?address={token}&type={timeframe}&time_from={time_from}&time_to={time_to}"

            headers = {
                "accept": "application/json",
                "X-API-KEY": API_Key
            }

            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                try:
                    data = json.loads(response.text)
                    
                    if 'data' in data and 'items' in data['data']:
                        items = data['data']['items']
                        
                        if items:
                            df = pd.DataFrame(items)
                            df['datetime'] = pd.to_datetime(df['unixTime'], unit='s')
                            df.set_index('datetime', inplace=True)
                            columns_order = [col for col in df.columns if col != 'unixTime']
                            df = df[columns_order]

                            results[token][timeframe] = df
                            
                            # Print the head of each token-timeframe table
                            #print(f"\nHead of {token} - {timeframe} table:")
                            #print(df.head())
                            
                            # Print that the token-timeframe data was successfully retrieved
                            print(f"\nSuccessfully retrieved {token} - {timeframe} data.")
                        else:
                            results[token][timeframe] = pd.DataFrame()
                            print(f"\nNo data items found for token {token}, timeframe {timeframe}.")
                    else:
                        results[token][timeframe] = pd.DataFrame()
                        print(f"\nUnexpected response structure for token {token}, timeframe {timeframe}.")
                except json.JSONDecodeError:
                    results[token][timeframe] = pd.DataFrame()
                    print(f"\nFailed to parse JSON response for token {token}, timeframe {timeframe}.")
            else:
                results[token][timeframe] = pd.DataFrame()
                print(f"\nRequest failed with status code: {response.status_code} for token {token}, timeframe {timeframe}.")

    return results

### Token APIs

# Market Cap Filter

def get_token_list(sort_by, sort_type, min_liquidity, min_volume_24h, min_market_cap, max_market_cap, total_tokens, chain, API_Key):
    """
    Fetch and filter token list based on specified criteria.

    Args:
        sort_by (str): Criterion to sort by (e.g., 'mc', 'rank', 'v24hUSD', 'v24hChangePercent').
        sort_type (str): Sort order ('asc' or 'desc').
        min_liquidity (float): Minimum liquidity in USD.
        min_volume_24h (float): Minimum 24-hour trading volume in USD.
        min_market_cap (float): Minimum market cap in USD.
        max_market_cap (float): Maximum market cap in USD.
        total_tokens (int): Number of tokens to retrieve.
        chain (str): Blockchain to query.
        API_Key (str): Your Birdeye API key.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered token list.
    """
    limit = 50
    all_tokens = []
    max_offsets = total_tokens * 3  # Adjust multiplier based on expected data density

    for offset in range(0, max_offsets, limit):
        url = (
            f"https://public-api.birdeye.so/defi/tokenlist?"
            f"sort_by={sort_by}&sort_type={sort_type}&offset={offset}&limit={limit}&min_liquidity={min_liquidity}"
        )

        headers = {
            "accept": "application/json",
            "x-chain": chain,
            "X-API-KEY": API_Key
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raises HTTPError for bad responses
            data = response.json()

            if 'data' in data and 'tokens' in data['data']:
                new_tokens = [
                    {
                        'name': token.get('name'),
                        'symbol': token.get('symbol'),
                        'mc': token.get('mc'),
                        'v24hUSD': token.get('v24hUSD'),
                        'v24hChangePercent': token.get('v24hChangePercent'),
                        'liquidity': token.get('liquidity'),
                        'address': token.get('address')
                    }
                    for token in data['data']['tokens']
                    if token.get('v24hUSD', 0) >= min_volume_24h
                    and min_market_cap <= token.get('mc', 0) <= max_market_cap
                ]
                all_tokens.extend(new_tokens)
            else:
                print(f"Unexpected response structure at offset {offset}: {data}")
                break

            if len(all_tokens) >= total_tokens:
                break

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred at offset {offset}: {http_err}")
            break
        except requests.exceptions.RequestException as req_err:
            print(f"Request exception at offset {offset}: {req_err}")
            break
        except Exception as e:
            print(f"An unexpected error occurred at offset {offset}: {e}")
            break

        # Respect rate limits
        time.sleep(0.08)

    # Trim the list to the desired number of tokens
    all_tokens = all_tokens[:total_tokens]

    # Create DataFrame regardless of whether tokens were found
    df = pd.DataFrame(all_tokens)

    if not df.empty:
        # Optional: Rename columns for clarity
        columns_to_display = {
            'name': 'Name',
            'symbol': 'Symbol',
            'mc': 'Market Cap',
            'v24hUSD': 'Volume 24h',
            'v24hChangePercent': '24h Change (%)',
            'liquidity': 'Liquidity',
            'address': 'Address'
        }

        # Ensure columns exist before renaming
        available_columns = [col for col in columns_to_display.keys() if col in df.columns]
        df = df[available_columns].rename(columns={col: columns_to_display[col] for col in available_columns})

    return df  # Always return a DataFrame, empty or populated

# Token Market Data

def get_token_market_data(address, API_Key):
    """
    Fetch market data for a specific token and return it as a pandas DataFrame.

    Args:
    address (str): The token address to fetch market data for.
    API_Key (str): Your Birdeye API key.

    Returns:
    pd.DataFrame: A DataFrame containing the token's market data.
    """
    url = f"https://public-api.birdeye.so/defi/v3/token/market-data?address={address}"

    headers = {
        "accept": "application/json",
        "X-API-KEY": API_Key
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if 'data' in data and data['success']:
            # Convert the data to a DataFrame
            df = pd.DataFrame([data['data']])
            
            # Rename columns for clarity
            df = df.rename(columns={
                'address': 'Address',
                'price': 'Price',
                'liquidity': 'Liquidity',
                'supply': 'Total Supply',
                'marketcap': 'Market Cap',
                'circulating_supply': 'Circulating Supply',
                'circulating_marketcap': 'Circulating Market Cap'
            })
            
            return df
        else:
            print("Unexpected response structure or request unsuccessful")
            return pd.DataFrame()
    else:
        print(f"Request failed with status code: {response.status_code}")
        return pd.DataFrame()


# Token Security

def get_token_security_data_multi(tokens, API_Key):
    """
    Fetch token security data for multiple tokens.

    Args:
    tokens (list): List of token addresses to fetch data for.
    API_Key (str): Your Birdeye API key.

    Returns:
    dict: A dictionary with token addresses as keys and their security data as values.
    """
    results = {}

    for token in tokens:
        url = f"https://public-api.birdeye.so/defi/token_security?address={token}"

        headers = {
            "accept": "application/json",
            "X-API-KEY": API_Key
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = json.loads(response.text)
            if 'data' in data:
                # Extract the specific values
                token_data = data['data']
                selected_data = {
                    'top10HolderPercent': token_data.get('top10HolderPercent', 'N/A'),
                    'top10UserPercent': token_data.get('top10UserPercent', 'N/A'),
                    'preMarketHolder': token_data.get('preMarketHolder', 'N/A'),
                    'mutableMetadata': token_data.get('mutableMetadata', 'N/A'),
                    'creatorPercentage': token_data.get('creatorPercentage', 'N/A'),
                    'ownerPercentage': token_data.get('ownerPercentage', 'N/A')
                }

                # Create a DataFrame from the selected data
                df_selected = pd.DataFrame.from_dict(selected_data, orient='index', columns=['Value'])
                df_selected.index.name = 'Attribute'
                df_selected.reset_index(inplace=True)

                # Create full data DataFrame
                df_full = pd.json_normalize(data['data'])
                df_full = df_full.map(lambda x: json.dumps(x, indent=2) if isinstance(x, (dict, list)) else x)
                df_full_pivoted = df_full.T.reset_index()
                df_full_pivoted.columns = ['Attribute', 'Value']

                # Store both selected and full data for the token
                results[token] = {
                    'selected_data': df_selected,
                    'full_data': df_full_pivoted
                }
            else:
                results[token] = {
                    'error': "Unexpected response structure. 'data' key not found.",
                    'data': json.dumps(data, indent=2)
                }
        else:
            results[token] = {
                'error': f"Request failed with status code: {response.status_code}",
                'data': response.text
            }

        # Respect rate limits by adding a small delay between requests
        time.sleep(0.08)

    return results


# Token Overview

def get_token_overview_data_multi(tokens, API_Key):
    """
    Fetch token overview data for multiple tokens.

    Args:
    tokens (list): List of token addresses to fetch data for.
    API_Key (str): Your Birdeye API key.

    Returns:
    dict: A dictionary with token addresses as keys and their overview data as pandas DataFrames.
    """
    results = {}

    for address in tokens:
        url = f"https://public-api.birdeye.so/defi/token_overview?address={address}"

        headers = {
            "accept": "application/json",
            "X-API-KEY": API_Key
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = json.loads(response.text)
            if 'data' in data:
                token_data = data['data']
                selected_data = {
                    'address': token_data.get('address', 'N/A'),
                    'decimals': token_data.get('decimals', 'N/A'),
                    'symbol': token_data.get('symbol', 'N/A'),
                    'name': token_data.get('name', 'N/A'),
                    'price': token_data.get('price', 'N/A'),
                    'liquidity': token_data.get('liquidity', 'N/A'),
                    'uniqueWallet30m': token_data.get('uniqueWallet30m', 'N/A'),
                    'uniqueWallet30mChangePercent': token_data.get('uniqueWallet30mChangePercent', 'N/A'),
                    'uniqueWallet1h': token_data.get('uniqueWallet1h', 'N/A'),
                    'uniqueWallet1hChangePercent': token_data.get('uniqueWallet1hChangePercent', 'N/A'),
                    'uniqueWallet2h': token_data.get('uniqueWallet2h', 'N/A'),
                    'uniqueWallet2hChangePercent': token_data.get('uniqueWallet2hChangePercent', 'N/A'),
                    'uniqueWallet4h': token_data.get('uniqueWallet4h', 'N/A'),
                    'uniqueWallet4hChangePercent': token_data.get('uniqueWallet4hChangePercent', 'N/A'),
                    'uniqueWallet6h': token_data.get('uniqueWallet6h', 'N/A'),
                    'uniqueWallet6hChangePercent': token_data.get('uniqueWallet6hChangePercent', 'N/A'),
                    'uniqueWallet8h': token_data.get('uniqueWallet8h', 'N/A'),
                    'uniqueWallet8hChangePercent': token_data.get('uniqueWallet8hChangePercent', 'N/A'),
                    'uniqueWallet12h': token_data.get('uniqueWallet12h', 'N/A'),
                    'uniqueWallet12hChangePercent': token_data.get('uniqueWallet12hChangePercent', 'N/A'),
                    'uniqueWallet24h': token_data.get('uniqueWallet24h', 'N/A'),
                    'uniqueWallet24hChangePercent': token_data.get('uniqueWallet24hChangePercent', 'N/A'),
                    'realMc': token_data.get('realMc', 'N/A'),
                    'holder': token_data.get('holder', 'N/A'),
                    'numberMarkets': token_data.get('numberMarkets', 'N/A'),
                    'coingeckoId': token_data.get('extensions', {}).get('coingeckoId', 'N/A'),
                    'telegram': token_data.get('extensions', {}).get('telegram', 'N/A'),
                    'twitter': token_data.get('extensions', {}).get('twitter', 'N/A'),
                    'website': token_data.get('extensions', {}).get('website', 'N/A')
                }

                # Create a DataFrame from the selected data
                df = pd.DataFrame.from_dict(selected_data, orient='index', columns=['Value'])
                df.index.name = 'Attribute'
                df.reset_index(inplace=True)

                results[address] = df
            else:
                results[address] = pd.DataFrame({'Error': ["Unexpected response structure. 'data' key not found."]})
        else:
            results[address] = pd.DataFrame({'Error': [f"Request failed with status code: {response.status_code}"]})

        # Respect rate limits by adding a small delay between requests
        time.sleep(0.08)

    return results

# Token Trade Data (Single)

def get_token_trade_data_multi(tokens, API_Key):
    """
    Fetch token trade data for multiple tokens.

    Args:
    tokens (list): List of token addresses to fetch data for.
    API_Key (str): Your Birdeye API key.

    Returns:
    dict: A dictionary with token addresses as keys and their trade data as pandas DataFrames.
    """
    results = {}

    for address in tokens:
        url = f"https://public-api.birdeye.so/defi/v3/token/trade-data/single?address={address}"

        headers = {
            "accept": "application/json",
            "X-API-KEY": API_Key
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = json.loads(response.text)
            if 'data' in data:
                token_data = data['data']
                
                # Convert the token data to a DataFrame
                df = pd.DataFrame([token_data])
                
                # Transpose the DataFrame for better readability
                df_transposed = df.T.reset_index()
                df_transposed.columns = ['Attribute', 'Value']
                
                results[address] = df_transposed
            else:
                results[address] = pd.DataFrame({'Attribute': ['Error'], 'Value': ["Unexpected response structure. 'data' key not found."]})
        else:
            results[address] = pd.DataFrame({'Attribute': ['Error'], 'Value': [f"Request failed with status code: {response.status_code}"]})

        # Respect rate limits by adding a small delay between requests
        time.sleep(0.08)

    return results


# Trending List

def get_trending_tokens(total_tokens, API_Key, chain):
    """
    Fetch trending tokens data.

    Args:
    total_tokens (int): Number of trending tokens to fetch.
    API_Key (str): Your Birdeye API key.
    chain (str): The blockchain to fetch data for (e.g., 'solana').

    Returns:
    pd.DataFrame: A DataFrame containing the trending tokens data.
    """
    sort_by = "volume24hUSD"  # rank, volume24hUSD, liquidity
    sort_type = "desc"  # asc, desc
    limit = 20  # API limit per request
    all_tokens = []

    for offset in range(0, total_tokens, limit):
        url = f"https://public-api.birdeye.so/defi/token_trending?sort_by={sort_by}&sort_type={sort_type}&offset={offset}&limit={limit}"

        headers = {
            "accept": "application/json",
            "x-chain": chain,
            "X-API-KEY": API_Key
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = json.loads(response.text)
            if 'data' in data and 'tokens' in data['data']:
                all_tokens.extend(data['data']['tokens'])
            else:
                print(f"Unexpected response structure at offset {offset}")
                print("Response:", response.text)  # Print the response for debugging
        else:
            print(f"Request failed with status code: {response.status_code} at offset {offset}")
            print("Response:", response.text)  # Print the response for debugging
            break

        if len(all_tokens) >= total_tokens:
            break

        # Respect rate limits
        time.sleep(0.08)

    all_tokens = all_tokens[:total_tokens]

    if all_tokens:
        df = pd.DataFrame(all_tokens)

        # Select and rename columns
        columns_to_display = {
            'symbol': 'Symbol',
            'name': 'Name',
            'price': 'Price',
            'volume24hUSD': 'Volume 24h',
            'liquidity': 'Liquidity',
            'rank': 'Rank'
        }
        df = df[columns_to_display.keys()].rename(columns=columns_to_display)

        # Ensure we have exactly the requested number of rows
        df = df.head(total_tokens)

        return df
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no data is available


# New Listings 0-3 days max

def get_new_listings(days_back, hours_back, minutes_back, API_Key, liquidity_filter):
    """
    Fetch new token listings data.

    Args:
    days_back (int): Number of days to look back (1, 2, or 3).
    hours_back (int): Number of hours to look back within the selected day.
    minutes_back (int): Number of minutes to look back within the selected hour.
    API_Key (str): Your Birdeye API key.
    liquidity_filter (float): Minimum liquidity for filtered results.

    Returns:
    tuple: A tuple containing two DataFrames (full_df, filtered_df) with new listings data.
    """
    def iso8601_to_timestamp(iso_string):
        return int(datetime.fromisoformat(iso_string.replace('Z', '+00:00')).timestamp())

    end_time = datetime.now(pytz.UTC)
    
    if days_back == 1:
        start_time = end_time - timedelta(hours=hours_back, minutes=minutes_back)
    elif days_back == 2:
        start_time = end_time - timedelta(days=1, hours=hours_back, minutes=minutes_back)
    elif days_back == 3:
        start_time = end_time - timedelta(days=2, hours=hours_back, minutes=minutes_back)
    else:
        raise ValueError("days_back must be 1, 2, or 3")
    
    all_tokens = []
    current_time = int(end_time.timestamp())
    
    while current_time > int(start_time.timestamp()):
        url = f"https://public-api.birdeye.so/defi/v2/tokens/new_listing?time_to={current_time}&limit=10"

        headers = {
            "accept": "application/json",
            "X-API-KEY": API_Key
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if 'data' in data and 'items' in data['data']:
                new_tokens = data['data']['items']
                if not new_tokens:
                    break
                all_tokens.extend(new_tokens)
                current_time = min(iso8601_to_timestamp(token['liquidityAddedAt']) for token in new_tokens)
            else:
                print(f"Unexpected response structure")
                print("Response:", response.text)
                break
        else:
            print(f"Request failed with status code: {response.status_code}")
            print("Response:", response.text)
            break

        # Respect rate limits
        time.sleep(0.08)

    # Remove duplicates based on 'address'
    unique_tokens = list({token['address']: token for token in all_tokens}.values())
    
    if unique_tokens:
        # Create full DataFrame
        df_full = pd.DataFrame(unique_tokens)
        
        # Convert liquidityAddedAt to datetime
        df_full['liquidityAddedAt'] = pd.to_datetime(df_full['liquidityAddedAt'])
        
        # Sort by liquidityAddedAt in descending order
        df_full = df_full.sort_values('liquidityAddedAt', ascending=False)
        
        # Apply filters for the second table
        df_filtered = df_full[
            (df_full['name'].notna() & (df_full['name'] != 'None')) & 
            (df_full['logoURI'].notna() & (df_full['logoURI'] != 'None')) & 
            (df_full['liquidity'].astype(float) >= liquidity_filter)
        ]
        
        return df_full, df_filtered
    else:
        return pd.DataFrame(), pd.DataFrame()

# Top Traders

def get_top_traders(address, time_frame, sort_type, sort_by, total_traders, API_Key):
    """
    Fetch top traders data for a specific token.

    Args:
    address (str): The token address to fetch top traders for.
    time_frame (str): Time frame for the data (30m, 1h, 4h, 6h, 8h, 12h, 24h).
    sort_type (str): Sort order (asc, desc).
    sort_by (str): Sort criterion (trade, volume).
    total_traders (int): Number of top traders to retrieve.
    API_Key (str): Your Birdeye API key.

    Returns:
    pd.DataFrame: A DataFrame containing the top traders data.
    """
    limit = 10
    all_traders = []

    for offset in range(0, total_traders, limit):
        url = f"https://public-api.birdeye.so/defi/v2/tokens/top_traders?address={address}&time_frame={time_frame}&sort_type={sort_type}&sort_by={sort_by}&offset={offset}&limit={limit}"

        headers = {
            "accept": "application/json",
            "X-API-KEY": API_Key
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = json.loads(response.text)
            if 'data' in data and 'items' in data['data']:
                all_traders.extend(data['data']['items'])
            else:
                print(f"Unexpected response structure at offset {offset}")
                break
        else:
            print(f"Request failed with status code: {response.status_code} at offset {offset}")
            break

        if len(all_traders) >= total_traders:
            break

        # Respect rate limits
        time.sleep(0.08)

    return pd.DataFrame(all_traders[:total_traders])

# Top Markets for a Token

def get_markets(address, time_frame, sort_type, sort_by, total_markets, API_Key):
    """
    Fetch markets data for a specific token.

    Args:
    address (str): The token address to fetch markets for.
    time_frame (str): Time frame for the data (30m, 1h, 4h, 6h, 8h, 12h, 24h).
    sort_type (str): Sort order (asc, desc).
    sort_by (str): Sort criterion (volume24h, liquidity).
    total_markets (int): Number of markets to retrieve.
    API_Key (str): Your Birdeye API key.

    Returns:
    pd.DataFrame: A DataFrame containing the markets data.
    """
    limit = 10
    all_markets = []

    for offset in range(0, total_markets, limit):
        url = f"https://public-api.birdeye.so/defi/v2/markets?address={address}&time_frame={time_frame}&sort_type={sort_type}&sort_by={sort_by}&offset={offset}&limit={limit}"

        headers = {
            "accept": "application/json",
            "X-API-KEY": API_Key
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = json.loads(response.text)
            if 'data' in data and 'items' in data['data']:
                all_markets.extend(data['data']['items'])
            else:
                print(f"Unexpected response structure at offset {offset}")
                break
        else:
            print(f"Request failed with status code: {response.status_code} at offset {offset}")
            break

        if len(all_markets) >= total_markets:
            break

        # Respect rate limits
        time.sleep(0.08)

    if all_markets:
        df = pd.DataFrame(all_markets[:total_markets])

        # Flatten nested dictionaries
        df['base_address'] = df['base'].apply(lambda x: x.get('address'))
        df['base_symbol'] = df['base'].apply(lambda x: x.get('symbol'))
        df['quote_address'] = df['quote'].apply(lambda x: x.get('address'))
        df['quote_symbol'] = df['quote'].apply(lambda x: x.get('symbol'))

        # Select and rename columns based on available data
        columns_to_display = {
            'address': 'Market Address',
            'name': 'Name',
            'base_symbol': 'Base Symbol',
            'quote_symbol': 'Quote Symbol',
            'liquidity': 'Liquidity',
            'volume24h': 'Volume 24h',
            'price': 'Price',
            'trade24h': 'Trades 24h',
            'uniqueWallet24h': 'Unique Wallets 24h',
            'source': 'Source'
        }
        
        # Only include columns that exist in the DataFrame
        available_columns = [col for col in columns_to_display.keys() if col in df.columns]
        df = df[available_columns].rename(columns={col: columns_to_display[col] for col in available_columns})

        return df
    else:
        return pd.DataFrame()

### Wallet APIs

# Supported Chains

url = "https://public-api.birdeye.so/v1/wallet/list_supported_chain"

headers = {
    "accept": "application/json",
    "X-API-KEY": API_Key
}

response = requests.get(url, headers=headers)

print(response.text)

# Wallet Portfolio Multichain

def get_multichain_wallet_tokens(wallet, multichain, API_Key):
    """
    Fetch multichain wallet token list.

    Args:
    wallet (str): The wallet address to fetch tokens for.
    multichain (str): Comma-separated list of chains to query.
    API_Key (str): Your Birdeye API key.
        """
    url = f"https://public-api.birdeye.so/v1/wallet/token_list?wallet={wallet}"

    headers = {
        "accept": "application/json",
        "x-chain": multichain,
        "X-API-KEY": API_Key
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = json.loads(response.text)
        if 'data' in data and 'items' in data['data']:
            tokens = data['data']['items']
            df = pd.DataFrame(tokens)
            
            # Flatten nested dictionaries
            df['token_address'] = df['token'].apply(lambda x: x.get('address'))
            df['token_symbol'] = df['token'].apply(lambda x: x.get('symbol'))
            df['token_name'] = df['token'].apply(lambda x: x.get('name'))
            df['token_decimals'] = df['token'].apply(lambda x: x.get('decimals'))
            
            # Select and rename columns
            columns_to_display = {
                'chain': 'Chain',
                'token_address': 'Token Address',
                'token_symbol': 'Symbol',
                'token_name': 'Name',
                'token_decimals': 'Decimals',
                'price': 'Price',
                'value': 'Value',
                'amount': 'Amount'
            }
            df = df[columns_to_display.keys()].rename(columns=columns_to_display)
            
            return df
        else:
            print("Unexpected response structure")
            return pd.DataFrame()
    else:
        print(f"Request failed with status code: {response.status_code}")
        return pd.DataFrame()

# Wallet Portfolio History

def get_wallet_portfolio_history(wallet, time_frame, API_Key):
    """
    Fetch wallet portfolio history.

    Args:
    wallet (str): The wallet address to fetch history for.
    time_frame (str): Time frame for the data (1D, 7D, 30D, 90D, 180D, 1Y, ALL).
    API_Key (str): Your Birdeye API key.

    Returns:
    pd.DataFrame: A DataFrame containing the wallet's portfolio history.
    """
    url = f"https://public-api.birdeye.so/v1/wallet/portfolio_history?wallet={wallet}&time_frame={time_frame}"

    headers = {
        "accept": "application/json",
        "X-API-KEY": API_Key
    }

    response = requests.get(url, headers=headers)

    print(f"Response status code: {response.status_code}")
    print(f"Response content: {response.text}")

    if response.status_code == 200:
        try:
            data = json.loads(response.text)
            if 'data' in data and 'items' in data['data']:
                history = data['data']['items']
                df = pd.DataFrame(history)
                
                # Convert timestamp to datetime
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                
                # Select and rename columns
                columns_to_display = {
                    'datetime': 'Datetime',
                    'timestamp': 'Timestamp',
                    'value': 'Value'
                }
                df = df[columns_to_display.keys()].rename(columns=columns_to_display)
                
                return df
            else:
                print("Unexpected response structure")
                return pd.DataFrame()
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return pd.DataFrame()
    else:
        print(f"Request failed with status code: {response.status_code}")
        return pd.DataFrame()

# Wallet Transaction History

def get_wallet_transaction_history(wallet, offset, limit, API_Key):
    """
    Fetch wallet transaction history.

    Args:
    wallet (str): The wallet address to fetch transaction history for.
    offset (int): The offset for pagination.
    limit (int): The number of transactions to retrieve per request.
    API_Key (str): Your Birdeye API key.

    Returns:
    pd.DataFrame: A DataFrame containing the wallet's transaction history.
    """
    url = f"https://public-api.birdeye.so/v1/wallet/transaction_history?wallet={wallet}&offset={offset}&limit={limit}"

    headers = {
        "accept": "application/json",
        "X-API-KEY": API_Key
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = json.loads(response.text)
        if 'data' in data and 'items' in data['data']:
            transactions = data['data']['items']
            df = pd.DataFrame(transactions)
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['blockTime'], unit='s')
            
            # Select and rename columns
            columns_to_display = {
                'datetime': 'Datetime',
                'blockTime': 'Block Time',
                'signature': 'Signature',
                'type': 'Type',
                'status': 'Status',
                'tokenAddress': 'Token Address',
                'tokenSymbol': 'Token Symbol',
                'tokenName': 'Token Name',
                'amount': 'Amount',
                'amountUSD': 'Amount USD',
                'fee': 'Fee'
            }
            df = df[columns_to_display.keys()].rename(columns=columns_to_display)
            
            return df
        else:
            print("Unexpected response structure")
            return pd.DataFrame()
    else:
        print(f"Request failed with status code: {response.status_code}")
        return pd.DataFrame()
