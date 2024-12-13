import requests
import sys
import json
import base64
import time
import logging
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction 
from solana.rpc.api import Client
from solana.rpc.types import TxOpts
import dontshare as d 
from pprint import pprint
from functools import lru_cache
from datetime import datetime, timedelta
import pytz
import json
import pandas as pd
from pprint import pprint 
from IPython.display import display, HTML
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import dontshare as d

API_Key = d.birdeye
wallet = d.sol_wallet
chain = "solana"

def format_number(num):
    # ... (no changes in this function) ...
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}Bil"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}Mil"
    elif num >= 1_000:
        return f"{num / 1_000:.0f}K"
    else:
        return str(num)

def get_token_overview_data_multi(tokens, API_Key):
    # ... (no changes in this function) ...
    results = {}
    logging.info(f"Fetching token overview data for {len(tokens)} tokens...")

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
                
                # Safely get extensions, handling cases where it might be None
                extensions = token_data.get('extensions')
                if extensions is None:
                    extensions = {}  # Treat it as an empty dictionary if it's None

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
                    'coingeckoId': extensions.get('coingeckoId', 'N/A'),
                    'telegram': extensions.get('telegram', 'N/A'),
                    'twitter': extensions.get('twitter', 'N/A'),
                    'website': extensions.get('website', 'N/A'),
                    'discord': extensions.get('discord', 'N/A')
                }

                # Convert the selected data into a single-row DataFrame
                df = pd.DataFrame([selected_data])

                results[address] = df
                logging.info(f"Successfully fetched overview data for token: {address}")
            else:
                results[address] = pd.DataFrame({'Error': ["Unexpected response structure. 'data' key not found."]})
                logging.warning(f"Unexpected response structure for token: {address}. 'data' key not found.")
        else:
            results[address] = pd.DataFrame({'Error': [f"Request failed with status code: {response.status_code}"]})
            logging.error(f"Request failed for token: {address} with status code: {response.status_code}")

        # Respect rate limits by adding a small delay between requests
        time.sleep(0.07)

    logging.info("Finished fetching token overview data.")
    return results

def get_token_list(sort_by, sort_type, min_liquidity, min_volume_24h, min_market_cap, max_market_cap, total_tokens, chain, API_Key):
    # ... (no changes in this function) ...
    limit = 50
    all_tokens = []
    logging.info(f"Starting to fetch token list with parameters: sort_by={sort_by}, sort_type={sort_type}, min_liquidity={min_liquidity}, min_volume_24h={min_volume_24h}, min_market_cap={min_market_cap}, max_market_cap={max_market_cap}, total_tokens={total_tokens}, chain={chain}")

    for offset in range(0, total_tokens * 3, limit):  # Fetch more tokens to account for filtering
        logging.info(f"Fetching tokens with offset: {offset}")
        url = f"https://public-api.birdeye.so/defi/tokenlist?sort_by={sort_by}&sort_type={sort_type}&offset={offset}&limit={limit}&min_liquidity={min_liquidity}"

        headers = {
            "accept": "application/json",
            "x-chain": chain,
            "X-API-KEY": API_Key
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = json.loads(response.text)
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
                    if token.get('v24hUSD', 0) >= min_volume_24h and \
                       token.get('mc') is not None and \
                       min_market_cap <= token.get('mc') <= max_market_cap
                ]
                logging.info(f"Fetched {len(new_tokens)} tokens at offset {offset} that meet the criteria.")
                all_tokens.extend(new_tokens)
            else:
                logging.warning(f"Unexpected response structure at offset {offset}")
                break
        else:
            logging.error(f"Request failed with status code: {response.status_code} at offset {offset}")
            break

        if len(all_tokens) >= total_tokens:
            logging.info(f"Reached total_tokens limit ({total_tokens}). Stopping.")
            break

        # Respect rate limits
        time.sleep(0.07)

    all_tokens = all_tokens[:total_tokens]
    logging.info(f"Total tokens fetched and filtered: {len(all_tokens)}")

    if all_tokens:
        df = pd.DataFrame(all_tokens)

        # Select and rename columns
        columns_to_display = {
            'name': 'Name',
            'symbol': 'Symbol',
            'mc': 'Market Cap',
            'v24hUSD': 'Volume 24h',
            'v24hChangePercent': '24h Change (%)',
            'liquidity': 'Liquidity',
            'address': 'Address'
        }
        
        # Only include columns that exist in the DataFrame
        available_columns = [col for col in ['name', 'symbol', 'mc', 'v24hUSD', 'v24hChangePercent', 'liquidity', 'address'] if col in df.columns]
        df = df[available_columns].rename(columns={col: columns_to_display[col] for col in available_columns})

        # Format 'Market Cap' and 'Volume 24h' columns
        if 'Market Cap' in df.columns:
            df['Market Cap'] = df['Market Cap'].apply(format_number)
        if 'Volume 24h' in df.columns:
            df['Volume 24h'] = df['Volume 24h'].apply(format_number)

        # Get token overview data
        token_addresses = df['Address'].tolist()
        logging.info("Preparing to fetch overview data for filtered tokens...")
        overview_data = get_token_overview_data_multi(token_addresses, API_Key)

        # Merge overview data with the main DataFrame
        # Convert overview_data to a DataFrame
        overview_df = pd.concat(overview_data.values(), ignore_index=True)

        # Merge with the main DataFrame
        df_final = pd.merge(df, overview_df, left_on='Address', right_on='address', how='left')

        # Drop the redundant 'address' column
        df_final.drop(columns='address', inplace=True)
        
        # Format the 'realMc' column using format_number
        if 'realMc' in df_final.columns:
            df_final['realMc'] = df_final['realMc'].apply(lambda x: format_number(x) if pd.notnull(x) else x)

        logging.info("Token list fetching and processing complete.")
        return df_final

    else:
        logging.info("No tokens met the specified criteria.")
        return pd.DataFrame()

if __name__ == "__main__":
    # ... (no changes in the main block) ...
    sort_by = "mc"
    sort_type = "asc"
    min_liquidity = 100000
    min_volume_24h = 1000000
    min_market_cap = 500000
    max_market_cap = 1000000
    total_tokens = 750
    chain = "solana"

    logging.info("Starting script execution...")
    df = get_token_list(sort_by, sort_type, min_liquidity, min_volume_24h, min_market_cap, max_market_cap, total_tokens, chain, API_Key)

    # Save to CSV
    df.to_csv("token_data.csv", index=False)
    logging.info(f"Token data saved to token_data.csv")
    logging.info("Script execution completed.")