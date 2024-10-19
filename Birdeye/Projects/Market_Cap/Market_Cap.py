import requests
import json
import time
import logging
import dontshare as d 
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_token_list(sort_by, sort_type, min_liquidity, min_volume_24h, min_market_cap, max_market_cap, total_tokens, chain, API_Key):
    """
    Fetch and filter a list of tokens based on specified criteria.

    Args:
        sort_by (str): Criterion to sort by (e.g., 'mc', 'v24hUSD', 'v24hChangePercent').
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
        list: A list of error messages encountered during fetching.
    """
    limit = 50
    all_tokens = []
    errors = []

    headers = {
        "accept": "application/json",
        "x-chain": chain,
        "X-API-KEY": API_Key
    }

    for offset in range(0, total_tokens * 3, limit):  # Fetch more tokens to account for filtering
        url = f"https://public-api.birdeye.so/defi/tokenlist?sort_by={sort_by}&sort_type={sort_type}&offset={offset}&limit={limit}&min_liquidity={min_liquidity}"

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
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
                    logging.info(f"Fetched {len(new_tokens)} tokens at offset {offset}. Total tokens collected: {len(all_tokens)}")
                else:
                    error_msg = f"Unexpected response structure at offset {offset}"
                    logging.warning(error_msg)
                    errors.append(error_msg)
                    break
            elif response.status_code == 429:
                # Rate limit exceeded, wait and retry
                retry_after = int(response.headers.get("Retry-After", 60))
                error_msg = f"Rate limit exceeded. Waiting for {retry_after} seconds."
                logging.warning(error_msg)
                errors.append(error_msg)
                time.sleep(retry_after)
                continue
            elif response.status_code == 400:
                # Bad Request - log the response content for debugging
                try:
                    error_content = response.json()
                except json.JSONDecodeError:
                    error_content = response.text
                error_msg = f"Bad Request (400) at offset {offset}: {error_content}"
                logging.error(error_msg)
                errors.append(error_msg)
                break  # Exit the loop on error
            else:
                error_msg = f"Request failed with status code: {response.status_code} at offset {offset}"
                logging.error(error_msg)
                errors.append(error_msg)
                break  # Exit the loop on error
        except Exception as e:
            error_msg = f"An exception occurred: {e}"
            logging.error(error_msg)
            errors.append(error_msg)
            break  # Exit on exception

        if len(all_tokens) >= total_tokens:
            break

        # Respect rate limits
        time.sleep(1)

    all_tokens = all_tokens[:total_tokens]

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

        return df, errors
    else:
        return pd.DataFrame(), errors