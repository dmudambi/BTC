import requests
import time
import logging
from datetime import datetime
import asyncio

# --- Configurable Parameters ---
TOKEN_TO_QUERY = "HMdmbv35DvbH7VxWSwXaxE1NNyA34HHhs9Cu9ycZpump"  # The token to monitor
LOG_FILE = "price_monitor.log"  # The file to store logs
PRICE_STABILITY_THRESHOLD = 2  # Percentage difference threshold for price stability
REFRESH_INTERVAL = 0.5  # Time in seconds between each data refresh
# ---------------------------------

def setup_logger():
    """
    Sets up a logger to log messages to a file and the console.

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger("price_monitor")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def get_jupiter_price_v2(token_id, show_extra_info=True, logger=None):
    """
    Fetches the price of a token using the Jupiter V2 Price API.

    Args:
        token_id (str): The symbol or address of the token.
        show_extra_info (bool, optional): Whether to include additional information. Defaults to True.
        logger (logging.Logger, optional): The logger to use. Defaults to None.

    Returns:
        dict: The JSON response from the API.
        None: If there was an error during the API request.
    """
    base_url = "https://api.jup.ag/price/v2"
    params = {"ids": token_id, "showExtraInfo": "true" if show_extra_info else "false"}

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if logger:
            logger.error(f"Error during Jupiter API request: {e}")
        return None

def get_token_pairs(token_address):
    """Fetch all pairs for a specific token from DexScreener"""
    url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
    
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json"
            }
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error fetching token pairs from DexScreener: {str(e)}")
        return None

def get_token_name(token_address):
    """Fetch the name of the token from DexScreener."""
    data = get_token_pairs(token_address)
    if not data or 'pairs' not in data or not data['pairs']:
        raise ValueError(f"No pairs found for token {token_address}")

    # Assuming the base token is the one we're interested in
    base_token = data['pairs'][0].get('baseToken', {})
    token_name = base_token.get('name', 'Unknown Token')
    return token_name

async def get_token_price(token_address: str) -> float:
    """Get the current price of a token from its most liquid pair asynchronously."""
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, get_token_pairs, token_address)
    
    if not data or 'pairs' not in data or not data['pairs']:
        raise ValueError(f"No pairs found for token {token_address}")
    
    # Sort pairs by liquidity to find the most liquid one
    pairs = sorted(data['pairs'], 
                   key=lambda x: float(x.get('liquidity', {}).get('usd', 0)), 
                   reverse=True)
    
    most_liquid_pair = pairs[0]
    price = float(most_liquid_pair.get('priceUsd', 0))
    
    if price <= 0:
        raise ValueError(f"Invalid price received for token {token_address}")
    
    return price

def format_price_change(change):
    """Format price change with color and arrow"""
    if change > 0:
        return f"��� +{change}%"
    elif change < 0:
        return f"↓ {change}%"
    return f"→ {change}%"

def monitor_price(token_to_query, logger):
    """
    Monitors the price of a token using both Jupiter and DexScreener APIs,
    and logs the data in a human-readable format.
    Also assesses price stability and logs a corresponding indicator.
    """
    while True:
        # Fetch data from Jupiter
        jupiter_price_data = get_jupiter_price_v2(token_to_query, logger=logger)
        
        # Fetch data from DexScreener
        dexscreener_data = get_token_pairs(token_to_query)

        logger.info(f"Monitoring Token: {token_to_query}")

        # Process Jupiter data
        jupiter_price = None
        if jupiter_price_data and jupiter_price_data['data']:
            for id, data in jupiter_price_data['data'].items():
                if data['type'] == 'derivedPrice' and 'extraInfo' in data:
                    extra_info = data['extraInfo']
                    buy_price = extra_info.get('quotedPrice', {}).get('buyPrice')
                    sell_price = extra_info.get('quotedPrice', {}).get('sellPrice')

                    # Use average of buy and sell price as a proxy for Jupiter price
                    if buy_price and sell_price:
                        jupiter_price = (float(buy_price) + float(sell_price)) / 2
                    
                    # Extract and format data
                    buy_at = extra_info.get('quotedPrice', {}).get('buyAt')
                    sell_at = extra_info.get('quotedPrice', {}).get('sellAt')
                    
                    buy_impact_ratio = extra_info.get('depth', {}).get('buyPriceImpactRatio', {}).get('depth', {})
                    sell_impact_ratio = extra_info.get('depth', {}).get('sellPriceImpactRatio', {}).get('depth', {})
                    
                    # Convert ratios to percentages, handling None values
                    buy_impact_pct = {k: f"{float(v) * 100:.2f}%" if v is not None else "N/A" for k, v in buy_impact_ratio.items()}
                    sell_impact_pct = {k: f"{float(v) * 100:.2f}%" if v is not None else "N/A" for k, v in sell_impact_ratio.items()}
                    
                    # Convert epoch time to human-readable format
                    buy_at_dt = datetime.fromtimestamp(buy_at) if buy_at else "N/A"
                    sell_at_dt = datetime.fromtimestamp(sell_at) if sell_at else "N/A"
                    
                    # Log the formatted data
                    logger.info(f"Jupiter Data:")
                    if buy_price:
                        logger.info(f"  Buy Price: {buy_price} (at: {buy_at_dt})")
                    if sell_price:
                        logger.info(f"  Sell Price: {sell_price} (at: {sell_at_dt})")
                    logger.info(f"  Buy Price Impact Ratio (SOL to %): {buy_impact_pct}")
                    logger.info(f"  Sell Price Impact Ratio (SOL to %): {sell_impact_pct}")
                else:
                    logger.warning(f"Unexpected Jupiter data format for token {id}")
        else:
            logger.warning("Failed to retrieve Jupiter price data or empty data set.")

        # Process DexScreener data
        dexscreener_price = None
        if dexscreener_data and 'pairs' in dexscreener_data:
            pairs = sorted(dexscreener_data['pairs'], 
                           key=lambda x: float(x.get('liquidity', {}).get('usd', 0)), 
                           reverse=True)
            most_liquid_pair = pairs[0]
            dexscreener_price = float(most_liquid_pair.get('priceUsd', 0))
            
            # Token info
            base_token = most_liquid_pair.get('baseToken', {})
            quote_token = most_liquid_pair.get('quoteToken', {})
            logger.info(f"DexScreener Data (Most Liquid Pair):")
            logger.info(f"  Pair: {most_liquid_pair.get('dexId')}/{most_liquid_pair.get('symbol')}")
            logger.info(f"  Price: ${most_liquid_pair.get('priceUsd', 'N/A')}")
            logger.info(f"  Base Token: {base_token.get('symbol')} ({base_token.get('address')})")
            logger.info(f"  Quote Token: {quote_token.get('symbol')} ({quote_token.get('address')})")
            
            # Price changes
            changes = most_liquid_pair.get('priceChange', {})
            logger.info(f"  Price Changes:")
            logger.info(f"    24h: {format_price_change(changes.get('h24', 0))}")
            logger.info(f"    6h: {format_price_change(changes.get('h6', 0))}")
            logger.info(f"    1h: {format_price_change(changes.get('h1', 0))}")
            
            # Volume
            volume = most_liquid_pair.get('volume', {})
            logger.info(f"  Volume:")
            logger.info(f"    24h: ${volume.get('h24', 'N/A'):,.2f}")
            logger.info(f"    6h: ${volume.get('h6', 'N/A'):,.2f}")
            logger.info(f"    1h: ${volume.get('h1', 'N/A'):,.2f}")
            
            # Liquidity
            liquidity = most_liquid_pair.get('liquidity', {})
            logger.info(f"  Liquidity: ${liquidity.get('usd', 'N/A'):,.2f}")
            
            # Transactions
            txns = most_liquid_pair.get('txns', {})
            logger.info(f"  Transactions:")
            for period in ['h1', 'h6', 'h24']:
                if period in txns:
                    buys = txns[period].get('buys', 0)
                    sells = txns[period].get('sells', 0)
                    logger.info(f"    {period}: Buys - {buys}, Sells - {sells}")
        else:
            logger.warning("Failed to retrieve DexScreener data or empty data set.")

        # Assess price stability
        price_stability_indicator = assess_price_stability(jupiter_price, dexscreener_price, logger)
        logger.info(f"Price Stability: {price_stability_indicator}")

        time.sleep(REFRESH_INTERVAL)

def assess_price_stability(jupiter_price, dexscreener_price, logger):
    """
    Assesses the price stability between Jupiter and DexScreener prices.

    Args:
        jupiter_price (float): The price from Jupiter.
        dexscreener_price (float): The price from DexScreener.
        logger (logging.Logger): The logger to use.

    Returns:
        str: A string indicating price stability with a corresponding emoji.
    """
    if jupiter_price is None or dexscreener_price is None:
        return "Price data incomplete - 🔴"

    price_difference_percentage = abs((jupiter_price - dexscreener_price) / dexscreener_price) * 100

    if price_difference_percentage <= PRICE_STABILITY_THRESHOLD:
        return "Price Stable - 🟢"
    else:
        return f"Price Unstable ({price_difference_percentage:.2f}% difference) - 🔴"

def main():
    logger = setup_logger()
    monitor_price(TOKEN_TO_QUERY, logger)

if __name__ == "__main__":
    main() 