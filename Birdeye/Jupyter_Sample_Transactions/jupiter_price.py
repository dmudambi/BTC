import requests
import time
import logging
from datetime import datetime

def setup_logger():
    """
    Sets up a logger to log messages to a file and the console.

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger("price_monitor")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler("price_monitor.log")
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
            logger.error(f"Error during API request: {e}")
        return None

def monitor_price(token_to_query, logger):
    """
    Monitors the price of a token every second and logs the data in a human-readable format.

    Args:
        token_to_query (str): The symbol or address of the token to monitor.
        logger (logging.Logger): The logger to use.
    """
    while True:
        price_data = get_jupiter_price_v2(token_to_query, logger=logger)
        if price_data and price_data['data']:
            for id, data in price_data['data'].items():
                if data['type'] == 'derivedPrice' and 'extraInfo' in data:
                    extra_info = data['extraInfo']
                    
                    # Extract and format data
                    buy_price = extra_info.get('quotedPrice', {}).get('buyPrice')
                    sell_price = extra_info.get('quotedPrice', {}).get('sellPrice')
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
                    logger.info(f"Token: {id}")
                    if buy_price:
                        logger.info(f"  Buy Price: {buy_price} (at: {buy_at_dt})")
                    if sell_price:
                        logger.info(f"  Sell Price: {sell_price} (at: {sell_at_dt})")
                    logger.info(f"  Buy Price Impact Ratio (SOL to %): {buy_impact_pct}")
                    logger.info(f"  Sell Price Impact Ratio (SOL to %): {sell_impact_pct}")
                else:
                    logger.warning(f"Unexpected data format for token {id}")
        else:
            logger.warning("Failed to retrieve price data or empty data set.")

        time.sleep(1)

def main():
    token_to_query = "2qEHjDLDLbuBgRYvsxhc5D6uDWAivNFZGan56P1tpump"
    logger = setup_logger()
    monitor_price(token_to_query, logger)

if __name__ == "__main__":
    main()