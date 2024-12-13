import requests
import logging

def get_jupiter_price(input_token: str, output_token: str) -> float:
    """
    Fetches the price of a token from Jupiter.

    Parameters:
    - input_token: The mint address of the input token (e.g., the token you have).
    - output_token: The mint address of the output token (e.g., the token you want).

    Returns:
    - price: The price of the output token in terms of the input token.
    """
    url = f'https://quote-api.jup.ag/v4/quote?inputMint={input_token}&outputMint={output_token}&amount=1000000'
    try:
        response = requests.get(url)
        if response.status_code == 404:
            logging.warning(f"Token pair not found on Jupiter: {input_token}/{output_token}")
            return 0.0
        response.raise_for_status()
        data = response.json()
        if data and 'data' in data and data['data']:
            # The price is calculated as output amount divided by input amount
            best_quote = data['data'][0]
            input_amount = float(best_quote['inputAmount'])
            output_amount = float(best_quote['outAmount'])
            price = output_amount / input_amount
            return price
    except Exception as e:
        logging.error(f"Error fetching price from Jupiter: {e}")
        return 0.0
    return 0.0 