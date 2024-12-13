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
from solders.signature import Signature  # Correct import for Signature

# Configure logging
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Constants
SLIPPAGE = 200  # Increase slippage to 2%
JUP_SETTLE_TIMEOUT = 30  # Timeout for transaction settling in seconds
MAX_RETRIES = 5  # Increased number of retries for better resilience
INITIAL_RETRY_DELAY = 5  # Initial delay between retries in seconds

# Initialize Solana RPC client with QuickNode
RPC_URL = "https://methodical-sleek-smoke.solana-mainnet.quiknode.pro/ce998890a7f93d71ff7c2f1979abf0510bf40d80"
http_client = Client(RPC_URL)

# Load wallet keypair from 'dontshare' module
try:
    KEY = Keypair.from_base58_string(d.key)
    USER_PUBLIC_KEY = str(KEY.pubkey())
except Exception as e:
    print(f"Error loading keypair: {e}")
    logging.error(f"Error loading keypair: {e}")
    sys.exit(1)

# Token configuration
QUOTE_TOKEN = 'So11111111111111111111111111111111111111112'  # SOL
OUTPUT_TOKEN = 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm'  # Replace with your target token address
USDT_Quantity = 1
def get_decimals(token_mint_address):
    """
    Retrieves the number of decimals for a given token mint address on Solana.
    
    Args:
        token_mint_address (str): The token's mint address.
    
    Returns:
        int: The number of decimals.
    """
    url = RPC_URL
    headers = {"Content-Type": "application/json"}

    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getAccountInfo",
        "params": [
            token_mint_address,
            {
                "encoding": "jsonParsed"
            }
        ]
    })

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        response_json = response.json()
        decimals = response_json['result']['value']['data']['parsed']['info']['decimals']
        return decimals
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while fetching decimals: {http_err} - {response.text}")
        logging.error(f"HTTP error while fetching decimals: {http_err} - {response.text}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request exception occurred while fetching decimals: {req_err}")
        logging.error(f"Request exception while fetching decimals: {req_err}")
    except KeyError as key_err:
        print(f"KeyError: {key_err} - Response: {response.text}")
        logging.error(f"KeyError while fetching decimals: {key_err} - {response.text}")
    except Exception as e:
        print(f"An error occurred while fetching decimals: {e}")
        logging.error(f"General error while fetching decimals: {e}")
    return 0

# Fetch token decimals for the OUTPUT_TOKEN
token_decimals = get_decimals(OUTPUT_TOKEN)
if token_decimals == 0:
    print(f"Failed to fetch decimals for token {OUTPUT_TOKEN}. Aborting.")
    logging.error(f"Failed to fetch decimals for token {OUTPUT_TOKEN}. Aborting.")
    sys.exit(1)
print(f"Token Decimals: {token_decimals}")
logging.info(f"Token Decimals: {token_decimals}")

def token_price(address):
    """
    Retrieves the current USD price for a given token mint address using Birdeye's Price API.

    Args:
        address (str): The token's mint address.

    Returns:
        float: The USD price of the token or None if unavailable.
    """
    API_KEY = d.birdeye
    url = f"https://public-api.birdeye.so/defi/price?address={address}"
    headers = {"X-API-KEY": API_KEY}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        price_data = response.json()
        
        if price_data.get('success') and price_data.get('data'):
            return float(price_data['data']['value'])
        else:
            print(f"Failed to fetch price for {address}: {json.dumps(price_data)}")
            logging.error(f"Failed to fetch price for {address}: {json.dumps(price_data)}")
            return None
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while fetching price: {http_err} - {response.text}")
        logging.error(f"HTTP error while fetching price: {http_err} - {response.text}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request exception occurred while fetching price: {req_err}")
        logging.error(f"Request exception while fetching price: {req_err}")
    except ValueError as val_err:
        print(f"Value error: {val_err} - Response: {response.text}")
        logging.error(f"Value error while fetching price: {val_err} - {response.text}")
    except KeyError as key_err:
        print(f"KeyError: {key_err} - Response: {response.text}")
        logging.error(f"KeyError while fetching price: {key_err} - {response.text}")
    return None

@lru_cache(maxsize=32)
def get_cached_token_price(mint_address):
    return token_price(mint_address)

def confirm_transaction(tx_id, timeout=30):
    """
    Confirms the transaction on the Solana blockchain.

    Args:
        tx_id (str): The transaction signature.
        timeout (int): Maximum time to wait for confirmation in seconds.

    Returns:
        bool: True if the transaction is confirmed, False otherwise.
    """
    start_time = time.time()
    signature = Signature.from_string(tx_id)  # Convert the string to a Signature object
    while time.time() - start_time < timeout:
        try:
            response = http_client.get_signature_statuses([signature])
            logging.info(f"Response from get_signature_statuses for {tx_id}: {response}")
            
            # Access the statuses attribute directly
            statuses = response.value  # Assuming 'value' is the correct attribute
            
            if statuses and statuses[0]:
                confirmation = statuses[0].confirmation_status  # Access the attribute directly
                if confirmation in ('confirmed', 'finalized'):
                    print(f"Transaction {tx_id} confirmed with status: {confirmation}")
                    logging.info(f"Transaction {tx_id} confirmed with status: {confirmation}")
                    return True
                elif confirmation == 'processed':
                    # Still processing
                    pass
            else:
                logging.info(f"Transaction {tx_id} status not yet available.")
            time.sleep(2)
        except Exception as e:
            print(f"Error confirming transaction {tx_id}: {e}")
            logging.error(f"Error confirming transaction {tx_id}: {e}")
            time.sleep(2)
    print(f"Transaction {tx_id} not confirmed within {timeout} seconds.")
    logging.error(f"Transaction {tx_id} not confirmed within {timeout} seconds.")
    return False

def create_swap_transaction(quote):
    """
    Creates a swap transaction based on the quote provided by Jupiter.

    Args:
        quote (dict): The quote response from Jupiter.

    Returns:
        dict: The swap transaction response from Jupiter.
    """
    if not quote:
        print("Empty quote received, cannot create swap transaction.")
        logging.error("Empty quote received, cannot create swap transaction.")
        return {}
    url = 'https://quote-api.jup.ag/v6/swap'
    headers = {"Content-Type": "application/json"}
    payload = {
        "quoteResponse": quote,
        "userPublicKey": USER_PUBLIC_KEY,
        "dynamicSlippage": {"maxBps": 500}  # Enable dynamic slippage with a maximum of 5%
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while creating swap transaction: {http_err} - {response.text}")
        logging.error(f"HTTP error while creating swap transaction: {http_err} - {response.text}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request exception occurred while creating swap transaction: {req_err}")
        logging.error(f"Request exception occurred while creating swap transaction: {req_err}")
    except ValueError as val_err:
        print(f"Value error: {val_err} - Response: {response.text}")
        logging.error(f"Value error while creating swap transaction: {val_err} - {response.text}")
    return {}

def send_transaction(swap_tx):
    """
    Sends the swap transaction to the Solana blockchain.

    Args:
        swap_tx (bytes): The serialized swap transaction.

    Returns:
        str: The transaction signature (txId) if successful, None otherwise.
    """
    try:
        versioned_tx = VersionedTransaction.from_bytes(swap_tx)
        signed_tx = VersionedTransaction(versioned_tx.message, [KEY])
        response = http_client.send_raw_transaction(bytes(signed_tx), TxOpts(skip_preflight=False))
        logging.info(f"Raw response from send_raw_transaction: {response}")
        
        # Extract the signature from the response object
        tx_id = str(response.value)  # Assuming 'value' holds the Signature object
        
        if tx_id:
            print(f"Transaction submitted with TxID: {tx_id}")
            logging.info(f"Transaction submitted with TxID: {tx_id}")
            # Confirm the transaction
            if confirm_transaction(tx_id):
                return tx_id
            else:
                print(f"Transaction {tx_id} failed to confirm.")
                logging.error(f"Transaction {tx_id} failed to confirm.")
                return None
        else:
            print("Failed to send transaction:", response)
            logging.error(f"Failed to send transaction: {response}")
            return None
    except Exception as e:
        print(f"An error occurred while sending the transaction: {e}")
        logging.error(f"An error occurred while sending the transaction: {e}")
        return None

def execute_with_exponential_backoff(func, *args, **kwargs):
    """
    Executes a function with retries upon failure, implementing exponential backoff.

    Args:
        func (callable): The function to execute.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        The result of the function if successful, else None.
    """
    backoff = INITIAL_RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429:
                print(f"Attempt {attempt} failed with 429 Too Many Requests.")
                logging.warning(f"Attempt {attempt} failed with 429 Too Many Requests.")
            else:
                print(f"Attempt {attempt} failed with HTTP error: {http_err}")
                logging.error(f"Attempt {attempt} failed with HTTP error: {http_err}")
        except Exception as e:
            print(f"Attempt {attempt} failed with exception: {e}")
            logging.error(f"Attempt {attempt} failed with exception: {e}")
        
        if attempt < MAX_RETRIES:
            print(f"Retrying in {backoff} seconds...")
            logging.info(f"Retrying in {backoff} seconds...")
            time.sleep(backoff)
            backoff *= 2  # Exponential backoff
        else:
            print("Max retries reached. Aborting.")
            logging.error("Max retries reached. Aborting.")
    return None

def place_buy_order():
    """
    Places a buy order swapping SOL to the specified token for a defined USD amount.
    """

    sol_price = get_sol_price()
    if sol_price == 0:
        print("Cannot fetch SOL price. Aborting buy order.")
        logging.error("Cannot fetch SOL price. Aborting buy order.")
        return

    # Calculate amount of SOL needed for the specified USD amount
    sol_amount = USDT_Quantity / sol_price  # USD / (USD/SOL) = SOL
    lamports = int(sol_amount * 1_000_000_000)  # Convert SOL to lamports

    print(f"Placing Buy Order: Swapping {sol_amount:.9f} SOL (${USDT_Quantity}) to Token")
    logging.info(f"Placing Buy Order: Swapping {sol_amount:.9f} SOL (${USDT_Quantity}) to Token")

    # Log the calculated lamports amount
    logging.info(f"Calculated SOL amount in lamports for buy order: {lamports}")

    # Get quote for swapping SOL to Token
    quote = get_quote(QUOTE_TOKEN, OUTPUT_TOKEN, lamports)
    if not quote:
        print("No quote received. Aborting buy order.")
        logging.error("No quote received. Aborting buy order.")
        return
    #pprint(quote)
    logging.info(f"Quote received for Buy Order: {quote}")

    # Create swap transaction
    tx_res = create_swap_transaction(quote)
    if not tx_res:
        print("Failed to create swap transaction. Aborting buy order.")
        logging.error("Failed to create swap transaction. Aborting buy order.")
        return
    #pprint(tx_res)
    logging.info(f"Swap transaction created: {tx_res}")

    # Decode and send transaction with retries
    swap_tx_encoded = tx_res.get('swapTransaction', '')
    if not swap_tx_encoded:
        print("No swap transaction data available. Aborting buy order.")
        logging.error("No swap transaction data available. Aborting buy order.")
        return
    try:
        swap_tx = base64.b64decode(swap_tx_encoded)
    except base64.binascii.Error as e:
        print(f"Error decoding swap transaction: {e}")
        logging.error(f"Error decoding swap transaction: {e}")
        return

    tx_id = execute_with_exponential_backoff(send_transaction, swap_tx)
    log_transaction(tx_id, bool(tx_id))

def place_sell_order():
    """
    Places a sell order swapping the specified token back to SOL for a defined USD amount.
    """

    token_price_usd = get_cached_token_price(OUTPUT_TOKEN)
    if token_price_usd is None or token_price_usd == 0:
        print("Cannot fetch Token price. Aborting sell order.")
        logging.error("Cannot fetch Token price. Aborting sell order.")
        return

    # Calculate amount of tokens needed for the specified USD amount
    token_amount = USDT_Quantity / token_price_usd  # USD / (USD/Token) = Token
    token_amount_int = int(token_amount * (10 ** token_decimals))  # Adjust based on token decimals

    print(f"Placing Sell Order: Swapping {token_amount:.6f} Token (${USDT_Quantity}) to SOL")
    logging.info(f"Placing Sell Order: Swapping {token_amount:.6f} Token (${USDT_Quantity}) to SOL")

    # Log the calculated token amount
    logging.info(f"Calculated token amount for sell order: {token_amount_int}")

    # Get quote for swapping Token to SOL
    quote = get_quote(OUTPUT_TOKEN, QUOTE_TOKEN, token_amount_int)
    if not quote:
        print("No quote received. Aborting sell order.")
        logging.error("No quote received. Aborting sell order.")
        return
    #pprint(quote)
    logging.info(f"Quote received for Sell Order: {quote}")

    # Create swap transaction
    tx_res = create_swap_transaction(quote)
    if not tx_res:
        print("Failed to create swap transaction. Aborting sell order.")
        logging.error("Failed to create swap transaction. Aborting sell order.")
        return
    #pprint(tx_res)
    logging.info(f"Swap transaction created: {tx_res}")

    # Decode and send transaction with retries
    swap_tx_encoded = tx_res.get('swapTransaction', '')
    if not swap_tx_encoded:
        print("No swap transaction data available. Aborting sell order.")
        logging.error("No swap transaction data available. Aborting sell order.")
        return
    try:
        swap_tx = base64.b64decode(swap_tx_encoded)
    except base64.binascii.Error as e:
        print(f"Error decoding swap transaction: {e}")
        logging.error(f"Error decoding swap transaction: {e}")
        return

    tx_id = execute_with_exponential_backoff(send_transaction, swap_tx)
    log_transaction(tx_id, bool(tx_id))

def get_quote(input_mint, output_mint, amount):
    """
    Fetches a quote from Jupiter's API for swapping tokens.

    Args:
        input_mint (str): The mint address of the input token.
        output_mint (str): The mint address of the output token.
        amount (int): The amount of input token in lamports.

    Returns:
        dict: The quote response from Jupiter.
    """
    url = (
        'https://quote-api.jup.ag/v6/quote'
        f'?inputMint={input_mint}&outputMint={output_mint}&amount={amount}&slippageBps={SLIPPAGE}'
    )
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while fetching quote: {http_err} - {response.text}")
        logging.error(f"HTTP error while fetching quote: {http_err} - {response.text}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request exception occurred while fetching quote: {req_err}")
        logging.error(f"Request exception while fetching quote: {req_err}")
    except ValueError as val_err:
        print(f"Value error: {val_err} - Response: {response.text}")
        logging.error(f"Value error while fetching quote: {val_err} - {response.text}")
    return {}

def get_sol_price():
    """
    Retrieves the current SOL price in USD using Birdeye's Price API.

    Returns:
        float: The USD price of SOL or 0.0 if unavailable.
    """
    # Using Birdeye's Price API to get SOL price
    sol_address = 'So11111111111111111111111111111111111111112'  # SOL Mint Address
    price = token_price(sol_address)
    if price is None:
        print("Failed to fetch SOL price.")
        logging.error("Failed to fetch SOL price.")
        return 0.0
    return price

def log_transaction(tx_id, status):
    """
    Logs the transaction status.

    Args:
        tx_id (str): The transaction ID.
        status (bool): True if successful, False otherwise.
    """
    if status:
        logging.info(f"Transaction successful! TxID: {tx_id}")
    else:
        logging.error(f"Transaction failed: {tx_id}")

if __name__ == "__main__":
    print("Starting Trading Bot...")
    logging.info("Trading Bot started.")
    try:
        place_buy_order()
        time.sleep(10)  # Wait for 10 seconds before placing the sell order
        place_sell_order()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logging.error(f"An unexpected error occurred: {e}")
    print("Trading Bot Finished.")
    logging.info("Trading Bot finished.")