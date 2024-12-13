import requests
import json
import base64
import time
import logging
import csv
from datetime import datetime
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solana.rpc.api import Client
from solana.rpc.types import TxOpts
import dontshare as d
from functools import lru_cache

# Configure logging
logging.basicConfig(
    filename='auto_trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Constants
SLIPPAGE = 1000  # Slippage in basis points (1%)
MAX_RETRIES = 5  # Maximum number of retries for requests
INITIAL_RETRY_DELAY = 5  # Initial delay between retries in seconds
CSV_FILE = 'trade_log.csv'

# Initialize Solana RPC client
RPC_URL = "https://methodical-sleek-smoke.solana-mainnet.quiknode.pro/ce998890a7f93d71ff7c2f1979abf0510bf40d80"
http_client = Client(RPC_URL)

# Load wallet keypair
KEY = Keypair.from_base58_string(d.key)
USER_PUBLIC_KEY = str(KEY.pubkey())

# Token configuration
QUOTE_TOKEN = 'So11111111111111111111111111111111111111112'  # SOL
OUTPUT_TOKEN = '2qEHjDLDLbuBgRYvsxhc5D6uDWAivNFZGan56P1tpump'  # Replace with your target token address

def get_decimals(token_mint_address):
    """
    Retrieves the number of decimals for a given token mint address on Solana.
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
    except Exception as e:
        logging.error(f"Error fetching decimals: {e}")
    return 0

def token_price(address):
    """
    Retrieves the current USD price for a given token mint address using Birdeye's Price API.
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
    except Exception as e:
        logging.error(f"Error fetching price: {e}")
    return None

@lru_cache(maxsize=32)
def get_cached_token_price(mint_address):
    return token_price(mint_address)

def confirm_transaction(tx_id, timeout=60):
    """
    Confirms the transaction on the Solana blockchain.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = http_client.get_signature_statuses([tx_id])
            statuses = response['result']['value']
            if statuses and statuses[0]:
                confirmation = statuses[0].get('confirmationStatus')
                if confirmation in ('confirmed', 'finalized'):
                    logging.info(f"Transaction {tx_id} confirmed with status: {confirmation}")
                    return True
            time.sleep(2)
        except Exception as e:
            logging.error(f"Error confirming transaction {tx_id}: {e}")
            time.sleep(2)
    logging.error(f"Transaction {tx_id} not confirmed within {timeout} seconds.")
    return False

def create_swap_transaction(quote):
    """
    Creates a swap transaction based on the quote provided by Jupiter.
    """
    if not quote:
        logging.error("Empty quote received, cannot create swap transaction.")
        return {}
    url = 'https://quote-api.jup.ag/v6/swap'
    headers = {"Content-Type": "application/json"}
    payload = {
        "quoteResponse": quote,
        "userPublicKey": USER_PUBLIC_KEY
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Error creating swap transaction: {e}")
    return {}

def send_transaction(swap_tx):
    """
    Sends the swap transaction to the Solana blockchain.
    """
    try:
        versioned_tx = VersionedTransaction.from_bytes(swap_tx)
        signed_tx = VersionedTransaction(versioned_tx.message, [KEY])
        response = http_client.send_raw_transaction(bytes(signed_tx), TxOpts(skip_preflight=False))
        tx_id = response
        if tx_id:
            logging.info(f"Transaction submitted with TxID: {tx_id}")
            if confirm_transaction(tx_id):
                return tx_id
        logging.error("Failed to send transaction.")
    except Exception as e:
        logging.error(f"Error while sending transaction: {e}")
    return None

def execute_with_exponential_backoff(func, *args, **kwargs):
    """
    Executes a function with retries upon failure, implementing exponential backoff.
    """
    backoff = INITIAL_RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Attempt {attempt} failed with exception: {e}")
        
        if attempt < MAX_RETRIES:
            logging.info(f"Retrying in {backoff} seconds...")
            time.sleep(backoff)
            backoff *= 2  # Exponential backoff
        else:
            logging.error("Max retries reached. Aborting.")
    return None

def get_quote(input_mint, output_mint, amount):
    """
    Fetches a quote from Jupiter's API for swapping tokens.
    """
    url = (
        'https://quote-api.jup.ag/v6/quote'
        f'?inputMint={input_mint}&outputMint={output_mint}&amount={amount}&slippageBps={SLIPPAGE}'
    )
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Error fetching quote: {e}")
    return {}

def get_sol_price():
    """
    Retrieves the current SOL price in USD using Birdeye's Price API.
    """
    sol_address = 'So11111111111111111111111111111111111111112'  # SOL Mint Address
    price = token_price(sol_address)
    if price is None:
        logging.error("Failed to fetch SOL price.")
        return 0.0
    return price

def log_transaction(tx_id, status, usd_amount, token_amount, slippage, gas_cost):
    """
    Logs the transaction status and details to a CSV file.
    """
    if status:
        logging.info(f"Transaction successful! TxID: {tx_id}")
    else:
        logging.error(f"Transaction failed: {tx_id}")

    # Log to CSV
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            tx_id,
            usd_amount,
            token_amount,
            slippage,
            gas_cost,
            "Success" if status else "Failed"
        ])

def get_token_balance(token_address):
    """
    Gets the token balance for the user's wallet.
    """
    try:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTokenAccountsByOwner",
            "params": [
                USER_PUBLIC_KEY,
                {
                    "mint": token_address
                },
                {
                    "encoding": "jsonParsed"
                }
            ]
        }
        
        response = requests.post(RPC_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        
        if data['result']['value']:
            balance = int(data['result']['value'][0]['account']['data']['parsed']['info']['tokenAmount']['amount'])
            return balance
        return 0
    except Exception as e:
        logging.error(f"Error fetching token balance: {e}")
        return 0

def place_buy_order(usd_amount):
    """
    Places a buy order swapping SOL to the specified token for a defined USD amount.
    """
    sol_price = get_sol_price()
    if sol_price == 0:
        logging.error("Cannot fetch SOL price. Aborting buy order.")
        return

    sol_amount = usd_amount / sol_price  # USD / (USD/SOL) = SOL
    lamports = int(sol_amount * 1_000_000_000)  # Convert SOL to lamports

    logging.info(f"Placing Buy Order: Swapping {sol_amount:.9f} SOL (${usd_amount}) to Token")

    quote = get_quote(QUOTE_TOKEN, OUTPUT_TOKEN, lamports)
    if not quote:
        logging.error("No quote received. Aborting buy order.")
        return

    tx_res = create_swap_transaction(quote)
    if not tx_res:
        logging.error("Failed to create swap transaction. Aborting buy order.")
        return

    swap_tx_encoded = tx_res.get('swapTransaction', '')
    if not swap_tx_encoded:
        logging.error("No swap transaction data available. Aborting buy order.")
        return
    try:
        swap_tx = base64.b64decode(swap_tx_encoded)
    except base64.binascii.Error as e:
        logging.error(f"Error decoding swap transaction: {e}")
        return

    tx_id = execute_with_exponential_backoff(send_transaction, swap_tx)
    log_transaction(tx_id, bool(tx_id), usd_amount, lamports, SLIPPAGE, "N/A")  # Gas cost is not available

def place_sell_order():
    """
    Places a sell order swapping the specified token back to SOL for 100% of the token amount.
    """
    token_price_usd = get_cached_token_price(OUTPUT_TOKEN)
    if token_price_usd is None or token_price_usd == 0:
        logging.error("Cannot fetch Token price. Aborting sell order.")
        return

    # Get actual token balance
    token_amount = get_token_balance(OUTPUT_TOKEN)
    if token_amount == 0:
        logging.error("No tokens available to sell. Aborting sell order.")
        return

    logging.info(f"Placing Sell Order: Swapping {token_amount} Token to SOL")

    quote = get_quote(OUTPUT_TOKEN, QUOTE_TOKEN, token_amount)
    if not quote:
        logging.error("No quote received. Aborting sell order.")
        return

    tx_res = create_swap_transaction(quote)
    if not tx_res:
        logging.error("Failed to create swap transaction. Aborting sell order.")
        return

    swap_tx_encoded = tx_res.get('swapTransaction', '')
    if not swap_tx_encoded:
        logging.error("No swap transaction data available. Aborting sell order.")
        return
    try:
        swap_tx = base64.b64decode(swap_tx_encoded)
    except base64.binascii.Error as e:
        logging.error(f"Error decoding swap transaction: {e}")
        return

    tx_id = execute_with_exponential_backoff(send_transaction, swap_tx)
    log_transaction(tx_id, bool(tx_id), "N/A", token_amount, SLIPPAGE, "N/A")  # Gas cost is not available

def initialize_csv():
    """
    Initializes the CSV file with headers if it doesn't exist.
    """
    try:
        with open(CSV_FILE, mode='x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "TxID", "USD Amount", "Token Amount", "Slippage", "Gas Cost", "Status"])
    except FileExistsError:
        pass  # File already exists

if __name__ == "__main__":
    logging.info("Starting Automated Trading Bot...")
    initialize_csv()
    try:
        usd_amount_to_buy = 1  # Set the USD amount you want to spend on buying the token
        place_buy_order(usd_amount_to_buy)
        time.sleep(15)  # Wait for 15 seconds before placing the sell order
        place_sell_order()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    logging.info("Automated Trading Bot Finished.")