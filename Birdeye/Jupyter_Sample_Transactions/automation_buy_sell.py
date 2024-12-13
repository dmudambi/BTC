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
import asyncio
import subprocess
import threading
from dexscreener_pricefeed import get_token_price, get_token_name  # Importing the token name
# from jupiter_pricefeed import get_jupiter_price  # Importing Jupiter price feed function
import os
import pandas as pd
from solders.signature import Signature  # Add this import at the top

# Directory setup
current_dir = os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Configure logging with more detailed settings
logging.basicConfig(level=logging.INFO)  # Set basic config first

# Create a logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove all existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create file handler
file_handler = logging.FileHandler('auto_trading_bot.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(file_handler)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(console_handler)

# Constants
SLIPPAGE = 100  # Slippage in basis points (1%)
MAX_RETRIES = 5  # Maximum number of retries for requests
INITIAL_RETRY_DELAY = 5  # Initial delay between retries in seconds
CSV_FILE = 'trade_log.csv'
PRICE_CHANGE_THRESHOLD = 3.0  # Price change percentage to trigger buy/sell

# Initialize Solana RPC client
RPC_URL = "https://methodical-sleek-smoke.solana-mainnet.quiknode.pro/ce998890a7f93d71ff7c2f1979abf0510bf40d80"
http_client = Client(RPC_URL)

# Load wallet keypair
KEY = Keypair.from_base58_string(d.key)
USER_PUBLIC_KEY = str(KEY.pubkey())

# Token configuration
QUOTE_TOKEN = 'So11111111111111111111111111111111111111112'  # SOL
OUTPUT_TOKEN = 'HMdmbv35DvbH7VxWSwXaxE1NNyA34HHhs9Cu9ycZpump'  # Target token address
OUTPUT_TOKEN_NAME = get_token_name(OUTPUT_TOKEN)
PAIR_ADDRESS = "FmKAfMMnxRMaqG1c4emgA4AhaThi4LQ4m2A12hwoTibb"

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
        logger.error(f"Error fetching decimals: {e}")
    return 0

async def get_sol_price():
    """
    Retrieves the current SOL price in USD using Dexscreener's Price API.
    """
    sol_address = 'So11111111111111111111111111111111111111112'  # SOL Mint Address
    price = await get_token_price(sol_address)  # Await the coroutine here
    if price is None:
        logger.error("Failed to fetch SOL price.")
        return 0.0
    return price

def confirm_transaction(tx_id, timeout=120):
    """
    Confirms the transaction with aggressive retry strategy and enhanced logging.
    """
    start_time = time.time()
    signature = Signature.from_string(tx_id)
    check_interval = 3  # Increased interval between checks
    
    while time.time() - start_time < timeout:
        try:
            response = http_client.get_signature_statuses(
                [signature],
                search_transaction_history=True
            )
            
            if response.value[0] is None:
                elapsed = time.time() - start_time
                logger.info(f"Transaction {tx_id[:8]}... pending for {elapsed:.1f}s...")
                time.sleep(check_interval)
                continue
                
            status = response.value[0].confirmation_status
            elapsed = time.time() - start_time
            
            if status == 'finalized':
                logger.info(f"Transaction {tx_id[:8]}... finalized after {elapsed:.1f}s!")
                return True
            elif status == 'confirmed':
                logger.info(f"Transaction {tx_id[:8]}... confirmed after {elapsed:.1f}s, waiting for finalization...")
            elif status == 'processed':
                logger.info(f"Transaction {tx_id[:8]}... processed after {elapsed:.1f}s, waiting for confirmation...")
            
            time.sleep(check_interval)
            
        except Exception as e:
            logger.error(f"Error checking transaction {tx_id[:8]}...: {e}")
            time.sleep(check_interval)
    
    logger.error(f"Transaction {tx_id[:8]}... timed out after {timeout} seconds")
    return False

def create_swap_transaction(quote):
    """
    Creates a swap transaction based on the quote provided by Jupiter.
    """
    if not quote:
        logger.error("Empty quote received, cannot create swap transaction.")
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
        logger.error(f"Error creating swap transaction: {e}")
    return {}

def simulate_transaction(swap_tx):
    """
    Simulates the transaction before sending.
    """
    try:
        # Convert bytes to VersionedTransaction using from_bytes
        versioned_tx = VersionedTransaction.from_bytes(swap_tx)
        
        # Simulate the transaction without options
        simulation = http_client.simulate_transaction(versioned_tx)
        
        if simulation.value.err:
            logger.error(f"Transaction simulation failed: {simulation.value.err}")
            return False
            
        logger.info("Transaction simulation successful")
        return True
        
    except Exception as e:
        logger.error(f"Error simulating transaction: {e}")
        return False

def send_transaction(swap_tx):
    """
    Sends the swap transaction to the Solana blockchain with improved settings.
    """
    if not simulate_transaction(swap_tx):
        logger.error("Transaction simulation failed, aborting send")
        return None
        
    try:
        versioned_tx = VersionedTransaction.from_bytes(swap_tx)
        signed_tx = VersionedTransaction(versioned_tx.message, [KEY])
        
        # More aggressive transaction options
        opts = TxOpts(
            skip_preflight=True,
            max_retries=15,  # Increased retries
            preflight_commitment="processed"  # Changed commitment level
        )
        
        logger.info("Sending transaction with aggressive settings...")
        response = http_client.send_raw_transaction(
            bytes(signed_tx),
            opts=opts
        )
        
        tx_id = str(response.value)
        
        if tx_id:
            logger.info(f"Transaction submitted with TxID: {tx_id}")
            time.sleep(2)  # Reduced initial wait time
            if confirm_transaction(tx_id, timeout=120):  # Increased timeout
                return tx_id
            else:
                logger.error("Transaction failed confirmation")
                return None
        else:
            logger.error("Failed to send transaction - no transaction ID received")
            return None
            
    except Exception as e:
        logger.error(f"Error while sending transaction: {str(e)}")
        return None

def execute_with_exponential_backoff(func, *args, **kwargs):
    """
    Executes a function with exponential backoff retry strategy.
    """
    max_attempts = 5
    base_delay = 2
    
    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_attempts - 1:
                raise e
            
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay} seconds...")
            time.sleep(delay)

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
        logger.error(f"Error fetching quote: {e}")
    return {}

def log_transaction(tx_id, status, usd_amount, token_amount, slippage, gas_cost):
    """
    Logs the transaction status and details to a CSV file.
    """
    if status:
        logger.info(f"Transaction successful! TxID: {tx_id}")
    else:
        logger.error(f"Transaction failed: {tx_id}")

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
        logger.error(f"Error fetching token balance: {e}")
        return 0

def place_buy_order(usd_amount):
    """
    Places a buy order swapping SOL to the specified token for a defined USD amount.
    """
    sol_price = get_sol_price()
    if sol_price == 0:
        logger.error("Cannot fetch SOL price. Aborting buy order.")
        return

    sol_amount = usd_amount / sol_price  # USD / (USD/SOL) = SOL
    lamports = int(sol_amount * 1_000_000_000)  # Convert SOL to lamports

    logger.info("="*30)
    logger.info("Placing Buy Order:")
    logger.info(f"Token Name: {OUTPUT_TOKEN_NAME}")
    logger.info(f"Token Address: {OUTPUT_TOKEN}")
    logger.info(f"Price Source: DexScreener API")
    logger.info(f"Swapping {sol_amount:.9f} SOL (${usd_amount}) to {OUTPUT_TOKEN_NAME}")
    logger.info("="*30)

    quote = get_quote(QUOTE_TOKEN, OUTPUT_TOKEN, lamports)
    if not quote:
        logger.error("No quote received. Aborting buy order.")
        return

    tx_res = create_swap_transaction(quote)
    if not tx_res:
        logger.error("Failed to create swap transaction. Aborting buy order.")
        return

    swap_tx_encoded = tx_res.get('swapTransaction', '')
    if not swap_tx_encoded:
        logger.error("No swap transaction data available. Aborting buy order.")
        return
    try:
        swap_tx = base64.b64decode(swap_tx_encoded)
    except base64.binascii.Error as e:
        logger.error(f"Error decoding swap transaction: {e}")
        return

    tx_id = execute_with_exponential_backoff(send_transaction, swap_tx)
    log_transaction(tx_id, bool(tx_id), usd_amount, lamports, SLIPPAGE, "N/A")  # Gas cost is not available

def place_sell_order():
    """
    Places a sell order swapping the specified token back to SOL for 100% of the token amount.
    """
    token_price_usd = get_token_price(OUTPUT_TOKEN)
    if token_price_usd is None or token_price_usd == 0:
        logger.error("Cannot fetch Token price. Aborting sell order.")
        return

    # Get actual token balance
    token_amount = get_token_balance(OUTPUT_TOKEN)
    if token_amount == 0:
        logger.error("No tokens available to sell. Aborting sell order.")
        return

    logger.info("="*30)
    logger.info("Placing Sell Order:")
    logger.info(f"Token Name: {OUTPUT_TOKEN_NAME}")
    logger.info(f"Token Address: {OUTPUT_TOKEN}")
    logger.info(f"Price Source: DexScreener API")
    logger.info(f"Swapping {token_amount} {OUTPUT_TOKEN_NAME} to SOL")
    logger.info("="*30)

    quote = get_quote(OUTPUT_TOKEN, QUOTE_TOKEN, token_amount)
    if not quote:
        logger.error("No quote received. Aborting sell order.")
        return

    tx_res = create_swap_transaction(quote)
    if not tx_res:
        logger.error("Failed to create swap transaction. Aborting sell order.")
        return

    swap_tx_encoded = tx_res.get('swapTransaction', '')
    if not swap_tx_encoded:
        logger.error("No swap transaction data available. Aborting sell order.")
        return
    try:
        swap_tx = base64.b64decode(swap_tx_encoded)
    except base64.binascii.Error as e:
        logger.error(f"Error decoding swap transaction: {e}")
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

class TokenTradeManager:
    def __init__(self):
        self.monitored_tokens = set()  # Currently monitored tokens
        self.trade_lock = asyncio.Lock()
        self.trade_log_file = 'trade_history.csv'
        self.initialize_trade_log()

    def initialize_trade_log(self):
        """Initialize trade log CSV with headers"""
        headers = [
            'timestamp',
            'token_name',
            'token_address',
            'trade_type',
            'dollar_amount',
            'total_pnl',
            'slippage_percentage',
            'holding_time_seconds',
            'execution_time_seconds',
            'status',
            'retry_count'
        ]
        
        if not os.path.exists(self.trade_log_file):
            with open(self.trade_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    async def monitor_qualified_tokens(self):
        """Monitor qualified tokens directory for new files"""
        base_path = os.path.join(root_dir, 'Bots', 'Fib', 'Data', 'New_Token_Data')
        
        while True:
            try:
                # Check if base path exists
                if not os.path.exists(base_path):
                    logger.error(f"Base path does not exist: {base_path}")
                    await asyncio.sleep(30)
                    continue

                # Get all date folders and find the most recent one
                date_folders = [
                    f for f in os.listdir(base_path) 
                    if os.path.isdir(os.path.join(base_path, f)) 
                    and f[0].isdigit()  # Ensure it's a date folder
                ]
                
                if not date_folders:
                    logger.warning(f"No date folders found in {base_path}")
                    await asyncio.sleep(30)
                    continue

                # Sort by date (assuming YYYY_MM_DD format)
                most_recent_date = max(date_folders)
                date_folder = os.path.join(base_path, most_recent_date, 'Results')
                
                if not os.path.exists(date_folder):
                    logger.warning(f"No Results folder found in most recent date folder: {most_recent_date}")
                    await asyncio.sleep(30)
                    continue

                # Get most recent time folder
                time_folders = [
                    f for f in os.listdir(date_folder) 
                    if os.path.isdir(os.path.join(date_folder, f))
                ]
                
                if not time_folders:
                    logger.warning(f"No time folders found in {date_folder}")
                    await asyncio.sleep(30)
                    continue

                latest_time_folder = max(time_folders)
                latest_file = os.path.join(date_folder, latest_time_folder, 'qualified_tokens.csv')
                
                if not os.path.exists(latest_file):
                    logger.warning(f"No qualified_tokens.csv found in {latest_time_folder}")
                    await asyncio.sleep(30)
                    continue

                logger.info(f"Processing qualified tokens from: {latest_file}")

                # Read new token set
                new_tokens = set()
                df = pd.read_csv(latest_file)
                for _, row in df.iterrows():
                    new_tokens.add(row['token_address'])

                # Handle token updates
                tokens_to_add = new_tokens - self.monitored_tokens
                tokens_to_remove = self.monitored_tokens - new_tokens

                # Remove tokens no longer in list
                for token in tokens_to_remove:
                    self.monitored_tokens.remove(token)
                    logger.info(f"Stopped monitoring token: {token}")

                # Add new tokens
                for token in tokens_to_add:
                    self.monitored_tokens.add(token)
                    asyncio.create_task(self.monitor_and_trade_token(token))
                    logger.info(f"Started monitoring token: {token}")

                # Log current monitoring status
                logger.info(f"Currently monitoring {len(self.monitored_tokens)} tokens")
                
                await asyncio.sleep(30)  # Check for new files every 30 seconds

            except Exception as e:
                logger.error(f"Error in monitor_qualified_tokens: {e}")
                await asyncio.sleep(30)

    async def monitor_and_trade_token(self, token_address):
        """Execute trade cycles for a token"""
        while token_address in self.monitored_tokens:
            try:
                # Buy order
                logger.info(f"\n{'='*50}")
                logger.info(f"Starting trade cycle for {token_address}")
                
                buy_result = await self.execute_buy_order(token_address, 1.0)  # $1.0 USD
                
                if not buy_result['success']:
                    logger.error(f"Buy order failed for {token_address}, waiting for next cycle...")
                    await asyncio.sleep(120)  # Wait 2 minutes before next attempt
                    continue
                
                # Wait 15 seconds before selling
                logger.info("Waiting 15 seconds before executing sell order...")
                await asyncio.sleep(15)
                
                # Sell order
                sell_result = await self.execute_sell_order(token_address)
                
                # Log the complete trade cycle
                self.log_trade_cycle(
                    token_name=get_token_name(token_address),
                    token_address=token_address,
                    buy_status=buy_result,
                    sell_status=sell_result,
                    execution_time=time.time() - start_time,
                    holding_time=15  # 15 seconds holding time
                )
                
                # Wait for next cycle
                logger.info(f"Trade cycle completed for {token_address}")
                logger.info("Waiting 120 seconds before next cycle...")
                await asyncio.sleep(120)  # 2 minutes between cycles
                
            except Exception as e:
                logger.error(f"Error in trade cycle for {token_address}: {e}")
                await asyncio.sleep(120)  # Wait 2 minutes before retrying on error

    def log_trade_cycle(self, token_name, token_address, buy_status, sell_status, execution_time, holding_time):
        """Log trade cycle details to CSV"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Calculate total P&L
        total_pnl = sell_status.get('amount_received', 0) - buy_status.get('amount_spent', 0)
        
        # Log buy transaction
        buy_row = [
            timestamp,
            token_name,
            token_address,
            'BUY',
            buy_status.get('amount_spent', 0),
            0,  # P&L not applicable for buy
            buy_status.get('slippage', 0),
            0,  # Holding time not applicable for buy
            execution_time,
            'SUCCESS' if buy_status['success'] else 'FAILED',
            buy_status.get('retries', 0)
        ]
        
        # Log sell transaction
        sell_row = [
            timestamp,
            token_name,
            token_address,
            'SELL',
            sell_status.get('amount_received', 0),
            total_pnl,
            sell_status.get('slippage', 0),
            holding_time,
            execution_time,
            'SUCCESS' if sell_status['success'] else 'FAILED',
            sell_status.get('retries', 0)
        ]
        
        with open(self.trade_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(buy_row)
            writer.writerow(sell_row)

    async def execute_buy_order(self, token_address, usd_amount):
        """Execute a buy order for a token"""
        try:
            start_time = time.time()
            retries = 0
            max_retries = 3
            
            logger.info(f"\nInitiating buy order for {token_address}")
            logger.info(f"Amount: ${usd_amount} USD")
            
            # Check if we have enough SOL for fees (0.01 SOL buffer)
            sol_balance = float(http_client.get_balance(KEY.pubkey()).value) / 1e9
            if sol_balance < 0.01:
                logger.error(f"Insufficient SOL balance for fees: {sol_balance}")
                return {'success': False, 'error': 'Insufficient balance for fees'}
            
            while retries < max_retries:
                try:
                    # Get current SOL price
                    sol_price = await get_sol_price()  # Await the coroutine here
                    if sol_price == 0:
                        raise ValueError("Cannot fetch SOL price")
                    
                    logger.info(f"Current SOL price: ${sol_price}")
                    
                    sol_amount = usd_amount / sol_price
                    lamports = int(sol_amount * 1_000_000_000)
                    logger.info(f"Converting ${usd_amount} to {sol_amount:.6f} SOL ({lamports} lamports)")

                    # Get quote
                    logger.info("Fetching quote from Jupiter...")
                    quote = get_quote(QUOTE_TOKEN, token_address, lamports)
                    if not quote:
                        raise ValueError("No quote received")
                    
                    logger.info("Creating swap transaction...")
                    tx_res = create_swap_transaction(quote)
                    if not tx_res:
                        raise ValueError("Failed to create swap transaction")

                    swap_tx_encoded = tx_res.get('swapTransaction', '')
                    if not swap_tx_encoded:
                        raise ValueError("No swap transaction data")

                    logger.info("Sending transaction...")
                    swap_tx = base64.b64decode(swap_tx_encoded)
                    tx_id = send_transaction(swap_tx)

                    if tx_id:
                        logger.info(f"Buy order successful! TX ID: {tx_id}")
                        return {
                            'success': True,
                            'amount_spent': usd_amount,
                            'slippage': SLIPPAGE / 100,
                            'retries': retries,
                            'tx_id': tx_id
                        }

                except Exception as e:
                    logger.error(f"Buy attempt {retries + 1} failed: {str(e)}")
                    retries += 1
                    if retries < max_retries:
                        wait_time = 2 ** retries
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        await asyncio.sleep(wait_time)
            
            logger.error("All buy attempts failed")
            return {
                'success': False,
                'amount_spent': 0,
                'slippage': 0,
                'retries': retries
            }

        except Exception as e:
            logger.error(f"Error in execute_buy_order: {str(e)}")
            return {
                'success': False,
                'amount_spent': 0,
                'slippage': 0,
                'retries': retries
            }

    async def execute_sell_order(self, token_address):
        """Execute a sell order for a token"""
        try:
            start_time = time.time()
            retries = 0
            max_retries = 3
            
            while retries < max_retries:
                try:
                    # Get token balance
                    token_amount = get_token_balance(token_address)
                    if token_amount == 0:
                        raise ValueError("No tokens available to sell")

                    # Get quote for selling to SOL
                    quote = get_quote(token_address, QUOTE_TOKEN, token_amount)
                    if not quote:
                        raise ValueError("No quote received")

                    # Create and execute swap transaction
                    tx_res = create_swap_transaction(quote)
                    if not tx_res:
                        raise ValueError("Failed to create swap transaction")

                    swap_tx_encoded = tx_res.get('swapTransaction', '')
                    if not swap_tx_encoded:
                        raise ValueError("No swap transaction data")

                    swap_tx = base64.b64decode(swap_tx_encoded)
                    tx_id = send_transaction(swap_tx)

                    if tx_id:
                        # Calculate amount received in USD
                        sol_amount = quote.get('outAmount', 0) / 1_000_000_000  # Convert lamports to SOL
                        sol_price = await get_sol_price()  # Await the coroutine here
                        usd_amount = sol_amount * sol_price

                        return {
                            'success': True,
                            'amount_received': usd_amount,
                            'slippage': SLIPPAGE / 100,
                            'retries': retries,
                            'tx_id': tx_id
                        }

                except Exception as e:
                    logger.error(f"Sell attempt {retries + 1} failed: {str(e)}")
                    retries += 1
                    if retries < max_retries:
                        await asyncio.sleep(2 ** retries)

            return {
                'success': False,
                'amount_received': 0,
                'slippage': 0,
                'retries': retries
            }

        except Exception as e:
            logger.error(f"Error in execute_sell_order: {str(e)}")
            return {
                'success': False,
                'amount_received': 0,
                'slippage': 0,
                'retries': retries
            }

async def main():
    try:
        # Add path verification logging
        logger.info("Starting main loop with directory setup:")
        logger.info(f"Current Directory: {current_dir}")
        logger.info(f"Root Directory: {root_dir}")
        
        # Initialize token trade manager
        trade_manager = TokenTradeManager()
        
        # Start monitoring qualified tokens
        await trade_manager.monitor_qualified_tokens()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")