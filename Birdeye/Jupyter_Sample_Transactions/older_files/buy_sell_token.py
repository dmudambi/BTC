import requests
import json
import base64
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solana.rpc.api import Client
from solana.rpc.types import TxOpts
import sys
import subprocess  # Import the subprocess module

# Configuration (Replace with your actual values or secure them properly)
import dontshare as d

KEY = Keypair.from_base58_string(d.key)
SLIPPAGE = 100  # Example slippage value
HTTP_CLIENT = Client("https://api.mainnet-beta.solana.com")

# --- SET YOUR DESIRED TRADE PARAMETERS HERE ---
QUOTE_TOKEN = "So11111111111111111111111111111111111111112"  # Mint address of the token you're using to buy or receiving when selling (e.g., SOL)
TOKEN_TO_TRADE = "HMdmbv35DvbH7VxWSwXaxE1NNyA34HHhs9Cu9ycZpump"  # Mint address of the token you want to buy or sell
USD_AMOUNT_BUY = 2  # USD amount you want to trade
USD_AMOUNT_SELL = 2
# ------------------------------------------------

def get_decimals(token_mint_address):
    """Fetches the number of decimals for a given token mint address."""
    url = "https://api.mainnet-beta.solana.com/"
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
    response = requests.post(url, headers=headers, data=payload)
    response_json = response.json()
    return response_json['result']['value']['data']['parsed']['info']['decimals']

def token_price(address):
    """Retrieves the current price of a token using Birdeye's API."""
    url = f"https://public-api.birdeye.so/defi/price?address={address}"
    headers = {"X-API-KEY": d.birdeye}
    response = requests.get(url, headers=headers)
    price_data = response.json()
    if price_data['success']:
        return price_data['data']['value']
    else:
        return None

def get_solana_balance(wallet_address):
    """Gets the SOL balance of a wallet using the Solana CLI."""
    try:
        # Use the Solana CLI to get the balance
        result = subprocess.run(
            ["solana", "balance", wallet_address, "--output", "json"],
            capture_output=True,
            text=True,
            check=True
        )
        balance_data = json.loads(result.stdout)
        # Extract the balance in lamports (the CLI returns it in SOL by default)
        balance_lamports = int(float(balance_data.split(" ")[0]) * 10**9)
        return balance_lamports
    except subprocess.CalledProcessError as e:
        print(f"Error getting balance: {e}")
        return None

def buy_token(quote_token, token, usd_amount):
    """Buys a specified USD amount of a token."""
    token_decimals = get_decimals(token)
    token_price_value = token_price(token)

    print(f"Token decimals: {token_decimals}")
    print(f"Token price: {token_price_value}")

    if token_price_value is None:
        print("Failed to retrieve token price.")
        return

    # Calculate the amount in lamports
    amount = int((usd_amount / token_price_value) * (10 ** token_decimals))
    print(f"Amount in lamports: {amount}")

    # Get wallet balance using Solana CLI
    wallet_balance = get_solana_balance(str(KEY.pubkey()))
    print(f"Wallet balance in lamports: {wallet_balance}")

    if wallet_balance is None:
        print("Failed to retrieve wallet balance.")
        return

    # Check if there's enough balance for the transaction
    if wallet_balance < amount:
        print(f"Insufficient funds: need {amount} lamports, have {wallet_balance} lamports")
        return

    # Get a quote
    quote_response = requests.get(
        f'https://quote-api.jup.ag/v6/quote?inputMint={quote_token}'
        f'&outputMint={token}&amount={amount}&slippageBps={SLIPPAGE}'
    ).json()

    # Construct the transaction
    tx_response = requests.post(
        'https://quote-api.jup.ag/v6/swap',
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "quoteResponse": quote_response,
            "userPublicKey": str(KEY.pubkey())
        })
    ).json()

    # Deserialize, sign, and send
    swap_tx = base64.b64decode(tx_response['swapTransaction'])
    tx = VersionedTransaction.from_bytes(swap_tx)
    tx = VersionedTransaction(tx.message, [KEY])
    response = HTTP_CLIENT.send_raw_transaction(
        bytes(tx),
        TxOpts(skip_preflight=False)
    )
    tx_id = response.get('result')

    if tx_id:
        print(f"Transaction successful: https://solscan.io/tx/{tx_id}")
    else:
        print("Failed to send transaction:", response)

def sell_token(quote_token, token, usd_amount):
    """Sells a specified USD amount of a token."""
    # Logic is similar to buy_token, but the input and output tokens are reversed
    token_decimals = get_decimals(quote_token)
    token_price_value = token_price(quote_token)

    if token_price_value is None:
        print("Failed to retrieve token price.")
        return

    amount = int((usd_amount / token_price_value) * (10 ** token_decimals))

    quote_response = requests.get(
        f'https://quote-api.jup.ag/v6/quote?inputMint={token}'
        f'&outputMint={quote_token}&amount={amount}&slippageBps={SLIPPAGE}'
    ).json()

    tx_response = requests.post(
        'https://quote-api.jup.ag/v6/swap',
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "quoteResponse": quote_response,
            "userPublicKey": str(KEY.pubkey())
        })
    ).json()

    swap_tx = base64.b64decode(tx_response['swapTransaction'])
    tx = VersionedTransaction.from_bytes(swap_tx)
    tx = VersionedTransaction(tx.message, [KEY])
    response = HTTP_CLIENT.send_raw_transaction(
        bytes(tx),
        TxOpts(skip_preflight=False)
    )
    tx_id = response.get('result')

    if tx_id:
        print(f"Transaction successful: https://solscan.io/tx/{tx_id}")
    else:
        print("Failed to send transaction:", response)

if __name__ == "__main__":
    # First, buy the token
    print(f"Buying ${USD_AMOUNT_BUY} worth of {TOKEN_TO_TRADE}...")
    buy_token(QUOTE_TOKEN, TOKEN_TO_TRADE, USD_AMOUNT_BUY)

    # Then, sell the token
    print(f"Selling ${USD_AMOUNT_SELL} worth of {TOKEN_TO_TRADE}...")
    sell_token(QUOTE_TOKEN, TOKEN_TO_TRADE, USD_AMOUNT_SELL) 