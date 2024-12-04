import requests
import json
from datetime import datetime
import time

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
        print(f"Error fetching token pairs: {str(e)}")
        return None

def format_price_change(change):
    """Format price change with color and arrow"""
    if change > 0:
        return f"↑ +{change}%"
    elif change < 0:
        return f"↓ {change}%"
    return f"→ {change}%"

def monitor_token(token_address, interval=5):
    """Monitor a token's most liquid pair with periodic updates"""
    print(f"Starting monitoring for token: {token_address}")
    
    # First, get all pairs for this token
    data = get_token_pairs(token_address)
    
    if not data or 'pairs' not in data or not data['pairs']:
        print("No pairs found for this token. Please verify the token address.")
        return
    
    # Sort pairs by liquidity to find the most liquid one
    pairs = sorted(data['pairs'], 
                  key=lambda x: float(x.get('liquidity', {}).get('usd', 0)), 
                  reverse=True)
    
    print(f"\nFound {len(pairs)} pairs. Using most liquid pair:")
    for i, pair in enumerate(pairs[:3], 1):  # Show top 3 pairs
        print(f"{i}. {pair.get('dexId')}/{pair.get('symbol')} - "
              f"Liquidity: ${float(pair.get('liquidity', {}).get('usd', 0)):,.2f}")
    
    most_liquid_pair = pairs[0]
    pair_address = most_liquid_pair['pairAddress']
    print(f"\nMonitoring most liquid pair: {most_liquid_pair.get('dexId')}/{most_liquid_pair.get('symbol')}")
    
    while True:
        print(f"\n{datetime.now()} - Fetching data...")
        data = get_token_pairs(token_address)
        
        if data and 'pairs' in data:
            # Find our pair in the updated data
            pair_data = next((p for p in data['pairs'] if p['pairAddress'] == pair_address), None)
            
            if pair_data:
                # Extract and display relevant information
                print(f"\n{'='*50}")
                print(f"Pair: {pair_data.get('dexId')}/{pair_data.get('symbol')}")
                print(f"Price: ${pair_data.get('priceUsd', 'N/A')}")
                
                # Token info
                base_token = pair_data.get('baseToken', {})
                quote_token = pair_data.get('quoteToken', {})
                print(f"\nToken Info:")
                print(f"Base Token:  {base_token.get('symbol')} ({base_token.get('address')})")
                print(f"Quote Token: {quote_token.get('symbol')} ({quote_token.get('address')})")
                
                # Price changes
                changes = pair_data.get('priceChange', {})
                print(f"\nPrice Changes:")
                print(f"24h: {format_price_change(changes.get('h24', 0))}")
                print(f"6h:  {format_price_change(changes.get('h6', 0))}")
                print(f"1h:  {format_price_change(changes.get('h1', 0))}")
                
                # Volume
                volume = pair_data.get('volume', {})
                print(f"\nVolume:")
                print(f"24h: ${volume.get('h24', 'N/A'):,.2f}")
                print(f"6h:  ${volume.get('h6', 'N/A'):,.2f}")
                print(f"1h:  ${volume.get('h1', 'N/A'):,.2f}")
                
                # Liquidity
                liquidity = pair_data.get('liquidity', {})
                print(f"\nLiquidity: ${liquidity.get('usd', 'N/A'):,.2f}")
                
                # Transactions
                txns = pair_data.get('txns', {}).get('h24', {})
                print(f"\nTransactions (24h):")
                print(f"Buys:  {txns.get('buys', 0)}")
                print(f"Sells: {txns.get('sells', 0)}")
            else:
                print(f"Pair data not found")
        else:
            print(f"No data received for token")
        
        print(f"\nWaiting {interval} seconds before next update...")
        time.sleep(interval)

async def get_token_price(token_address: str) -> float:
    """Get the current price of a token from its most liquid pair"""
    data = get_token_pairs(token_address)
    
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

if __name__ == "__main__":
    # Example token addresses
    EXAMPLE_TOKENS = {
        "SOL": "So11111111111111111111111111111111111111112",
        "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
        "JUP": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN"
    }
    
    print("DexScreener Token Monitor")
    print("Example tokens:")
    for name, address in EXAMPLE_TOKENS.items():
        print(f"- {name}: {address}")
    
    # Get token address from user
    token_address = "HGW9fbnC7RDNPAe5NJdyh9qvJ6EkYafMm5Bh8Hh9gu4d"
    
    try:
        monitor_token(token_address)
    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
    except Exception as e:
        print(f"Error in main loop: {str(e)}") 