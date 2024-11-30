import requests
import json
import logging
from typing import Optional, Dict, List
import aiohttp
import asyncio
import time
import backoff  

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set maximum risk score threshold
MAX_RISK_SCORE = 10000  # Less than or equal to

# Add rate limiting parameters
RATE_LIMIT_CALLS = 1000  # Number of calls allowed
RATE_LIMIT_PERIOD = 0.05  # Time period in seconds
MIN_RETRY_DELAY = 0.05  # Minimum delay between retries in seconds

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, aiohttp.ClientError),
    max_tries=3,
    max_time=30
)
def get_token_risk_report(mint_address: str) -> Optional[Dict]:
    """Get token risk report from RugCheck API with rate limiting"""
    try:
        url = f"https://api.rugcheck.xyz/v1/tokens/{mint_address}/report/summary"
        
        # Add delay between requests
        time.sleep(MIN_RETRY_DELAY)
        
        response = requests.get(url, headers={"accept": "application/json"}, timeout=10)
        
        if response.status_code == 429:  # Too Many Requests
            retry_after = int(response.headers.get('Retry-After', MIN_RETRY_DELAY))
            #logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            return get_token_risk_report(mint_address)  # Retry after waiting
            
        if response.status_code == 404:
            logger.warning(f"Token not found: {mint_address}")
            return None
            
        response.raise_for_status()
        return response.json()
        
    except Exception as e:
        logger.error(f"Error fetching data for {mint_address}: {e}")
        return None

@backoff.on_exception(
    backoff.expo,
    (aiohttp.ClientError, asyncio.TimeoutError),
    max_tries=3,
    max_time=1
)
async def get_token_risk_report_async(mint_address: str, session: Optional[aiohttp.ClientSession] = None) -> Optional[Dict]:
    """Async version of get_token_risk_report with rate limiting"""
    try:
        url = f"https://api.rugcheck.xyz/v1/tokens/{mint_address}/report/summary"
        
        should_close = False
        if session is None:
            session = aiohttp.ClientSession()
            should_close = True
        
        # Add delay between requests
        await asyncio.sleep(MIN_RETRY_DELAY)
        
        try:
            async with session.get(
                url, 
                headers={"accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 429:  # Too Many Requests
                    retry_after = int(response.headers.get('Retry-After', MIN_RETRY_DELAY))
                    logger.warning(f"Rate limited. Waiting {MIN_RETRY_DELAY} seconds...")
                    await asyncio.sleep(retry_after)
                    return await get_token_risk_report_async(mint_address, session)
                    
                if response.status == 404:
                    logger.warning(f"Token not found: {mint_address}")
                    return None
                    
                if response.status == 200:
                    return await response.json()
                    
                response.raise_for_status()
                
        finally:
            if should_close:
                await session.close()
                
    except Exception as e:
        logger.error(f"Error fetching data for {mint_address}: {e}")
        return None

async def check_multiple_tokens_async(token_addresses: List[str], batch_size: int = 5) -> Dict[str, Dict]:
    """Check multiple tokens concurrently with rate limiting and batch processing"""
    results = {}
    timeout = aiohttp.ClientTimeout(total=30)
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for i in range(0, len(token_addresses), batch_size):
                try:
                    batch = token_addresses[i:i + batch_size]
                    tasks = []
                    
                    for address in batch:
                        if tasks:  # If not the first request in batch
                            await asyncio.sleep(0.1)
                        tasks.append(get_token_risk_report_async(address, session=session))
                    
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for address, result in zip(batch, batch_results):
                        if isinstance(result, Exception):
                            logger.error(f"Error checking {address}: {result}")
                            results[address] = None
                        else:
                            results[address] = result
                    
                    await asyncio.sleep(0.5)  # Rate limiting between batches
                except Exception as e:
                    logger.error(f"Error processing batch starting at index {i}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Critical error in check_multiple_tokens_async: {e}")
    
    return results

def main():
    # List of token addresses to check
    tokens = [
        "Gg7yp9ZL4Fszk26zPmVToCvEqSXWLRR25KsgKQdFpump",  # Example token
        "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU",  # SAMO
        "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",  # BONK
    ]
    
    print(f"\nChecking tokens for risk score <= {MAX_RISK_SCORE}")
    print("="*50)
    
    low_risk_tokens = []
    high_risk_tokens = []
    
    for token_address in tokens:
        logger.info(f"\nChecking Token: {token_address}")
        
        data = get_token_risk_report(token_address)
        if not data:
            continue
            
        score = data.get('score', 0)
        risks = data.get('risks', [])
        
        # Categorize token based on score
        token_info = {
            'address': token_address,
            'score': score,
            'risks': risks,
            'program': data.get('tokenProgram', 'N/A')
        }
        
        if score <= MAX_RISK_SCORE:
            low_risk_tokens.append(token_info)
        else:
            high_risk_tokens.append(token_info)
            
        # Print immediate results
        print(f"\nToken: {token_address}")
        print(f"Score: {score}/1000")
        print(f"Status: {'✅ LOW RISK' if score <= MAX_RISK_SCORE else '❌ HIGH RISK'}")
        
        if risks:
            print("\nRisk Factors:")
            for risk in risks:
                print(f"- {risk['name']} ({risk['level']}): {risk['description']}")
        else:
            print("\nNo risks found")
            
        print("-"*50)
    
    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Total tokens checked: {len(tokens)}")
    print(f"Low risk tokens (score <= {MAX_RISK_SCORE}): {len(low_risk_tokens)}")
    print(f"High risk tokens (score > {MAX_RISK_SCORE}): {len(high_risk_tokens)}")
    
    if low_risk_tokens:
        print("\nLow Risk Tokens:")
        for token in low_risk_tokens:
            print(f"✅ {token['address']} (Score: {token['score']})")
    
    if high_risk_tokens:
        print("\nHigh Risk Tokens:")
        for token in high_risk_tokens:
            print(f"❌ {token['address']} (Score: {token['score']})")

if __name__ == "__main__":
    main()
