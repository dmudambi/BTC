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
RATE_LIMIT_CALLS = 10000  # Number of calls allowed
RATE_LIMIT_PERIOD = 0.05  # Time period in seconds
MIN_RETRY_DELAY = 0.05  # Minimum delay between retries in seconds

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, aiohttp.ClientError),
    max_tries=5,
    max_time=60
)
def get_token_risk_report(mint_address: str) -> Optional[Dict]:
    """Get token risk report from RugCheck API with rate limiting"""
    try:
        url = f"https://api.rugcheck.xyz/v1/tokens/{mint_address}/report/summary"
        
        # Add delay between requests
        time.sleep(0.2)
        
        response = requests.get(url, headers={"accept": "application/json"}, timeout=30)
        
        if response.status_code == 429:  # Too Many Requests
            retry_after = int(response.headers.get('Retry-After', 1))
            logger.info(f"Rate limited for {mint_address}, retrying in {retry_after} seconds")
            time.sleep(retry_after)
            return get_token_risk_report(mint_address)  # Retry after waiting
            
        if response.status_code == 404:
            logger.info(f"Token not found: {mint_address}")
            return None
            
        response.raise_for_status()
        data = response.json()
        score = data.get('score', 'N/A')
        logger.info(f"Token: {mint_address}, Score: {score}")
        return data
        
    except Exception as e:
        logger.info(f"Error fetching data for {mint_address}: {e}")
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
        
        try:
            async with session.get(url) as response:
                if response.status == 429:  # Rate limit
                    retry_after = int(response.headers.get('Retry-After', MIN_RETRY_DELAY))
                    logger.info(f"Rate limited for {mint_address}, retrying in {retry_after} seconds")
                    await asyncio.sleep(retry_after)
                    # Retry the request after waiting
                    async with session.get(url) as retry_response:
                        if retry_response.status == 200:
                            data = await retry_response.json()
                            score = data.get('score', 'N/A')
                            logger.info(f"Token: {mint_address}, Score: {score}")
                            return data
                    
                if response.status == 404:
                    logger.info(f"Token not found in rugcheck: {mint_address}")
                    return None
                    
                if response.status == 200:
                    data = await response.json()
                    if 'score' not in data:
                        logger.info(f"No score in rugcheck response for {mint_address}")
                    score = data.get('score', 'N/A')
                    logger.info(f"Token: {mint_address}, Score: {score}")
                    return data
                    
                response.raise_for_status()
                
        finally:
            if should_close:
                await session.close()
                
    except Exception as e:
        logger.info(f"Error fetching rugcheck data for {mint_address}: {e}")
        return None

async def check_multiple_tokens_async(token_addresses: List[str], batch_size: int = 2, max_retries: int = 5) -> Dict[str, Dict]:
    """Check multiple tokens concurrently with robust retry mechanism"""
    results = {}
    timeout = aiohttp.ClientTimeout(total=60)  # Increased timeout
    
    async def fetch_with_retry(address: str, session: aiohttp.ClientSession, attempt: int = 1) -> Dict:
        try:
            if attempt > 1:
                # Longer exponential backoff: 5s, 10s, 20s, 40s, 80s between retries
                await asyncio.sleep(5 * (2 ** (attempt - 1)))
            result = await get_token_risk_report_async(address, session=session)
            if result and 'score' in result:
                # Remove duplicate logging - only log once here
                logger.info(f"Token: {address}, Score: {result['score']}")
                return result
            raise Exception("Invalid response format")
        except Exception as e:
            if attempt < max_retries:
                logger.info(f"Retry {attempt}/{max_retries} for {address}")
                return await fetch_with_retry(address, session, attempt + 1)
            # On final failure, make one last attempt with maximum delay
            await asyncio.sleep(90)  # 90 second cooldown
            try:
                result = await get_token_risk_report_async(address, session=session)
                if result and 'score' in result:
                    # Remove duplicate logging - only log once here
                    logger.info(f"Token: {address}, Score: {result['score']}")
                    return result
            except Exception as final_e:
                logger.warning(f"Final attempt failed for {address}: {str(final_e)}")
            return None
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Process tokens in very small batches
            for i in range(0, len(token_addresses), batch_size):
                batch = token_addresses[i:i + batch_size]
                tasks = []
                
                for address in batch:
                    tasks.append(fetch_with_retry(address, session))
                    await asyncio.sleep(1)  # 1s between requests within batch
                
                batch_results = await asyncio.gather(*tasks)
                
                for address, result in zip(batch, batch_results):
                    if result and 'score' in result:
                        results[address] = result
                    else:
                        # For failed tokens, try one final time sequentially
                        await asyncio.sleep(30)  # 30s cooldown
                        try:
                            final_result = await get_token_risk_report_async(address, session=session)
                            if final_result and 'score' in final_result:
                                results[address] = final_result
                                logger.info(f"Token: {address}, Score: {final_result['score']} (final attempt)")
                            else:
                                results[address] = {'score': 99999}  # Use high score instead of infinity
                                logger.info(f"Token: {address}, Score: 99999 (failed all attempts)")
                        except Exception:
                            results[address] = {'score': 99999}
                            logger.info(f"Token: {address}, Score: 99999 (failed all attempts)")
                
                # Longer delay between batches
                await asyncio.sleep(3)  # 3s between batches
    
    except Exception as e:
        logger.error(f"Critical error in check_multiple_tokens_async: {e}")
    
    # Verify coverage and add any missing tokens
    missing_tokens = set(token_addresses) - set(results.keys())
    if missing_tokens:
        logger.warning(f"Missing rugcheck scores for {len(missing_tokens)} tokens")
        for address in missing_tokens:
            results[address] = {'score': 99999}
            logger.info(f"Token: {address}, Score: 99999 (missing)")
    
    # Log summary
    total_tokens = len(token_addresses)
    successful_checks = sum(1 for r in results.values() if r.get('score', 99999) != 99999)
    logger.info(f"\nRugcheck Summary:")
    logger.info(f"Total tokens checked: {total_tokens}")
    logger.info(f"Successful checks: {successful_checks} ({(successful_checks/total_tokens)*100:.1f}%)")
    logger.info(f"Failed checks: {total_tokens - successful_checks} ({((total_tokens-successful_checks)/total_tokens)*100:.1f}%)")
    
    return results

def main():
    # List of token addresses to check
    tokens = [
        "AMQ4GpGVopaymh1fQDmN9y37Yt5XME7QcqU9vQNgdqpZ",  # Example token
        "8GASDGJX94qJHRgkzLak7Xttad7UFriRrgMY1e5FjJaM",  # SAMO
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
