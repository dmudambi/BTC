import requests
import json
import logging
from typing import Optional, Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set maximum risk score threshold
MAX_RISK_SCORE = 500  # Less than or equal to

def get_token_risk_report(mint_address: str) -> Optional[Dict]:
    """Get token risk report from RugCheck API"""
    try:
        url = f"https://api.rugcheck.xyz/v1/tokens/{mint_address}/report/summary"
        response = requests.get(url, headers={"accept": "application/json"}, timeout=10)
        
        if response.status_code == 404:
            logger.warning(f"Token not found: {mint_address}")
            return None
            
        response.raise_for_status()
        return response.json()
        
    except Exception as e:
        logger.error(f"Error fetching data for {mint_address}: {e}")
        return None

def main():
    # List of token addresses to check
    tokens = [
        "Gg7yp9ZL4Fszk26zPmVToCvEqSXWLRR25KsgKQdFpump",  # Example token
        "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU",  # SAMO
        "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",  # BONK
        # Add more tokens here
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
