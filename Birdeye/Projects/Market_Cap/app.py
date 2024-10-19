import streamlit as st
import pandas as pd
from Market_Cap import get_token_list  # Ensure correct import path
import dontshare as d  # Ensure this module contains your API keys and other secrets
import logging

# Configure logging to display in Streamlit
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    st.title("Filtered Tokens Dashboard")

    st.sidebar.header("Filter Options")

    # Adjustable Inputs
    chain = st.sidebar.selectbox(
        "Chain",
        options=["solana", "ethereum", "arbitrum", "avalanche", "bsc", "optimism", "polygon", "base", "zksync"],
        help="Select the blockchain to query."
    )
    
    sort_by = st.sidebar.selectbox(
        "Sort By",
        options=["mc", "v24hUSD", "v24hChangePercent"],
        help="Choose the criterion to sort the tokens."
    )

    sort_type = st.sidebar.selectbox(
        "Sort Type",
        options=["desc", "asc"],
        help="Choose the sort order."
    )

    min_market_cap = st.sidebar.number_input(
        "Minimum Market Cap",
        min_value=300000,
        value=1000000,
        step=100000,
        help="Set the minimum market capitalization in USD. Minimum floored to 300k to avoid gutter."
    )

    max_market_cap = st.sidebar.number_input(
        "Maximum Market Cap",
        min_value=0,
        value=1000000000,
        step=10000000,
        help="Set the maximum market capitalization in USD."
    )

    min_volume_24h = st.sidebar.number_input(
        "Minimum Volume 24h",
        min_value=0,
        value=1000000,
        step=1000000,
        help="Set the minimum 24-hour trading volume in USD."
    )

    min_liquidity = st.sidebar.number_input(
        "Minimum Liquidity",
        min_value=0,
        value=100000,
        step=10000,
        help="Set the minimum liquidity in USD."
    )

    total_tokens = st.sidebar.number_input(
        "Total Tokens",
        min_value=100,
        max_value=5000,  # Adjusted to prevent excessive requests
        value=1000,
        step=200,
        help="Number of tokens to filter pre-filtering (limit to 1000 to avoid excessive requests)"
    )

    if st.sidebar.button("Fetch Filtered Tokens"):
        with st.spinner("Fetching tokens..."):
            filtered_tokens, errors = get_token_list(
                sort_by=sort_by,
                sort_type=sort_type,
                min_liquidity=min_liquidity,
                min_volume_24h=min_volume_24h,
                min_market_cap=min_market_cap,
                max_market_cap=max_market_cap,
                total_tokens=total_tokens,
                chain=chain,
                API_Key=d.birdeye
            )

        if not filtered_tokens.empty:
            st.success("Tokens fetched successfully!")
            st.dataframe(filtered_tokens)
            st.markdown(f"**Total Tokens Displayed:** {len(filtered_tokens)}")
        else:
            st.warning("No tokens found with the specified criteria.")

        if errors:
            st.error("Some errors occurred during fetching:")
            for error in errors:
                st.write(f"- {error}")

if __name__ == "__main__":
        main()