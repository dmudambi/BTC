# Import required libraries
import streamlit as st
import sys
import os
from datetime import datetime
import time

# Configure the Streamlit page
st.set_page_config(
    page_title="Fibonacci Retracement Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add the current directory to system path to import local modules
current_dir = os.getcwd()
sys.path.append(current_dir)

# Import custom modules
from Fib_Dashboard import (
    plot_price_and_fib_levels,
    Master_Functions
)

def main():
    """
    Main function to run the Streamlit dashboard
    """
    # Set up the main title
    st.title("🎯 Fibonacci Retracement Dashboard")
    
    # Add initial status message
    status_container = st.empty()
    status_container.info("Dashboard initializing... This may take 6-8 minutes to load all data and generate plots.")
    
    # Create progress bar
    progress_bar = st.progress(0)
    
    # Create sidebar controls
    st.sidebar.header("Controls")
    
    # Timeframe selection dropdown
    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        options=['5m', '15m', '1H', '4H'],
        index=0
    )
    
    # Data refresh button
    if st.sidebar.button("Refresh Data"):
        status_container.warning("Starting data collection process...")
        with st.spinner("Collecting fresh data..."):
            os.system('python Birdeye/Projects/Indicators/Fib/Dataset_Generator_New_Tokens.py')
        st.success("Data refreshed!")

    try:
        # Update progress
        status_container.info("Setting up data paths...")
        progress_bar.progress(10)
        
        # Set up paths for data
        base_path = 'Data/New_Token_Data'
        current_date = datetime.now().strftime('%Y_%m_%d')
        date_folder = os.path.join(base_path, current_date)
        ohlcv_folder = os.path.join(date_folder, 'OHLCV_Data')
        
        # Get the most recent data folder
        status_container.info("Locating most recent data folder...")
        progress_bar.progress(20)
        ohlcv_datetime_folder = Master_Functions.get_most_recent_folder(ohlcv_folder)
        
        # Load OHLCV data
        status_container.info("Loading OHLCV data... This may take a few minutes...")
        progress_bar.progress(30)
        
        start_time = time.time()
        imported_ohlcv_data = Master_Functions.import_ohlcv_data(ohlcv_datetime_folder)
        load_time = time.time() - start_time
        
        status_container.info(f"OHLCV data loaded successfully in {load_time:.2f} seconds")
        progress_bar.progress(50)
        
        # Define Fibonacci levels
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        # Generate plots
        status_container.info("Generating Fibonacci plots... This may take a few minutes...")
        progress_bar.progress(60)
        
        plot_start_time = time.time()
        plot_data = plot_price_and_fib_levels(
            imported_ohlcv_data, 
            fib_levels, 
            initial_timeframe=timeframe
        )
        plot_time = time.time() - plot_start_time
        
        status_container.info(f"Plots generated successfully in {plot_time:.2f} seconds")
        progress_bar.progress(80)
        
        # Display each figure in the dashboard
        status_container.info("Rendering plots...")
        for i, (fig, token_address) in enumerate(plot_data, 1):
            # Create a container for each plot and its link
            plot_container = st.container()
            
            # Add the Dexscreener link with an icon
            dexscreener_link = f"https://dexscreener.com/solana/{token_address}"
            
            # Create a centered link with icon and styling
            plot_container.markdown(
                f"""
                <div style="text-align: center; padding: 10px;">
                    <a href="{dexscreener_link}" target="_blank" 
                       style="text-decoration: none; 
                              background-color: #1e1e1e; 
                              color: #00cc00; 
                              padding: 8px 16px; 
                              border: 1px solid #00cc00; 
                              border-radius: 5px; 
                              font-weight: bold;">
                        🔍 View on Dexscreener
                    </a>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Display the plot
            plot_container.plotly_chart(fig, use_container_width=True)
            
            # Update progress
            progress_value = 80 + (i / len(plot_data)) * 20
            progress_bar.progress(int(progress_value))
        
        # Final success message
        total_time = time.time() - start_time
        status_container.success(f"""
        ✅ Dashboard loaded successfully!
        - Total time: {total_time:.2f} seconds
        - Data loading time: {load_time:.2f} seconds
        - Plot generation time: {plot_time:.2f} seconds
        - Number of plots: {len(plot_data)}
        """)
        progress_bar.progress(100)
            
    except Exception as e:
        # Error handling with detailed message
        error_message = f"""
        ❌ An error occurred: {str(e)}
        
        Troubleshooting steps:
        1. Make sure you have run the data collection script first
        2. Check that all required data folders exist
        3. Verify that the data files are not corrupted
        
        Current paths being checked:
        - Base path: {base_path}
        - Date folder: {current_date}
        - OHLCV folder: {ohlcv_folder}
        
        If the error persists, try clicking the 'Refresh Data' button in the sidebar.
        """
        status_container.error(error_message)
        progress_bar.empty()

# Run the main function when the script is executed
if __name__ == "__main__":
    main()