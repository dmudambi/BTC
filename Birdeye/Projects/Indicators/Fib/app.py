"""
Fibonacci Retracement Dashboard
------------------------------
A real-time dashboard for Fibonacci retracement analysis of new tokens.
Features parallel processing with continuous data generation and independent display refresh.

Architecture:
1. Background Thread: Continuous data generation
2. Main Thread: Dashboard display and refresh
3. Independent refresh cycles
4. Real-time plot updates
"""

# Import required libraries
import streamlit as st
import sys
import os
from datetime import datetime, timezone
import time
import pytz
import subprocess
import threading

# Configure the Streamlit page - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Fibonacci Retracement Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Get absolute path to the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_GENERATOR_PATH = os.path.join(SCRIPT_DIR, 'Dataset_Generator_New_Tokens.py')

# Add the current directory to system path to import local modules
current_dir = os.getcwd()
sys.path.append(current_dir)

# Import custom modules
from Fib_Dashboard import (
    plot_price_and_fib_levels,
    ATH_Breakout_Master
)

def get_formatted_times():
    """
    Get current time in EST and IST timezones
    
    Returns:
        tuple: (est_time, ist_time) containing timezone-aware datetime objects
    """
    current_time_utc = datetime.now(timezone.utc)
    est_time = current_time_utc.astimezone(pytz.timezone('US/Eastern'))
    ist_time = current_time_utc.astimezone(pytz.timezone('Asia/Kolkata'))
    return est_time, ist_time

def get_time_until_refresh(start_time, refresh_interval=700):
    """
    Calculate time remaining until next refresh
    
    Args:
        start_time (float): Timestamp when the refresh started
        refresh_interval (int): Refresh interval in seconds (default: 700)
    
    Returns:
        str: Formatted string showing minutes and seconds remaining
    """
    elapsed = time.time() - start_time
    remaining = refresh_interval - elapsed
    minutes = int(remaining // 60)
    seconds = int(remaining % 60)
    return f"{minutes:02d}:{seconds:02d}"

def continuous_data_generation():
    """
    Continuously generate new datasets in background thread
    
    This function runs independently of the dashboard refresh cycle:
    1. Runs Dataset Generator script
    2. Logs timing information
    3. Handles errors gracefully
    4. Continues running indefinitely
    """
    while True:
        try:
            # Get and display current times
            est_time, ist_time = get_formatted_times()
            print("\n" + "="*50)
            print(f"Starting new dataset generation at:")
            print(f"EST: {est_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"IST: {ist_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print("="*50 + "\n")
            
            # Time the dataset generation
            dataset_start_time = time.time()
            
            # Run Dataset Generator as subprocess with output streaming
            process = subprocess.Popen(
                ['python', DATASET_GENERATOR_PATH],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream the output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            # Wait for process completion
            process.communicate()
            
            # Log completion time
            dataset_time = time.time() - dataset_start_time
            print("\n" + "="*50)
            print(f"Dataset generation completed in: {dataset_time:.2f} seconds")
            print("="*50 + "\n")

            # Check process success
            if process.returncode != 0:
                raise Exception(f"Dataset Generator failed with return code {process.returncode}")
                
            # Wait before starting next generation
            time.sleep(10)  # Short pause between generations
                
        except Exception as e:
            print(f"Error in data generation: {str(e)}")
            time.sleep(60)  # Wait before retry after error

def display_dashboard():
    """
    Handle dashboard display and refresh
    
    This function:
    1. Displays the most recent data
    2. Updates plots on refresh interval
    3. Maintains dashboard state
    4. Handles errors gracefully
    """
    # Set up the main title
    st.title("🎯 Fibonacci Retracement Dashboard")
    
    # Create containers for different dashboard elements
    status_container = st.empty()     # For status messages
    progress_bar = st.progress(0)     # For progress tracking
    refresh_status = st.empty()       # For refresh countdown

    # Create a loop that runs indefinitely
    while True:
        try:
            refresh_start_time = time.time()
            
            # Update initial status
            status_container.info("Loading latest data and generating plots...")
            progress_bar.progress(10)
            
            # Set up paths for data
            base_path = 'Data/New_Token_Data'
            current_date = datetime.now().strftime('%Y_%m_%d')
            date_folder = os.path.join(base_path, current_date)
            ohlcv_folder = os.path.join(date_folder, 'OHLCV_Data')
            
            # Get the most recent folder using Master_Functions
            status_container.info("Locating most recent data folder...")
            progress_bar.progress(20)
            ohlcv_datetime_folder = ATH_Breakout_Master.get_most_recent_folder(ohlcv_folder)
            
            if not ohlcv_datetime_folder:
                status_container.warning("No data folders found. Waiting for data...")
                time.sleep(10)
                continue
                
            # Print the folder being used for import
            print(f"Importing OHLCV data from folder: {ohlcv_datetime_folder}")
            
            # Load OHLCV data
            status_container.info("Loading OHLCV data...")
            progress_bar.progress(30)
            
            try:
                imported_ohlcv_data = ATH_Breakout_Master.import_ohlcv_data(ohlcv_datetime_folder)
                if not imported_ohlcv_data or len(imported_ohlcv_data) == 0:
                    raise Exception("No valid OHLCV data found")
                    
            except Exception as e:
                status_container.warning(
                    f"Error loading data: {str(e)}. Waiting for new data...")
                time.sleep(10)
                continue
            
            status_container.info(f"OHLCV data loaded successfully")
            progress_bar.progress(50)
            
            # Define Fibonacci levels
            fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            
            # Generate plots with validation
            status_container.info("Generating Fibonacci plots...")
            progress_bar.progress(60)
            
            try:
                plot_data = plot_price_and_fib_levels(
                    imported_ohlcv_data, 
                    fib_levels, 
                    initial_timeframe='5m'
                )
                
                if not plot_data or len(plot_data) == 0:
                    raise Exception("No plots generated")
                    
            except Exception as e:
                status_container.warning(
                    f"Error generating plots: {str(e)}. Retrying...")
                time.sleep(10)
                continue
            
            # Calculate processing time
            total_processing_time = time.time() - refresh_start_time
            print(f"Data processing and plot generation completed in: {total_processing_time:.2f} seconds")
            
            status_container.info("Plots generated successfully")
            progress_bar.progress(80)
            
            # Clear existing plots from session state
            for key in list(st.session_state.keys()):
                if key.startswith('plot_'):
                    del st.session_state[key]
            
            # Display each figure in the dashboard
            status_container.info("Rendering plots...")
            for i, (fig, token_address) in enumerate(plot_data, 1):
                # Store plot data in session state
                plot_key = f'plot_{i}'
                st.session_state[plot_key] = {
                    'figure': fig,
                    'token_address': token_address
                }
                
                # Create plot container and Dexscreener link
                plot_container = st.container()
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
            
            # Get current time in EST and IST for final display
            est_time, ist_time = get_formatted_times()

            # Display success message with timezone information
            status_container.success(
                "✅ Dashboard refreshed successfully!\n\n"
                f"Data Period (EST): {est_time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n"
                f"Data Period (IST): {ist_time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n"
                f"Number of plots: {len(plot_data)}"
            )
            progress_bar.progress(100)
            
            # Wait for refresh interval while showing countdown
            while (time.time() - refresh_start_time) < 700:  # 5 minutes
                time_left = get_time_until_refresh(refresh_start_time)
                refresh_status.info(f"Next refresh in: {time_left}")
                time.sleep(1)
                
        except Exception as e:
            # Detailed error handling with troubleshooting steps
            error_message = f"""
            ❌ An error occurred: {str(e)}
            
            Troubleshooting steps:
            1. Check that all required data folders exist
            2. Verify that the data files are not corrupted
            3. Check network connectivity
            
            Current paths being checked:
            - Base path: Data/New_Token_Data
            - OHLCV folder: {ohlcv_datetime_folder if 'ohlcv_datetime_folder' in locals() else 'N/A'}
            
            Retrying in 10 seconds...
            """
            status_container.error(error_message)
            progress_bar.empty()
            time.sleep(10)
            continue

def main():
    """
    Main function to start both data generation and dashboard threads
    
    This function:
    1. Starts continuous data generation in background thread
    2. Runs dashboard display in main thread
    3. Handles graceful shutdown
    """
    # Start the continuous data generation in a separate thread
    data_thread = threading.Thread(target=continuous_data_generation, daemon=True)
    data_thread.start()
    
    # Run the dashboard in the main thread
    display_dashboard()

# Entry point
if __name__ == "__main__":
    main()