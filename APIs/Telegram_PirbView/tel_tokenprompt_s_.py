from telethon import TelegramClient
from telethon.sessions import StringSession
from datetime import datetime
import pandas as pd
from pathlib import Path
import os
import re
from dotenv import load_dotenv
import asyncio
import logging

class PirbViewBot:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize configurations
        self.api_id = os.getenv("TELEGRAM_API_ID")
        self.api_hash = os.getenv("TELEGRAM_API_HASH")
        self.bot_username = "PirbViewBot"
        self.session_file = 'session.txt'
        self.client = None
        
        # Setup logging
        self.setup_logging()
        
        # Setup data directory
        self.data_path = Path("data")
        self.data_path.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """Configure logging"""
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("PirbViewBot")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_dir / "pirb_view_bot.log")
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    async def initialize_client(self):
        """Initialize Telegram client"""
        try:
            # Try to load existing session
            try:
                with open(self.session_file, 'r') as f:
                    session_string = f.read()
                self.client = TelegramClient(StringSession(session_string), 
                                          self.api_id, 
                                          self.api_hash)
            except FileNotFoundError:
                self.logger.info("No saved session found, using new session")
                self.client = TelegramClient(StringSession(), 
                                          self.api_id, 
                                          self.api_hash)
            
            await self.client.start()
            return True

        except Exception as e:
            self.logger.error(f"Error initializing client: {e}")
            return False

    async def get_token_info(self, token_address: str):
        """Send command to bot and get token information"""
        try:
            print("\nStarting token info retrieval...")
            command = f"/s {token_address}"
            self.logger.info(f"Sending command: {command}")
            
            async with self.client.conversation(self.bot_username) as conv:
                # Send the command
                await conv.send_message(command)
                print("Command sent, waiting for initial response...")
                
                # Wait for initial response
                response = await conv.get_response(timeout=30)
                print(f"Initial response received with {len(response.buttons) if response.buttons else 0} buttons")
                
                if not response.buttons:
                    self.logger.error("No buttons found in response")
                    return None
                
                try:
                    # Print button information
                    for row_idx, row in enumerate(response.buttons):
                        for btn_idx, btn in enumerate(row):
                            print(f"Button [{row_idx},{btn_idx}]: {btn.text}")
                    
                    # Click the bubble map button
                    if len(response.buttons) > 0 and len(response.buttons[0]) > 2:
                        print("Clicking bubble map button...")
                        await response.click(0, 2)
                        
                        print("Waiting for generating message...")
                        # Wait for and collect all messages until we get the cluster data
                        messages = []
                        cluster_data = None
                        start_time = datetime.now()
                        timeout = 180  # 3 minutes timeout
                        
                        while (datetime.now() - start_time).seconds < timeout:
                            try:
                                msg = await conv.get_response(timeout=10)
                                print(f"Received message: {msg.text[:100]}...")
                                messages.append(msg)
                                
                                # Check if this message contains cluster data
                                if 'Aggregate Summary:' in msg.text:
                                    cluster_data = msg
                                    print("Found cluster data message!")
                                    break
                                
                            except asyncio.TimeoutError:
                                continue
                        
                        if not cluster_data:
                            print("Failed to get cluster data within timeout")
                            return None
                        
                        print("\nProcessing cluster data...")
                        data = self.parse_bubble_response(cluster_data.text)
                        
                        if data:
                            data['token_address'] = token_address
                            print("Successfully parsed bubble map data")
                            return data
                        else:
                            print("Failed to parse bubble map data")
                            return None
                    
                    else:
                        print("Bubble map button not found")
                        return None
                    
                except Exception as e:
                    print(f"Error during button interaction: {str(e)}")
                    self.logger.exception("Full traceback:")
                    return None
            
        except Exception as e:
            print(f"Error in get_token_info: {str(e)}")
            self.logger.exception("Full traceback:")
            return None

    def parse_bubble_response(self, response_text: str):
        """Parse the bubble map response"""
        print("\nParsing bubble response...")
        print("Raw text received:")
        print(response_text)
        
        data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'token_name': None,
            'token_address': None,
            'top_10_clusters_total': None,
            'clusters': []
        }
        
        try:
            lines = [line.strip() for line in response_text.split('\n') if line.strip()]
            print(f"\nProcessing {len(lines)} non-empty lines")
            
            for i, line in enumerate(lines):
                print(f"\nProcessing line {i}: {line}")
                
                # Get token name - updated pattern to match the actual format
                if '**BubbleMap for' in line:
                    token_match = re.search(r'\*\*BubbleMap for \*\*\[\*\*([^\]]+)', line)
                    if token_match:
                        data['token_name'] = token_match.group(1)
                        print(f"Found token name: {data['token_name']}")
                    else:
                        print(f"Failed to extract token name from line: {line}")
                
                # Rest of the parsing logic remains the same
                elif 'Top 10 clusters:' in line:
                    percentage = re.search(r'(\d+\.\d+)%', line)
                    if percentage:
                        data['top_10_clusters_total'] = float(percentage.group(1))
                        print(f"Found top 10 clusters: {data['top_10_clusters_total']}%")
                
                elif 'Cluster' in line and '(' in line:
                    cluster_match = re.search(r'Cluster (\d+)\s*\((\d+\.\d+)%\)', line)
                    if cluster_match:
                        cluster_num = int(cluster_match.group(1))
                        cluster_percentage = float(cluster_match.group(2))
                        
                        cluster_data = {
                            'cluster_number': cluster_num,
                            'cluster_total_percentage': cluster_percentage,
                            'wallets': []
                        }
                        
                        data['clusters'].append(cluster_data)
                        print(f"Added cluster {cluster_num}")
                
                # Get wallet information
                elif '➡️' in line:
                    if data['clusters']:  # Make sure we have a current cluster
                        current_cluster = data['clusters'][-1]
                        wallet_matches = re.finditer(r'\[([^\]]+)\]\(https://solscan\.io/account/([^\)]+)\)\s*\((\d+\.\d+)%\)', line)
                        
                        for wallet_match in wallet_matches:
                            wallet_data = {
                                'address': wallet_match.group(2),  # Use full address
                                'percentage': float(wallet_match.group(3))
                            }
                            current_cluster['wallets'].append(wallet_data)
                            print(f"Found wallet: {wallet_data}")
            
            if not data['clusters']:
                print("No clusters were parsed from the response")
                return None
            
            print("\nParsing completed successfully")
            return data

        except Exception as e:
            print(f"Error parsing bubble response: {str(e)}")
            self.logger.exception("Full traceback:")
            return None

    def get_organized_filepath(self, data: dict):
        """Create organized file path based on day/token_name_address/time"""
        # Get current date and time
        now = datetime.now()
        day = now.strftime('%Y_%m_%d')  # Format: 2024_03_11
        time_folder = now.strftime('%Y_%b_%d_%I%M%p')  # Format: 2024_Mar_11_0504AM
        time = now.strftime('%H%M%S')
        
        # Create token directory name combining name and address
        token_name = data.get('token_name', 'UNKNOWN')
        token_address = data.get('token_address', 'UNKNOWN')
        token_dir_name = f"{token_name}_{token_address}"
        
        # Create directory structure
        day_dir = self.data_path / day
        token_dir = day_dir / token_dir_name
        time_dir = token_dir / time_folder
        
        # Create directories if they don't exist
        time_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with human-readable time
        filename = f"bubble_map_{time}.csv"
        return time_dir / filename

    def save_to_csv(self, data: dict):
        """Save bubble map data to CSV file with organized structure"""
        try:
            if not data:
                self.logger.error("No data provided to save_to_csv")
                return
            
            if not data.get('token_name'):
                self.logger.error(f"No token name found in data. Available data: {data}")
                return
            
            # Get organized file path
            filepath = self.get_organized_filepath(data)
            
            # Create a list of rows for the DataFrame
            rows = []
            for cluster in data['clusters']:
                for wallet in cluster['wallets']:
                    row = {
                        'timestamp': data['timestamp'],
                        'token_name': data['token_name'],
                        'token_address': data.get('token_address', 'UNKNOWN'),
                        'top_10_clusters_total': data.get('top_10_clusters_total'),
                        'cluster_number': cluster['cluster_number'],
                        'cluster_total_percentage': cluster['cluster_total_percentage'],
                        'wallet_address': wallet['address'],
                        'wallet_percentage': wallet['percentage']
                    }
                    rows.append(row)
            
            self.logger.info(f"Created {len(rows)} rows for CSV")
            
            if not rows:
                self.logger.error("No rows created for CSV file")
                return
            
            df = pd.DataFrame(rows)
            
            # Save to CSV
            df.to_csv(filepath, index=False)
            self.logger.info(f"Bubble map data saved to {filepath} with {len(df)} rows")
            
            # Also save to a daily summary file
            self.update_daily_summary(df, data, filepath.parent)
            
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            self.logger.exception("Full traceback:")

    def update_daily_summary(self, new_df: pd.DataFrame, data: dict, time_dir: Path):
        """Update the daily summary file for the token"""
        try:
            # Create summary filename inside time directory
            summary_file = time_dir / "daily_summary.csv"
            
            # If summary file exists, append to it, otherwise create new
            if summary_file.exists():
                existing_df = pd.read_csv(summary_file)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                # Remove duplicates based on all columns except timestamp
                cols_for_dedup = [col for col in combined_df.columns if col != 'timestamp']
                combined_df = combined_df.drop_duplicates(subset=cols_for_dedup, keep='last')
            else:
                combined_df = new_df
            
            combined_df.to_csv(summary_file, index=False)
            self.logger.info(f"Daily summary updated at {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Error updating daily summary: {e}")

async def main():
    print("\nInitializing PirbViewBot...")
    bot = PirbViewBot()
    
    # Initialize client
    if not await bot.initialize_client():
        print("Failed to initialize client")
        return

    token_address = "61trovNRXxwEyDraZnyZA1U1XW5anmug5DjjDLR7pump"
    max_retries = 5
    retry_delay = 10  # seconds between retries
    
    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1} of {max_retries}")
            print(f"Requesting data for token: {token_address}")
            
            # Get token information
            token_data = await bot.get_token_info(token_address)
            
            if token_data:
                print("\nSaving data to CSV...")
                bot.save_to_csv(token_data)
                print("Token data retrieved and saved successfully")
                break  # Exit the retry loop if successful
            else:
                if attempt < max_retries - 1:  # Don't wait after the last attempt
                    print(f"Failed to get token data. Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    print("Failed to get token data after all retry attempts")
        
        except Exception as e:
            print(f"\nError in attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                print("Failed after all retry attempts")
    
    try:
        await bot.client.disconnect()
    except:
        pass

if __name__ == "__main__":
    asyncio.run(main())
