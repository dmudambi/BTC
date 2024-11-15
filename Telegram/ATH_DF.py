"""
Telegram Message Monitor and Parser with CSV Saving
This script continuously monitors a Telegram channel for new messages,
parses them, and saves unique entries to a human-readable CSV file.

Required Environment Variables:
    TELEGRAM_API_ID: Your Telegram API ID
    TELEGRAM_API_HASH: Your Telegram API Hash
    PHONE_NUM: Your phone number for Telegram authentication
"""

from telethon import TelegramClient
import pandas as pd
import json 
from pathlib import Path
import os
from dotenv import load_dotenv
import asyncio
from asyncio import TimeoutError
import re
from datetime import datetime
from telethon.errors import ChatAdminRequiredError, ChannelPrivateError
from telethon.sessions import MemorySession, StringSession
from telethon.tl.functions.channels import GetForumTopicsRequest

# Load environment variables
print(load_dotenv())
api_id = os.getenv("TELEGRAM_API_ID")
api_hash = os.getenv("TELEGRAM_API_HASH")
phone_num = os.getenv("PHONE_NUM")

def format_dataframe_for_csv(df):
    """
    Formats DataFrame columns to be more human-readable and handles empty values.
    """
    # Create a copy to avoid modifying the original
    formatted_df = df.copy()
    
    # Rename columns to more readable format
    column_mapping = {
        'name': 'Token Name',
        'token': 'Token Address',
        '5m_change': '5min Change',
        '1h_change': '1hr Change',
        '6h_change': '6hr Change',
        '5m_transactions': '5min Transactions',
        '5m_volume': '5min Volume',
        'mcp': 'Market Cap',
        'liquidity_sol': 'Liquidity (SOL)',
        'holders': 'Total Holders',
        'open_time': 'Time Since Open',
        'top10_holders': 'Top 10 Holders %',
        'dev_hold_%': 'Dev Holdings',
        'dev_burnt': 'Dev Tokens Burnt',
        'telegram': 'Telegram Link',
        'timestamp': 'Timestamp'
    }
    
    formatted_df = formatted_df.rename(columns=column_mapping)
    
    # Handle empty/None values with appropriate defaults
    default_values = {
        '5min Change': '0%',
        '1hr Change': '0%',
        '6hr Change': '0%',
        '5min Transactions': '0',
        '5min Volume': '$0',
        'Market Cap': '$0',
        'Liquidity (SOL)': '0 SOL',
        'Total Holders': '0',
        'Time Since Open': '0min',
        'Top 10 Holders %': '0%',
        'Dev Holdings': '0%',
        'Dev Tokens Burnt': '0%',
        'Token Name': 'Unknown',
        'Token Address': 'None'
    }
    
    # Apply default values for empty/None cells
    for col, default in default_values.items():
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].fillna(default)
            formatted_df[col] = formatted_df[col].replace('', default)
            formatted_df[col] = formatted_df[col].replace('None', default)
    
    # Format percentage columns
    percentage_columns = ['5min Change', '1hr Change', '6hr Change', 'Top 10 Holders %', 'Dev Holdings', 'Dev Tokens Burnt']
    for col in percentage_columns:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x}%" if not str(x).endswith('%') else x)
    
    # Format currency columns
    if '5min Volume' in formatted_df.columns:
        formatted_df['5min Volume'] = formatted_df['5min Volume'].apply(lambda x: f"${x}" if not str(x).startswith('$') else x)
    
    if 'Market Cap' in formatted_df.columns:
        formatted_df['Market Cap'] = formatted_df['Market Cap'].apply(lambda x: f"${x}" if not str(x).startswith('$') else x)
    
    # Format liquidity
    if 'Liquidity (SOL)' in formatted_df.columns:
        formatted_df['Liquidity (SOL)'] = formatted_df['Liquidity (SOL)'].apply(lambda x: f"{x} SOL" if not str(x).endswith('SOL') else x)
    
    # Format timestamp
    if 'Timestamp' in formatted_df.columns:
        formatted_df['Timestamp'] = pd.to_datetime(formatted_df['Timestamp']).dt.strftime('%Y-%m-%d %I:%M:%S %p')
    
    return formatted_df

def parse_ath_price_messages(messages):
    """
    Parses Telegram messages with debug output.
    """
    data = []
    
    for message in messages:
        print("\n" + "="*50)
        print("RAW MESSAGE TEXT:")
        print(message.text)
        print("="*50 + "\n")
        
        message_data = {
            'name': None,
            'token': None,
            '5m_change': None,
            '1h_change': None,
            '6h_change': None,
            '5m_transactions': None,
            '5m_volume': None,
            'mcp': None,
            'liquidity_sol': None,
            'holders': None,
            'open_time': None,
            'top10_holders': None,
            'dev_hold_%': None,
            'dev_burnt': None,
            'telegram': None,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        lines = message.text.split('\n')
        
        for line in lines:
            print(f"\nProcessing line: {line}")
            
            # Extract name
            if '**$' in line:
                name_match = re.search(r'\((.*?)\)', line)
                if name_match:
                    message_data['name'] = name_match.group(1).strip()
                    print(f"Found name: {message_data['name']}")
            
            # Extract token address
            if line.strip().startswith('`') and line.strip().endswith('`'):
                message_data['token'] = line.strip('`')
                print(f"Found token: {message_data['token']}")
            
            # Extract percentage changes
            if '📈 5m | 1h | 6h:' in line:
                print("Found percentage line")
                percentages = re.findall(r'[-\d.]+%', line)
                print(f"Extracted percentages: {percentages}")
                if len(percentages) == 3:
                    message_data['5m_change'] = percentages[0]
                    message_data['1h_change'] = percentages[1]
                    message_data['6h_change'] = percentages[2]
            
            # Extract 5m Transactions and Volume
            if '🎲 5m TXs/Vol:' in line:
                print("Found transactions line")
                tx_vol = re.search(r'5m TXs/Vol:\s*\*+(\d+)\*+/\*+\$([\d.,]+K)\*+', line)
                if tx_vol:
                    message_data['5m_transactions'] = tx_vol.group(1)
                    message_data['5m_volume'] = tx_vol.group(2)
                    print(f"Extracted tx: {tx_vol.group(1)}, vol: {tx_vol.group(2)}")
            
            # Extract MCP
            if '💡 MCP:' in line:
                print("Found MCP line")
                mcp = re.search(r'MCP:\s*\*+\$([\d.,]+K)\*+', line)
                if mcp:
                    message_data['mcp'] = mcp.group(1)
                    print(f"Extracted MCP: {mcp.group(1)}")
            
            # Extract Liquidity SOL
            if '💧 Liq:' in line:
                print("Found liquidity line")
                liq = re.search(r'Liq:\s*\*+([\d.]+)\*+\s*\*+SOL\*+', line)
                if liq:
                    message_data['liquidity_sol'] = liq.group(1)
                    print(f"Extracted liquidity: {liq.group(1)}")
            
            # Extract DEV Burnt
            if 'DEV Burnt' in line:
                print("Found DEV burnt line")
                # Look for percentage in the burnt text
                burnt_percent = re.search(r'Rate:\s*(\d+\.?\d*)%', line)
                if burnt_percent:
                    message_data['dev_burnt'] = burnt_percent.group(1)
                    print(f"Extracted DEV burnt percentage: {message_data['dev_burnt']}%")
                else:
                    message_data['dev_burnt'] = '0'
                    print("No percentage found, setting DEV burnt to 0")
            
            # Extract Holders
            if '👥 Holder:' in line:
                print("Found holders line")
                holder = re.search(r'Holder:\s*\*+(\d+)\*+', line)
                if holder:
                    message_data['holders'] = holder.group(1)
                    print(f"Extracted holders: {holder.group(1)}")
            
            # Extract Open Time
            if '🕒 Open:' in line:
                print("Found open time line")
                open_time = re.search(r'Open:\s*\*+([\w\d]+)\*+\s*\*+ago\*+', line)
                if open_time:
                    message_data['open_time'] = open_time.group(1)
                    print(f"Extracted open time: {open_time.group(1)}")
            
            # Extract Top 10 Holders
            if 'TOP 10:' in line:
                print("Found top 10 line")
                top10 = re.search(r'TOP 10:\s*\*+([\d.]+)%\*+', line)
                if top10:
                    message_data['top10_holders'] = top10.group(1)
                    print(f"Extracted top10: {top10.group(1)}%")
            
            # Extract DEV Status
            if '⏳ DEV:' in line:
                print("Found DEV status line")
                message_data['dev_hold_%'] = "0%"
                print("Set DEV hold to 0%")
            
            # Extract DEV Burnt
            if 'DEV Burnt' in line:
                print("Found DEV burnt line")
                burnt = re.search(r'DEV Burnt.*?:\s*\*+(.+?)\*+', line)
                if burnt and burnt.group(1).strip() != '-':
                    message_data['dev_burnt'] = burnt.group(1).strip()
                else:
                    message_data['dev_burnt'] = '-'
                print(f"Extracted DEV burnt: {message_data['dev_burnt']}")
            
            # Extract Telegram Links
            if 'Backup BOT:' in line:
                print("Found telegram line")
                telegram_links = re.findall(r'\[.*?\]\((https?://t\.me/.*?)\)', line)
                message_data['telegram'] = ', '.join(telegram_links) if telegram_links else None
                print(f"Extracted telegram links: {message_data['telegram']}")
        
        print("\nFinal extracted data for this message:")
        for key, value in message_data.items():
            if value is not None:
                print(f"{key}: {value}")
        
        data.append(message_data)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Reorder columns
    desired_columns = [
        'name', 'token', '5m_change', '1h_change', '6h_change', 
        '5m_transactions', '5m_volume', 'mcp', 'liquidity_sol', 
        'holders', 'open_time', 'top10_holders', 'dev_hold_%', 
        'dev_burnt', 'telegram', 'timestamp'
    ]
    
    # Only include columns that exist in the DataFrame
    columns = [col for col in desired_columns if col in df.columns]
    
    return df[columns]

async def main(chat_name, limit, topic_id=None):
    """
    Connects to Telegram and retrieves messages.
    """
    try:
        with open('session.txt', 'r') as f:
            session_string = f.read()
        client = TelegramClient(StringSession(session_string), api_id, api_hash)
    except FileNotFoundError:
        print("No saved session found, using memory session")
        client = TelegramClient(MemorySession(), api_id, api_hash)
    
    try:
        print("Starting client...")
        await client.start()
        
        try:
            print(f"Attempting to connect to chat: {chat_name}")
            chat_info = await client.get_entity(chat_name)
            print(f"Successfully connected to chat: {chat_info.title}")
            
            topic_title = None
            if topic_id:
                result = await client(GetForumTopicsRequest(
                    channel=chat_info,
                    offset_date=0,
                    offset_id=0,
                    offset_topic=0,
                    limit=100
                ))
                topic_title = next((topic.title for topic in result.topics 
                                  if hasattr(topic, 'id') and topic.id == topic_id), "Unknown Topic")
                print(f"Fetching messages from topic: {topic_title} (ID: {topic_id})")
            
            messages = await client.get_messages(
                entity=chat_info, 
                limit=limit,
                reply_to=topic_id
            )
            print(f"Retrieved {len(messages)} messages")
            return {
                "messages": messages, 
                "channel": chat_info,
                "topic_title": topic_title
            }
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None
            
    finally:
        print("Disconnecting client...")
        await client.disconnect()

async def main_with_timeout(chat_name, limit, topic_id=None, timeout=120):
    """
    Wrapper function for main() with timeout.
    """
    try:
        result = await asyncio.wait_for(main(chat_name, limit, topic_id), timeout=timeout)
        return result
    except TimeoutError:
        print(f"Operation timed out after {timeout} seconds")
        return None

async def continuous_monitoring(chat_name, topic_id, limit, interval):
    """
    Continuously monitors Telegram messages and saves to CSV files.
    """
    print("Starting continuous monitoring...")
    seen_messages = set()  # Track processed tokens
    
    # Create more readable filename with start datetime
    start_time = datetime.now().strftime("%B_%d_%Y_%I_%M_%S_%p")  # Example: November_15_2024_03_18_17_AM
    csv_filename = f'Telegram_ATH_Data_{start_time}.csv'
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    csv_path = os.path.join('data', csv_filename)
    
    print(f"Data will be saved to: {csv_path}")
    
    while True:
        try:
            result = await main_with_timeout(chat_name, limit, topic_id)
            
            if result and result["messages"]:
                # Parse current messages
                current_df = parse_ath_price_messages(result["messages"])
                
                if not current_df.empty:  # Check if DataFrame is not empty
                    # Format DataFrame
                    current_df = format_dataframe_for_csv(current_df)
                    
                    # Check for new messages by comparing tokens
                    new_tokens = set(current_df['Token Address']) - seen_messages
                    
                    if new_tokens:
                        # Filter only new messages
                        new_df = current_df[current_df['Token Address'].isin(new_tokens)]
                        
                        # Update seen messages
                        seen_messages.update(new_tokens)
                        
                        # Append or create CSV file
                        if os.path.exists(csv_path):
                            existing_df = pd.read_csv(csv_path)
                            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                        else:
                            updated_df = new_df
                        
                        # Save updated DataFrame
                        updated_df.to_csv(csv_path, index=False)
                        print(f"Saved {len(new_tokens)} new messages at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"Total records in file: {len(updated_df)}")
                    else:
                        print(f"No new messages found at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Wait for next check
            await asyncio.sleep(interval)
            
        except Exception as e:
            print(f"Error occurred: {e}")
            await asyncio.sleep(10)  # Wait before retrying

if __name__ == "__main__":
    chat_name = "gmgnsignals"
    topic_id = 1115976  # ATH Price topic
    limit = 10  # Number of messages to fetch per check
    
    print("Starting continuous monitoring...")
    try:
        # INTERVAL SET TO 5 SECONDS TO SEARCH FOR NEW TOKENS
        asyncio.run(continuous_monitoring(chat_name, topic_id, limit, interval=5))
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"\nMonitoring stopped due to error: {e}")