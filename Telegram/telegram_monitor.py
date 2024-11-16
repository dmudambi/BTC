from pathlib import Path
import json
from datetime import datetime, timedelta
import os
import logging
import re
import pandas as pd
from typing import Dict, Optional, List, Any
from telethon import TelegramClient
from telethon.sessions import StringSession, MemorySession
from telethon.tl.functions.channels import GetForumTopicsRequest
from telethon.tl.types import Message, Channel, ForumTopicDeleted
import asyncio
from dotenv import load_dotenv

# Message limits for each group
MESSAGE_LIMITS = {
    "ATH_Price": 5,
    "PUMP_FDV_Surge": 5,
    "Solana_FDV_Surge": 5
}

class TopicConfig:
    def __init__(self):
        self.configs = {
            "ATH_Price": {
                "id": 1115976,  # ATH Price
                "min_market_cap": 10000,      # $10K
                "min_liquidity": 10,           # 10 SOL
                "min_5m_change": 2,           # 2%
                "min_5m_volume": 500,         # $500
                "min_holders": 10,            # 10 holders
                "max_top10_holders": 50,       # 50%
                "max_dev_holdings": 10,        # 10%
                "interval": 5,                 # 5 seconds
                "data_path": "data/ATH_Price/"
            },
            "PUMP_FDV_Surge": {
                "id": 1152511,  # PUMP FDV Surge
                "min_market_cap": 5000,       # $5K
                "min_liquidity": 5,            # 5 SOL
                "min_5m_change": 5,           # 5%
                "min_5m_volume": 250,         # $250
                "min_holders": 5,             # 5 holders
                "max_top10_holders": 60,       # 60%
                "max_dev_holdings": 15,        # 15%
                "interval": 5,                 # 5 seconds
                "data_path": "data/PUMP_FDV_Surge/"
            },
            "Solana_FDV_Surge": {
                "id": 1152525,  # Solana FDV Surge
                "min_market_cap": 7500,       # $7.5K
                "min_liquidity": 7,            # 7 SOL
                "min_5m_change": 17,           # 17%
                "min_5m_volume": 350,         # $350
                "min_holders": 7,             # 7 holders
                "max_top10_holders": 55,       # 55%
                "max_dev_holdings": 12,        # 12%
                "interval": 5,                 # 5 seconds
                "data_path": "data/Solana_FDV_Surge/"
            }
        }
        self._create_data_directories()

    def _create_data_directories(self):
        """Create data directories for each topic if they don't exist"""
        for config in self.configs.values():
            Path(config["data_path"]).mkdir(parents=True, exist_ok=True)

    def get_topic_config(self, topic_name):
        """Get configuration for a specific topic"""
        return self.configs.get(topic_name)

    def get_all_topics(self):
        """Get list of all configured topics"""
        return list(self.configs.keys())

class BaseMessageParser:
    def __init__(self):
        self.default_data = {
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
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Consistent datetime format
        }

    def _clean_numeric(self, value: str) -> float:
        """Clean numeric strings by removing symbols and converting K/M/B"""
        if not value:
            return 0.0
        
        value = str(value).strip().replace(',', '').replace('$', '').replace('%', '')
        
        multiplier = 1
        if value.endswith('K'):
            multiplier = 1000
            value = value[:-1]
        elif value.endswith('M'):
            multiplier = 1000000
            value = value[:-1]
        elif value.endswith('B'):
            multiplier = 1000000000
            value = value[:-1]
            
        try:
            return float(value) * multiplier
        except ValueError:
            return 0.0

    def _extract_telegram_links(self, line: str) -> str:
        """Extract telegram links from text"""
        links = re.findall(r'\[.*?\]\((https?://t\.me/.*?)\)', line)
        return ', '.join(links) if links else None

    def meets_requirements(self, data: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Check if parsed data meets the topic requirements"""
        try:
            mcp = self._clean_numeric(data.get('mcp', '0'))
            liq = self._clean_numeric(str(data.get('liquidity_sol', '0')))
            change_5m = self._clean_numeric(data.get('5m_change', '0'))
            volume_5m = self._clean_numeric(data.get('5m_volume', '0'))
            holders = self._clean_numeric(data.get('holders', '0'))
            top10 = self._clean_numeric(data.get('top10_holders', '100'))
            dev_holdings = self._clean_numeric(data.get('dev_hold_%', '100'))

            return (
                mcp >= config['min_market_cap'] and
                liq >= config['min_liquidity'] and
                change_5m >= config['min_5m_change'] and
                volume_5m >= config['min_5m_volume'] and
                holders >= config['min_holders'] and
                top10 <= config['max_top10_holders'] and
                dev_holdings <= config['max_dev_holdings']
            )
        except Exception as e:
            print(f"Error checking requirements: {e}")
            return False

class ATHPriceParser(BaseMessageParser):
    def parse_message(self, message):
        """Parse ATH Price message format"""
        message_data = self.default_data.copy()
        
        lines = message.text.split('\n')
        for line in lines:
            # Extract token name
            if '**$' in line:
                name_match = re.search(r'\((.*?)\)', line)
                if name_match:
                    message_data['name'] = name_match.group(1).strip()
            
            # Extract token address
            if line.strip().startswith('`') and line.strip().endswith('`'):
                message_data['token'] = line.strip('`')
            
            # Extract percentage changes
            if '📈 5m | 1h | 6h:' in line:
                percentages = re.findall(r'[-\d.]+%', line)
                if len(percentages) == 3:
                    message_data['5m_change'] = percentages[0]
                    message_data['1h_change'] = percentages[1]
                    message_data['6h_change'] = percentages[2]
            
            # Extract 5m Transactions and Volume
            if '🎲 5m TXs/Vol:' in line:
                tx_vol = re.search(r'TXs/Vol:\s*\*+([\d.,K]+)\*+/\*+\$([\d.,K]+)\*+', line)
                if tx_vol:
                    message_data['5m_transactions'] = tx_vol.group(1)
                    message_data['5m_volume'] = tx_vol.group(2)
            
            # Extract MCP
            if '💡 MCP:' in line:
                mcp = re.search(r'MCP:\s*\*+\$([\d.,K]+M?)\*+', line)
                if mcp:
                    message_data['mcp'] = mcp.group(1)
            
            # Extract Liquidity SOL
            if '💧 Liq:' in line:
                liq = re.search(r'Liq:\s*\*+([\d.]+)\*+\s*\*+SOL\*+', line)
                if liq:
                    message_data['liquidity_sol'] = liq.group(1)
            
            # Extract Holders
            if '👥 Holder:' in line:
                holders = re.search(r'Holder:\s*\*+([\d,]+)\*+', line)
                if holders:
                    message_data['holders'] = holders.group(1)
            
            # Extract Open Time
            if '🕒 Open:' in line:
                open_time = re.search(r'Open:\s*\*+([\w\d]+)\*+\s*\*+ago\*+', line)
                if open_time:
                    message_data['open_time'] = open_time.group(1)
            
            # Extract Top 10 Holders
            if 'TOP 10:' in line:
                top10 = re.search(r'TOP 10:\s*\*+([\d.]+%)\*+', line)
                if top10:
                    message_data['top10_holders'] = top10.group(1)
            
            # Extract DEV Status
            if '⏳ DEV:' in line:
                dev_status = re.search(r'\[(.*?)\]', line)
                if dev_status:
                    status = dev_status.group(1)
                    if status == "🚨 Sell All":
                        message_data['dev_hold_%'] = "0%"
                    elif "HOLD" in status.upper():
                        percentage = re.search(r'([\d.]+)%', status)
                        if percentage:
                            message_data['dev_hold_%'] = f"{percentage.group(1)}%"
                    else:
                        message_data['dev_hold_%'] = status
            
            # Extract DEV Burnt
            if 'DEV Burnt' in line:
                burnt = re.search(r'DEV Burnt.*?:\s*\*+([\w\-/]+)\*+', line)
                if burnt and burnt.group(1).strip() != '-':
                    message_data['dev_burnt'] = burnt.group(1).strip()
                else:
                    message_data['dev_burnt'] = '-'
            
            # Extract Telegram Links
            if '[✈️ Telegram]' in line:
                telegram = re.findall(r'\((https?://t\.me/.*?)\)', line)
                if telegram:
                    message_data['telegram'] = ', '.join(telegram)
        
        return message_data


class FDVSurgeParser(BaseMessageParser):
    def __init__(self):
        super().__init__()
        self.default_data.update({
            'fdv_surge_amount': None,
            'fdv_surge_percent': None
        })

    def parse_message(self, message):
        """Parse FDV Surge message format"""
        message_data = self.default_data.copy()
        
        lines = message.text.split('\n')
        for line in lines:
            # Extract FDV surge amount and percentage
            if 'FDV in 5 min' in line:
                fdv_amount = re.search(r'FDV in 5 min.*?\$([\d.,K]+)', line, re.IGNORECASE)
                fdv_percent = re.search(r'\(([\+\-]?\d+\.?\d*%)\)', line)
                if fdv_amount:
                    message_data['fdv_surge_amount'] = fdv_amount.group(1)
                if fdv_percent:
                    message_data['fdv_surge_percent'] = fdv_percent.group(1)
            
            # Extract token name
            if '**$' in line:
                name_match = re.search(r'\((.*?)\)', line)
                if name_match:
                    message_data['name'] = name_match.group(1).strip()
            
            # Extract token address
            if line.strip().startswith('`') and line.strip().endswith('`'):
                message_data['token'] = line.strip('`')
            
            # Extract percentage changes
            if '📈 5m | 1h | 6h:' in line:
                percentages = re.findall(r'[-\d.]+%', line)
                if len(percentages) == 3:
                    message_data['5m_change'] = percentages[0]
                    message_data['1h_change'] = percentages[1]
                    message_data['6h_change'] = percentages[2]
            
            # Extract 5m Transactions and Volume
            if '🎲 5m TXs/Vol:' in line:
                tx_vol = re.search(r'5m TXs/Vol:\s*\*+([\d.,K]+)\*+/\*+\$([\d.,K]+)\*+', line)
                if tx_vol:
                    message_data['5m_transactions'] = tx_vol.group(1)
                    message_data['5m_volume'] = tx_vol.group(2)
            
            # Extract MCP
            if '💡 MCP:' in line:
                mcp = re.search(r'MCP:\s*\*+\$?([\d.,K]+M?)\*+', line)
                if mcp:
                    message_data['mcp'] = mcp.group(1)
            
            # Extract Liquidity SOL
            if '💧 Liq:' in line:
                liq = re.search(r'Liq:\s*\*+([\d.]+)\*+\s*\*+SOL\*+', line)
                if liq:
                    message_data['liquidity_sol'] = liq.group(1)
            
            # Extract Holders
            if '👥 Holder:' in line:
                holders = re.search(r'Holder:\s*\*+([\d,]+)\*+', line)
                if holders:
                    message_data['holders'] = holders.group(1)
            
            # Extract Open Time
            if '🕒 Open:' in line:
                open_time = re.search(r'Open:\s*\*+([\w\d]+)\*+\s*\*+ago\*+', line)
                if open_time:
                    message_data['open_time'] = open_time.group(1)
            
            # Extract Top 10 Holders
            if 'TOP 10:' in line:
                top10 = re.search(r'TOP 10:\s*\*+([\d.]+%)\*+', line)
                if top10:
                    message_data['top10_holders'] = top10.group(1)
            
            # Extract DEV Status
            if '⏳ DEV:' in line:
                dev_status = re.search(r'\[(.*?)\]', line)
                if dev_status:
                    status = dev_status.group(1)
                    if status == "🚨 Sell All":
                        message_data['dev_hold_%'] = "0%"
                    elif "HOLD" in status.upper():
                        percentage = re.search(r'([\d.]+)%', status)
                        if percentage:
                            message_data['dev_hold_%'] = f"{percentage.group(1)}%"
                    else:
                        message_data['dev_hold_%'] = status
            
            # Extract DEV Burnt
            if 'DEV Burnt' in line:
                burnt = re.search(r'DEV Burnt.*?:\s*\*+([\w\-/]+)\*+', line)
                if burnt and burnt.group(1).strip() != '-':
                    message_data['dev_burnt'] = burnt.group(1).strip()
                else:
                    message_data['dev_burnt'] = '-'
            
            # Extract Telegram Links
            if '[✈️ Telegram]' in line:
                telegram = re.findall(r'\((https?://t\.me/.*?)\)', line)
                if telegram:
                    message_data['telegram'] = ', '.join(telegram)
        
        return message_data
    
class DataManager:
    def __init__(self, base_path="data/"):
        self.base_path = Path(base_path)
        self.retention_days = 7
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_cache = {}  # Store DataFrames in memory

    def save_data(self, df, topic_name, timestamp=None):
        """Save DataFrame to appropriate topic directory"""
        topic_path = self.base_path / topic_name
        filename = f"Telegram_{topic_name}_Data_{self.session_timestamp}.csv"
        filepath = topic_path / filename
        
        # Create directory if it doesn't exist
        topic_path.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp column if not present
        if 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Convert timestamp to datetime if it's not already
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Load existing data if available
        if topic_name in self.file_cache:
            existing_df = self.file_cache[topic_name]
        elif filepath.exists():
            existing_df = pd.read_csv(filepath)
            # Convert timestamp to datetime for existing data
            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
        else:
            existing_df = pd.DataFrame()

        # Combine existing and new data
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        
        # Drop duplicates based on all columns except timestamp
        cols_for_dedup = [col for col in combined_df.columns if col != 'timestamp']
        combined_df = combined_df.drop_duplicates(subset=cols_for_dedup, keep='last')
        
        # Sort by timestamp in descending order (newest first)
        combined_df = combined_df.sort_values('timestamp', ascending=False)
        
        # Save to file and update cache
        combined_df.to_csv(filepath, index=False)
        self.file_cache[topic_name] = combined_df
        return filepath

    def cleanup_old_files(self):
        """Remove files older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        for topic_dir in self.base_path.glob("**/"):
            if topic_dir.is_dir():
                for file in topic_dir.glob("*.csv"):
                    try:
                        file_date = datetime.strptime(
                            file.stem.split("_")[-1].split(".")[0], 
                            "%Y%m%d_%H%M%S"
                        )
                        if file_date < cutoff_date:
                            file.unlink()
                    except (ValueError, IndexError):
                        continue

class TelegramMonitorLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger("TelegramMonitor")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.log_dir / "telegram_monitor.log")
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log_error(self, topic: str, error_type: str, message: str):
        """Log error messages"""
        self.logger.error(f"Topic: {topic} - Error Type: {error_type} - Message: {message}")

    def log_filter_results(self, topic: str, total: int, passed: int):
        """Log filter results"""
        self.logger.info(f"Topic: {topic} - Filtered: {total} - Passed: {passed}")

class MonitoringStatus:
    def __init__(self, status_dir="status"):
        self.status_dir = Path(status_dir)
        self.status_dir.mkdir(exist_ok=True)
        self.status_file = self.status_dir / "monitoring_status.json"
        self.status = self._load_status()

    def _load_status(self) -> Dict:
        """Load status from file"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_status(self):
        """Save status to file"""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=4)

    def update_processing_stats(self, topic: str, processed: int, success: int, errors: int):
        """Update processing statistics for a topic"""
        if topic not in self.status:
            self.status[topic] = {}
        
        self.status[topic].update({
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages_processed": processed,
            "successful_filters": success,
            "errors": errors
        })
        self._save_status()

class TelegramClientManager:
    def __init__(self, api_id: str, api_hash: str, session_file: str = 'session.txt'):
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_file = session_file
        self.client = None

    async def initialize_client(self) -> TelegramClient:
        """Initialize and return Telegram client"""
        try:
            # Load existing session
            try:
                with open(self.session_file, 'r') as f:
                    session_string = f.read()
                self.client = TelegramClient(StringSession(session_string), 
                                          self.api_id, 
                                          self.api_hash)
            except FileNotFoundError:
                print("No saved session found, using memory session")
                self.client = TelegramClient(MemorySession(), 
                                          self.api_id, 
                                          self.api_hash)
            
            await self.client.start()
            return self.client

        except Exception as e:
            print(f"Error initializing client: {e}")
            return None

    async def get_messages(self, chat_name: str, topic_id: int, limit: int = 10):
        """Get messages from a specific chat and topic"""
        try:
            if not self.client:
                raise Exception("Client not initialized")

            # Get the chat entity
            chat = await self.client.get_entity(chat_name)
            
            # Get messages from the specific topic using the same parameters as the notebook
            messages = await self.client.get_messages(
                entity=chat, 
                limit=limit,
                reply_to=topic_id
            )
            
            return messages

        except Exception as e:
            print(f"Error getting messages: {e}")
            return []

    async def get_forum_topics(self, chat_name: str):
        """Get list of forum topics in the chat"""
        try:
            if not self.client:
                raise Exception("Client not initialized")

            chat = await self.client.get_entity(chat_name)
            
            result = await self.client(GetForumTopicsRequest(
                peer=chat,
                offset_date=0,
                offset_id=0,
                offset_topic=0,
                limit=100
            ))
            
            return result.topics

        except Exception as e:
            print(f"Error getting forum topics: {e}")
            return []

def get_parser_for_topic(topic_name: str):
    """Factory function to get appropriate parser for topic"""
    parser_map = {
        "ATH_Price": ATHPriceParser(),
        "PUMP_FDV_Surge": FDVSurgeParser(),
        "Solana_FDV_Surge": FDVSurgeParser()
    }
    return parser_map.get(topic_name, BaseMessageParser())

class TopicMonitor:
    def __init__(self, client_manager, topic_config, data_manager, logger, status_monitor):
        self.client_manager = client_manager
        self.topic_config = topic_config
        self.data_manager = data_manager
        self.logger = logger
        self.status_monitor = status_monitor
        self.seen_messages = {}
        self.running = False

    async def start_monitoring(self, chat_name: str):
        """Start monitoring all configured topics"""
        self.running = True
        monitoring_tasks = []
        
        for topic_name in self.topic_config.get_all_topics():
            config = self.topic_config.get_topic_config(topic_name)
            if config:
                self.seen_messages[topic_name] = set()
                task = asyncio.create_task(
                    self.monitor_topic(chat_name, topic_name, config)
                )
                monitoring_tasks.append(task)
        
        await asyncio.gather(*monitoring_tasks)

    async def monitor_topic(self, chat_name: str, topic_name: str, config: Dict[str, Any]):
        """Monitor a specific topic"""
        parser = get_parser_for_topic(topic_name)
        messages_processed = 0
        success_count = 0
        error_count = 0

        while self.running:
            try:
                messages = await self.client_manager.get_messages(
                    chat_name=chat_name,
                    topic_id=config['id'],
                    limit=MESSAGE_LIMITS.get(topic_name, 10)
                )

                if messages:
                    new_data = []
                    for message in messages:
                        if message.id not in self.seen_messages[topic_name]:
                            messages_processed += 1
                            
                            try:
                                parsed_data = parser.parse_message(message)
                                
                                if parser.meets_requirements(parsed_data, config):
                                    new_data.append(parsed_data)
                                    success_count += 1
                                
                                self.seen_messages[topic_name].add(message.id)
                                
                            except Exception as e:
                                error_count += 1
                                self.logger.log_error(
                                    topic_name, 
                                    "parse_error", 
                                    f"Message {message.id}: {str(e)}"
                                )

                    if new_data:
                        df = pd.DataFrame(new_data)
                        self.data_manager.save_data(df, topic_name)
                        
                        self.logger.log_filter_results(
                            topic_name, 
                            len(messages), 
                            len(new_data)
                        )

                self.status_monitor.update_processing_stats(
                    topic_name,
                    messages_processed,
                    success_count,
                    error_count
                )

                await asyncio.sleep(config['interval'])

            except Exception as e:
                self.logger.log_error(topic_name, "monitoring_error", str(e))
                await asyncio.sleep(10)

    def stop_monitoring(self):
        """Stop monitoring all topics"""
        self.running = False

class TelegramMonitorApp:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize configurations
        self.api_id = os.getenv("TELEGRAM_API_ID")
        self.api_hash = os.getenv("TELEGRAM_API_HASH")
        self.chat_name = "gmgnsignals"
        
        # Create base directories
        Path("data").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        Path("status").mkdir(exist_ok=True)
        
        # Initialize components
        self.topic_config = TopicConfig()
        self.data_manager = DataManager()
        self.logger = TelegramMonitorLogger()
        self.status_monitor = MonitoringStatus()
        
        # Initialize client manager
        self.client_manager = TelegramClientManager(
            api_id=self.api_id,
            api_hash=self.api_hash
        )
        
        # Initialize topic monitor
        self.monitor = None

    async def initialize(self) -> bool:
        """Initialize the monitoring system"""
        try:
            client = await self.client_manager.initialize_client()
            if not client:
                self.logger.logger.error("Failed to initialize Telegram client")
                return False
            
            self.monitor = TopicMonitor(
                self.client_manager,
                self.topic_config,
                self.data_manager,
                self.logger,
                self.status_monitor
            )
            
            self.logger.logger.info("Monitoring system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.logger.error(f"Initialization error: {e}")
            return False

    async def start(self):
        """Start the monitoring system"""
        try:
            # Initialize the system
            if not await self.initialize():
                self.logger.logger.error("Failed to initialize. Exiting...")
                return

            self.logger.logger.info("Starting Telegram monitoring system...")
            
            # Start monitoring
            await self.monitor.start_monitoring(self.chat_name)
            
        except KeyboardInterrupt:
            self.logger.logger.info("Received shutdown signal")
        except Exception as e:
            self.logger.logger.error(f"Unexpected error: {e}")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.monitor:
                self.monitor.stop_monitoring()
            
            await self.client_manager.close()
            self.logger.logger.info("Monitoring system shut down successfully")
            
        except Exception as e:
            self.logger.logger.error(f"Error during cleanup: {e}")

def main():
    """Entry point for the application"""
    app = TelegramMonitorApp()
    
    try:
        asyncio.run(app.start())
    except Exception as e:
        print(f"Fatal error: {e}")
        exit(1)

if __name__ == "__main__":
    main()