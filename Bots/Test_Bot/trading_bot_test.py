import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from decimal import Decimal
import asyncio
from datetime import datetime, timedelta
import logging
import csv
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OrderType(Enum):
    ENTRY = "Entry"
    TAKE_PROFIT = "Take Profit"
    STOP_LOSS = "Stop Loss"

@dataclass
class OrderLevel:
    price: Decimal
    quantity: Decimal
    filled: bool = False

@dataclass
class Position:
    token_address: str
    entry_orders: List[OrderLevel]
    take_profit_orders: List[OrderLevel]
    stop_loss: OrderLevel
    monitoring_start_time: datetime
    last_price: Decimal
    partial_exit_executed: bool = False

class PriceValidator:
    def __init__(self, max_deviation_percent: float = 10.0):
        self.price_history: List[Decimal] = []
        self.max_deviation = max_deviation_percent / 100

    async def validate_price(self, price: Decimal) -> bool:
        self.price_history.append(price)
        if len(self.price_history) < 4:
            return False
        
        # Keep only last 4 prices
        self.price_history = self.price_history[-4:]
        
        initial_price = self.price_history[0]
        for check_price in self.price_history[1:]:
            deviation = abs(check_price - initial_price) / initial_price
            if deviation > self.max_deviation:
                return False
        return True

class TradingBot:
    def __init__(self, max_gas_usd: float = 50.0):
        self.active_positions: Dict[str, Position] = {}
        self.price_validator = PriceValidator()
        self.max_gas_usd = max_gas_usd
        self.trade_log_file = 'trade_executions.csv'
        self.position_summary_file = 'position_summary.json'
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self._initialize_trade_log()

    def _initialize_trade_log(self):
        with open(self.trade_log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Token Address', 'Order Type', 'Price', 'Quantity'])

    def log_trade_execution(self, token_address: str, order_type: str, price: Decimal, quantity: Decimal):
        logging.info(f"{order_type} order executed for {token_address} at {price} with quantity {quantity}")
        with open(self.trade_log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now(), token_address, order_type, price, quantity])

    def setup_test_orders(self, token_address: str, current_price: Decimal):
        """Setup test orders at 2%, 3%, 5% below current price and take profits"""
        entry_prices = [
            current_price * Decimal('0.98'),  # 2% below
            current_price * Decimal('0.97'),  # 3% below
            current_price * Decimal('0.95')   # 5% below
        ]
        
        take_profit_prices = [
            current_price * Decimal('1.02'),  # 2% above
            current_price * Decimal('1.05')   # 5% above
        ]
        
        stop_loss_price = current_price * Decimal('0.90')  # 10% below
        
        entry_orders = [
            OrderLevel(price=price, quantity=Decimal('15'))
            for price in entry_prices
        ]
        
        take_profit_orders = [
            OrderLevel(price=price, quantity=Decimal('0'))  # Quantity will be set when entries are filled
            for price in take_profit_prices
        ]
        
        stop_loss = OrderLevel(price=stop_loss_price, quantity=Decimal('0'))
        
        position = Position(
            token_address=token_address,
            entry_orders=entry_orders,
            take_profit_orders=take_profit_orders,
            stop_loss=stop_loss,
            monitoring_start_time=datetime.now(),
            last_price=current_price
        )
        
        self.active_positions[token_address] = position

    def _safe_log_trade(self, token_address: str, order_type: str, price: Decimal, quantity: Decimal):
        try:
            self.log_trade_execution(token_address, order_type, price, quantity)
        except Exception as e:
            self.logger.error(f"Failed to log trade: {e}")

    async def process_price_update(self, token_address: str, new_price: Decimal):
        try:
            if token_address not in self.active_positions:
                return
            
            position = self.active_positions[token_address]
            
            # Check monitoring timeout
            if datetime.now() - position.monitoring_start_time > timedelta(hours=3):
                if not any(order.filled for order in position.entry_orders):
                    logging.info(f"Monitoring timeout for {token_address}")
                    del self.active_positions[token_address]
                    return
                
            # Validate price
            if not await self.price_validator.validate_price(new_price):
                logging.warning(f"Price validation failed for {token_address}: {new_price}")
                return
            
            position.last_price = new_price
            
            # Process entry orders
            for entry_order in position.entry_orders:
                if not entry_order.filled and new_price <= entry_order.price:
                    entry_order.filled = True
                    self._safe_log_trade(token_address, "Entry", new_price, entry_order.quantity)
                    
            # Calculate total position size
            total_filled = sum(
                order.quantity for order in position.entry_orders if order.filled
            )
            
            if total_filled > 0:
                # Update take profit quantities
                if not position.partial_exit_executed:
                    position.take_profit_orders[0].quantity = total_filled * Decimal('0.4')
                    position.take_profit_orders[1].quantity = total_filled * Decimal('0.6')
                
                # Process take profit orders
                for tp_order in position.take_profit_orders:
                    if not tp_order.filled and new_price >= tp_order.price:
                        tp_order.filled = True
                        self._safe_log_trade(token_address, "Take Profit", new_price, tp_order.quantity)
                        position.partial_exit_executed = True
                    
                # Process stop loss
                if new_price <= position.stop_loss.price:
                    self._safe_log_trade(token_address, "Stop Loss", new_price, total_filled)
                    del self.active_positions[token_address]
            
            self.save_position_summary()
        except Exception as e:
            self.logger.error(f"Error processing price update for {token_address}: {e}")
            # Optionally, we might want to remove the position if there's a critical error
            # self.active_positions.pop(token_address, None)

    def save_position_summary(self):
        """Save current position status to JSON file"""
        summary = {}
        for token_address, position in self.active_positions.items():
            summary[token_address] = {
                "entry_orders_filled": sum(1 for o in position.entry_orders if o.filled),
                "take_profits_filled": sum(1 for o in position.take_profit_orders if o.filled),
                "current_price": str(position.last_price),
                "monitoring_duration": str(datetime.now() - position.monitoring_start_time)
            }
        
        with open(self.position_summary_file, 'w') as f:
            json.dump(summary, f, indent=4)

async def simulate_price_feed(bot: TradingBot, token_address: str, price_sequence: List[Decimal], scenario_name: str):
    logging.info(f"Starting {scenario_name}")
    for price in price_sequence:
        await bot.process_price_update(token_address, price)
        await asyncio.sleep(1)
    logging.info(f"Completed {scenario_name}")

async def main():
    logging.info("Starting trading bot test")
    bot = TradingBot()
    
    # Test case 1: Token with successful entries and take profits
    token1 = "0xtoken1"
    initial_price = Decimal('100')
    bot.setup_test_orders(token1, initial_price)
    
    price_sequence = [
        Decimal('100'),  # Initial price
        Decimal('98'),   # First entry
        Decimal('97'),   # Second entry
        Decimal('95'),   # Third entry
        Decimal('102'),  # First take profit
        Decimal('105'),  # Second take profit
    ]
    
    await simulate_price_feed(bot, token1, price_sequence, "Normal Trading Scenario")
    logging.info("Test completed")

if __name__ == "__main__":
    asyncio.run(main())