import asyncio
from decimal import Decimal
from trading_bot_test import TradingBot
import logging

async def simulate_price_feed(bot: TradingBot, token_address: str, price_sequence: List[Decimal], scenario_name: str):
    logging.info(f"Starting {scenario_name}")
    for price in price_sequence:
        await bot.process_price_update(token_address, price)
        await asyncio.sleep(1)
    logging.info(f"Completed {scenario_name}")

async def main():
    bot = TradingBot()
    
    # Test case 1: Successful entries and take profits
    token1 = "0xtoken1"
    await simulate_price_feed(
        bot, 
        token1,
        [
            Decimal('100'),  # Initial price
            Decimal('98'),   # First entry
            Decimal('97'),   # Second entry
            Decimal('95'),   # Third entry
            Decimal('102'),  # First take profit
            Decimal('105'),  # Second take profit
        ],
        "Normal Trading Scenario"
    )

    # Test case 2: Stop loss scenario
    token2 = "0xtoken2"
    bot.setup_test_orders(token2, Decimal('100'))
    await simulate_price_feed(
        bot,
        token2,
        [
            Decimal('100'),
            Decimal('98'),
            Decimal('95'),
            Decimal('89'),  # Should trigger stop loss
        ],
        "Stop Loss Scenario"
    )

    # Test case 3: Price validation failure
    token3 = "0xtoken3"
    bot.setup_test_orders(token3, Decimal('100'))
    await simulate_price_feed(
        bot,
        token3,
        [
            Decimal('100'),
            Decimal('98'),
            Decimal('50'),  # Large price jump
            Decimal('95'),
        ],
        "Price Validation Scenario"
    )

if __name__ == "__main__":
    asyncio.run(main()) 