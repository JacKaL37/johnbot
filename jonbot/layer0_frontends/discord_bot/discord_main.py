import asyncio
import logging
import os

from dotenv import load_dotenv

from jonbot.layer0_frontends.discord_bot.discord_bot import DiscordBot
from jonbot.layer1_api_interface.app import health_check_api

logging.getLogger("discord").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


async def run_discord_bot_async():
    load_dotenv()
    try:
        discord_bot = DiscordBot()

        await health_check_api()

    except Exception as e:
        logger.exception(f"An error occurred while starting the Discord bot: {str(e)}")
        raise

    await discord_bot.start(os.getenv("DISCORD_TOKEN"))


def run_discord_bot():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_discord_bot_async())


if __name__ == "__main__":
    run_discord_bot()
