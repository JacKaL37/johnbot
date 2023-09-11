import discord

from jonbot.backend.data_layer.models.discord_stuff.environment_config.load_discord_config import (
    get_or_create_discord_environment_config,
)
from jonbot.system.setup_logging.get_logger import get_jonbot_logger

logger = get_jonbot_logger()

FINISHED_VOICE_RECORDING_PREFIX = "Finished! Recorded audio for"
TRANSCRIBED_AUDIO_PREFIX = "Transcribed audio for"
RESPONSE_INCOMING_TEXT = "response incoming..."
ERROR_MESSAGE_REPLY_PREFIX_TEXT = (
    f"Sorry, an error occurred while processing your request"
)
NEW_THREAD_MESSAGE_PREFIX_TEXT = "A new thread has been created! \n"

IGNORE_PREFIX = "~"  # If a message starts with this, the bot will ignore it


def this_message_is_from_a_bot(message: discord.Message) -> bool:
    # returns True if the author of the message is a bot
    return message.author.bot


# def check_if_transcribed_audio_message(message: discord.Message) -> bool:
#     return message.content.startswith(
#         TRANSCRIBED_AUDIO_PREFIX
#     ) or message.content.startswith(FINISHED_VOICE_RECORDING_PREFIX)


def check_if_new_thread_message(message: discord.Message) -> bool:
    return message.content.startswith(NEW_THREAD_MESSAGE_PREFIX_TEXT)


def message_starts_with_ignore_prefix(message: discord.Message) -> bool:
    return message.content.startswith(IGNORE_PREFIX)


def should_reply(message: discord.Message, bot_user_name: str) -> bool:
    if not allowed_to_reply(message):
        logger.debug(
            f"Message `{message.content}` was not handled by the bot {bot_user_name} (reason: not allowed to reply)"
        )
        return False

    if message_starts_with_ignore_prefix(message):
        logger.debug(
            f"Message `{message.content}` was not handled by the bot {bot_user_name} (reason: starts with ignore prefix{IGNORE_PREFIX})"
        )
        return False

    if check_if_new_thread_message(message):
        logger.debug(
            f"Message `{message.content}` was handled by the bot {bot_user_name} (reason: new thread message)"
        )
        return True

    if this_message_is_from_a_bot(message):
        logger.debug(
            f"Message `{message.content}` was not handled by the bot {bot_user_name} (reason: bot message)"
        )
        return False

    if message.author.name == bot_user_name:
        logger.debug(
            f"Message `{message.content}` was not handled by the bot {bot_user_name} (reason: self-generated message)"
        )
        return False

    logger.debug(
        f"Message `{message.content}` will be handled by the bot {bot_user_name} (reason: passed all checks)"
    )
    return True


def allowed_to_reply(message: discord.Message) -> bool:
    try:
        discord_config = get_or_create_discord_environment_config()
        logger.trace(
            f"Checking if message `{message.content}` is allowed to be handled by the bot {discord_config.BOT_NICK_NAME}"
        )

        # Handle DMs
        if message.channel.type == discord.ChannelType.private:
            logger.trace(
                f"Message `{message.content}` is allowed to be handled by the bot {discord_config.BOT_NICK_NAME} (reason: DM)"
            )
            return discord_config.DIRECT_MESSAGES_ALLOWED

        # Handle server messages
        if "thread" in message.channel.type.name:
            channel_id = message.channel.parent.id
        else:
            channel_id = message.channel.id

        server_data = None
        for server_name, details in discord_config.SERVERS_DETAILS.items():
            if message.guild.id == details["SERVER_ID"]:
                server_data = details
                break

        if not server_data:
            logger.error(
                f"Message received from server {message.guild.id} which is not in the list of allowed servers :O"
            )
            return False

        excluded_channels = server_data.get("EXCLUDED_CHANNEL_IDS", [])
        if channel_id in excluded_channels:
            logger.debug(
                f"Message `{message.content}` is not allowed to be handled by the bot {discord_config.BOT_NICK_NAME} (reason: excluded channel)"
            )
            return False

        allowed_categories = server_data.get("ALLOWED_CATEGORY_IDS", [])
        if allowed_categories == ["ALL"]:
            logger.trace(
                f"Message `{message.content}` is allowed to be handled by the bot {discord_config.BOT_NICK_NAME} (reason: allowed categories = ALL)"
            )
            return True

        if message.channel.category_id in allowed_categories:
            logger.trace(
                f"Message `{message.content}` is allowed to be handled by the bot {discord_config.BOT_NICK_NAME} (reason: allowed category)"
            )
            return True

        allowed_channels = server_data.get("ALLOWED_CHANNEL_IDS", [])
        if allowed_channels == ["ALL"]:
            logger.trace(
                f"Message `{message.content}` is allowed to be handled by the bot {discord_config.BOT_NICK_NAME} (reason: allowed channels = ALL)"
            )
            return True

        if channel_id not in allowed_channels:
            logger.debug(
                f"Message `{message.content}` is not allowed to be handled by the bot {discord_config.BOT_NICK_NAME} (reason: not allowed channel)"
            )
            return False

        logger.trace(
            f"Message `{message.content}` is allowed to be handled by the bot {discord_config.BOT_NICK_NAME} (reason: passed all checks)"
        )
        return True

    except Exception as e:
        logger.error(f"Error while checking if message is allowed to be handled: {e}")
        logger.exception(e)
        raise e