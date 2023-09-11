import tempfile
import traceback
from pathlib import Path
from typing import List, Union, Dict

import discord

from jonbot.api_interface.api_client.api_client import ApiClient
from jonbot.api_interface.api_client.get_or_create_api_client import (
    get_or_create_api_client,
)
from jonbot.api_interface.api_routes import CHAT_ENDPOINT, VOICE_TO_TEXT_ENDPOINT
from jonbot.backend.data_layer.models.conversation_models import ChatRequest
from jonbot.backend.data_layer.models.discord_stuff.environment_config.discord_environment import (
    DiscordEnvironmentConfig,
)
from jonbot.backend.data_layer.models.voice_to_text_request import VoiceToTextRequest
from jonbot.frontends.discord_bot.cogs.memory_scraper_cog import MemoryScraperCog
from jonbot.frontends.discord_bot.cogs.server_scraper_cog import ServerScraperCog
from jonbot.frontends.discord_bot.cogs.thread_cog import ThreadCog
from jonbot.frontends.discord_bot.cogs.voice_channel_cog import VoiceChannelCog
from jonbot.frontends.discord_bot.handlers.discord_message_responder import (
    DiscordMessageResponder,
)
from jonbot.frontends.discord_bot.handlers.should_process_message import (
    allowed_to_reply,
    should_reply,
    ERROR_MESSAGE_REPLY_PREFIX_TEXT,
)
from jonbot.frontends.discord_bot.operations.discord_database_operations import (
    DiscordDatabaseOperations,
)
from jonbot.frontends.discord_bot.utilities.print_pretty_terminal_message import (
    print_pretty_startup_message_in_terminal,
)
from jonbot.system.setup_logging.get_logger import get_jonbot_logger

logger = get_jonbot_logger()


class MyDiscordBot(discord.Bot):
    def __init__(
            self,
            environment_config: DiscordEnvironmentConfig,
            api_client: ApiClient = get_or_create_api_client(),
            **kwargs,
    ):
        super().__init__(**kwargs)
        self._local_message_prefix = ""
        if environment_config.IS_LOCAL:
            self._local_message_prefix = (
                f"(local - `{environment_config.BOT_NICK_NAME}`)\n"
            )

        self._api_client = api_client
        self._database_name = f"{environment_config.BOT_NICK_NAME}_database"
        self._database_operations = DiscordDatabaseOperations(
            api_client=api_client, database_name=self._database_name
        )
        self.add_cog(ServerScraperCog(database_operations=self._database_operations))
        self.add_cog(VoiceChannelCog(bot=self))
        self.add_cog(ThreadCog(bot=self))
        self.add_cog(
            MemoryScraperCog(database_name=self._database_name, api_client=api_client)
        )

    @discord.Cog.listener()
    async def on_ready(self):
        logger.success(f"Logged in as {self.user.name} ({self.user.id})")
        print_pretty_startup_message_in_terminal(self.user.name)

    @discord.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if not allowed_to_reply(message):
            return

        if not should_reply(message=message, bot_user_name=self.user.name):
            logger.debug(
                f"Message `{message.content}` was not handled by the bot: {self.user.name}"
            )
            return

        return await self.handle_message(message=message)

    async def handle_message(self, message: discord.Message):
        messages_to_upsert = [message]
        text_to_reply_to = f"{message.author}: {message.content}"
        try:
            async with message.channel.typing():
                if len(message.attachments) > 0:
                    logger.debug(f"Message has attachments: {message.attachments}")
                    for attachment in message.attachments:
                        if "audio" in attachment.content_type:
                            audio_response_dict = await self.handle_audio_message(message=message)
                            messages_to_upsert.extend(audio_response_dict["transcriptions_messages"])
                            new_text_to_reply_to = audio_response_dict["transcription_text"]
                            text_to_reply_to += f"\n\n{new_text_to_reply_to}"
                        else:
                            new_text_to_reply_to = await self.handle_text_attachments(attachment=attachment)
                            text_to_reply_to += f"\n\n{new_text_to_reply_to}"

            response_messages = await self.handle_text_message(
                message=message,
                respond_to_this_text=text_to_reply_to
            )

            messages_to_upsert.extend(response_messages)

        except Exception as e:
            await self._send_error_response(e, messages_to_upsert)
            return

        await self._database_operations.upsert_messages(messages=messages_to_upsert)

    async def _send_error_response(self, e: Exception, messages_to_upsert):
        # Create traceback string
        traceback_str = traceback.format_exc(limit=None)
        traceback_str = traceback_str.replace(str(Path.home()), "~")
        error_message = f"Error message: \n {str(e)}"

        # Write traceback to a temporary text file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(traceback_str.encode())
            temp_filepath = temp_file.name

        # Log the error message and traceback
        logger.exception(f"{error_message}\nTraceback: \n{traceback_str}")

        # Send the error message and the traceback file as an attachment
        await messages_to_upsert[-1].reply(f"{ERROR_MESSAGE_REPLY_PREFIX_TEXT} \n >  {error_message}", )
        # file=discord.File(temp_filepath))

        # Delete the temporary file after sending it
        Path(temp_filepath).unlink()

    async def handle_text_attachments(self, attachment: discord.Attachment) -> str:
        try:
            # Try to convert to text
            text_file = await attachment.read()
            text = text_file.decode("utf-8")
            return f"\n\n{attachment.filename}:\n\n++++++\n{text}\n++++++\n"
        except UnicodeDecodeError:
            logger.warning(f"Attachment type not supported: {attachment.content_type}")
            return f"\n\n{attachment.filename}:\n\n++++++\n{attachment.url}\n++++++(Note: Could not convert this file to text)\n"

    async def handle_text_message(
            self,
            message: discord.Message,
            respond_to_this_text: str,
    ) -> List[discord.Message]:
        chat_request = ChatRequest.from_discord_message(
            message=message,
            database_name=self._database_name,
            content=respond_to_this_text,
            extra_prompts={"test1": "Mention olive oil in your next message",
                           "test2": "Say it in a joke"}
        )
        message_responder = DiscordMessageResponder(message_prefix=self._local_message_prefix)
        await message_responder.initialize(message=message)

        async def callback(
                token: str, responder: DiscordMessageResponder = message_responder
        ):
            logger.trace(f"FRONTEND received token: `{repr(token)}`")
            await responder.add_token_to_queue(token=token)

        try:
            await self._api_client.send_request_to_api_streaming(
                endpoint_name=CHAT_ENDPOINT,
                data=chat_request.dict(),
                callbacks=[callback],
            )
            await message_responder.shutdown()
            return await message_responder.get_reply_messages()

        except Exception as e:
            await message_responder.add_token_to_queue(
                f"  --  \n!!!\n> `Oh no! An error while streaming reply...`"
            )
            await message_responder.shutdown()
            raise

    async def handle_audio_message(self, message: discord.Message) -> Dict[str, Union[str, List[discord.Message]]]:
        logger.info(f"Received voice memo from user: {message.author}")
        try:
            reply_message_content = (
                f"Transcribing audio from user `{message.author}`...\n\n"
            )
            responder = DiscordMessageResponder(message_prefix=self._local_message_prefix)
            await responder.initialize(
                message=message, initial_message_content=reply_message_content
            )
            for attachment in message.attachments:
                if attachment.content_type.startswith("audio"):
                    voice_to_text_request = VoiceToTextRequest(
                        audio_file_url=attachment.url
                    )

                    response = await self._api_client.send_request_to_api(
                        endpoint_name=VOICE_TO_TEXT_ENDPOINT,
                        data=voice_to_text_request.dict(),
                    )

                    reply_message_content += (
                        f"Transcribed Text:\n"
                        f"> {response['text']}\n\n"
                        f"File URL:{attachment.url}\n\n"
                    )

                    await responder.add_text_to_reply_message(
                        chunk=reply_message_content
                    )
                    await responder.shutdown()

                    logger.success(
                        f"VoiceToTextResponse payload received: \n {response}\n"
                        f"Successfully sent voice to text request payload to API!"
                    )
        except Exception as e:
            logger.exception(f"Error occurred while handling voice recording: {str(e)}")
            raise

        await responder.shutdown()
        transcriptions_messages = await responder.get_reply_messages()
        transcription_text = ""
        for message in transcriptions_messages:
            transcription_text += message.content
        return {"transcription_text": transcription_text, "transcriptions_messages": transcriptions_messages}
        # response_messages = await self.handle_text_message(
        #     message=transcriptions_messages[-1],
        #     respond_to_this_text=transcription_text,
        # )
        #
        # return transcriptions_messages + response_messages

    def get_text_to_reply_to(self, message: discord.Message) -> str:
        text_to_reply_to = ""
        if message.content:
            text_to_reply_to += f"{message.author.name}: {message.content}\n"

        text_from_attachments = self.handle_message_attachments()