from typing import Optional, TYPE_CHECKING

from fastapi import FastAPI, WebSocket
from starlette.responses import StreamingResponse
from starlette.websockets import WebSocketDisconnect

import base64
import json

from jonbot.backend.controller.controller import Controller
from jonbot.backend.data_layer.models.conversation_models import ChatRequest
from jonbot.backend.data_layer.models.database_request_response_models import UpsertDiscordMessagesRequest, \
    UpsertResponse, ContextMemoryDocumentRequest, UpsertDiscordChatsRequest
from jonbot.backend.data_layer.models.health_check_status import HealthCheckResponse
from jonbot.backend.data_layer.models.user_stuff.memory.context_memory_document import ContextMemoryDocument
from jonbot.backend.data_layer.models.voice_to_text_request import VoiceToTextRequest, VoiceToTextResponse
from jonbot.system.setup_logging.get_logger import get_jonbot_logger

if TYPE_CHECKING:
    from jonbot.backend.backend_database_operator.backend_database_operator import (
        BackendDatabaseOperations,
    )

logger = get_jonbot_logger()

HEALTH_ENDPOINT = "/health"

CHAT_ENDPOINT = "/chat"
VOICE_TO_TEXT_ENDPOINT = "/voice_to_text"

UPSERT_MESSAGES_ENDPOINT = "/upsert_messages"
UPSERT_CHATS_ENDPOINT = "/upsert_chats"

GET_CONTEXT_MEMORY_ENDPOINT = "/get_context_memory"

CHAT_STATELESS_ENDPOINT = "/chat_stateless"


def register_api_routes(
        app: FastAPI,
        database_operations: "BackendDatabaseOperations",
        controller: Controller
):
    @app.get(HEALTH_ENDPOINT, response_model=HealthCheckResponse)
    async def health_check_endpoint():
        return HealthCheckResponse(status="alive")

    @app.get(GET_CONTEXT_MEMORY_ENDPOINT, response_model=Optional[ContextMemoryDocument])
    async def get_context_memory_endpoint(
            get_request: ContextMemoryDocumentRequest,
    ) -> ContextMemoryDocument:
        response = await database_operations.get_context_memory_document(
            request=get_request
        )

        if not response.success:
            logger.warning(
                f"Could not load context memory from database for context route: {get_request.query}"
            )
            return

        return response.data

    @app.post(VOICE_TO_TEXT_ENDPOINT, response_model=VoiceToTextResponse)
    async def voice_to_text_endpoint(
            voice_to_text_request: VoiceToTextRequest,
    ) -> VoiceToTextResponse:
        response = await controller.transcribe_audio(
            voice_to_text_request=voice_to_text_request
        )
        if response is None:
            return VoiceToTextResponse(success=False)
        return response

    @app.post(CHAT_ENDPOINT)
    async def chat_endpoint(chat_request: ChatRequest):
        logger.info(f"Received chat request: {chat_request}")
        return StreamingResponse(
            controller.get_response_from_chatbot(chat_request=chat_request),
            media_type="text/event-stream",
        )
    
    @app.websocket(CHAT_STATELESS_ENDPOINT)

    async def chat_stateless_endpoint(websocket: WebSocket):
        
        user_id = websocket.query_params.get('id')
        crude_api_token = websocket.query_params.get('token')
        decoded_crude_api_token = base64.b64decode(crude_api_token.encode()).decode()
        
        if decoded_crude_api_token != 'ggbotapi-1199299301957388239120':
            await websocket.close(code=1000)
            return
        
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_text()
                
                logger.info(f"Received message: {json.dumps(json.loads(data), indent=4)}")
                
                received_data = json.loads(data)

                logger.info(f"Context Route: {received_data['context_route']}")

                # build up the memory
                from langchain.memory import ConversationTokenBufferMemory

                # construct the langchain dude on the fly
                from langchain.chains import ConversationChain
                from langchain.llms import OpenAI 

                covnersation

                # jam conversation into memory

                # call openAI

                # if we get a stream token:
                # send stream token

                stream_token = "Xx"
                # response = controller.get_response_from_chatbot(chat_request=chat_request)
                response = {'type': 'token', 'content': stream_token}

                # else if we get a full response

                await websocket.send_text(json.dumps(response))
        except WebSocketDisconnect:
            logger.info("WebSocket connection closed")

    @app.post(UPSERT_MESSAGES_ENDPOINT)
    async def upsert_messages_endpoint(
            request: UpsertDiscordMessagesRequest,
    ) -> UpsertResponse:
        return await database_operations.upsert_discord_messages(request=request)

    @app.post(UPSERT_CHATS_ENDPOINT)
    async def upsert_chats_endpoint(
            request: UpsertDiscordChatsRequest,
    ) -> UpsertResponse:
        return await database_operations.upsert_discord_chats(request=request)
