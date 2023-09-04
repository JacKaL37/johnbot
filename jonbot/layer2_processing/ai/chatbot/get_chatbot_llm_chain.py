from typing import Dict

from jonbot.layer2_processing.ai.chatbot.chatbot_llm_chain import (
    ChatbotLLMChain,
)
from jonbot.layer2_processing.backend_database_operator.backend_database_operator import (
    BackendDatabaseOperations,
)
from jonbot.models.conversation_models import ChatRequest
from jonbot.system.setup_logging.get_logger import get_jonbot_logger

logger = get_jonbot_logger()


async def get_chatbot_llm_chain(
        chat_request: ChatRequest,
        existing_chatbot_llm_chains: Dict[str, ChatbotLLMChain],
        database_operations: BackendDatabaseOperations,
) -> ChatbotLLMChain:
    context_path = str(chat_request.context_route.as_flat_dict)

    if context_path in existing_chatbot_llm_chains:
        return existing_chatbot_llm_chains[context_path]
    else:
        existing_chatbot_llm_chains[
            context_path
        ] = await ChatbotLLMChain.from_context_route(
            context_route=chat_request.context_route,
            database_name=chat_request.database_name,
            database_operations=database_operations,
        )
        return existing_chatbot_llm_chains[context_path]