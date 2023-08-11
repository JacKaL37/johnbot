from langchain import PromptTemplate
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from jonbot.layer2_core_processes.ai_chatbot.components.prompt.prompt_strings import DEFAULT_RULES_FOR_LIVING, \
    DEFAULT_CHATBOT_SYSTEM_PROMPT_TEMPLATE
from jonbot.layer3_data_layer.data_models.conversation_models import ConversationContext
from jonbot.layer3_data_layer.data_models.timestamp_model import Timestamp


class ChatbotPrompt(ChatPromptTemplate):
    @classmethod
    def build(cls,
              conversation_context: ConversationContext = None,
              system_prompt_template: str = DEFAULT_CHATBOT_SYSTEM_PROMPT_TEMPLATE,
              ) -> ChatPromptTemplate:

        system_prompt = PromptTemplate(template=system_prompt_template,
                                       input_variables=["timestamp",
                                                        "rules_for_living",
                                                        "context_route",
                                                        "context_description",
                                                        "chat_memory",
                                                        "vectorstore_memory"
                                                        ],
                                       )
        partial_system_prompt = system_prompt.partial(timestamp=str(Timestamp.now()),
                                                      rules_for_living=DEFAULT_RULES_FOR_LIVING,
                                                      context_route=conversation_context.context_route.parent if conversation_context else '',
                                                      context_description=conversation_context.context_description if conversation_context else '', )

        system_message_prompt = SystemMessagePromptTemplate(
            prompt=partial_system_prompt,
        )

        human_template = "{human_input}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            human_template
        )

        return cls.from_messages(
            [system_message_prompt, human_message_prompt]
        )
