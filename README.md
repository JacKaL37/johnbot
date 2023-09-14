# jonbot

a friendly machine whats nice to talk to

This is a template repo, clone it and make your own friendo! Or use mine or whatever! No kings, no masters!

this bot helped me build this bot 🤖❤️✨


---

jonbot is a chatbot with multiple frontends (Discord, Telegram) powered by AI
through [LangChain](https://langchain.readthedocs.io/en/latest/) and connected to a [MongDB](mongodb.com) database.

## Installation

Create and activate an environment:
e.g.

```
conda create -n jonbot-env python=3.11
conda activate jonbot-env
```

Clone the repo:

```
git clone https://github.com/jonmatthis/jonbot
cd jonbot
```

Install dependencies:

```
pip install -e .
```

## Setup

- Create a `.env` file with your API keys for OpenAI, Discord, Telegram etc. See `.env.example`
- Add all relevant numbers and ids for the Discord and Telegram bot
    - Discord - https://guide.pycord.dev/getting-started/creating-your-first-bot
    - Telegram - https://core.telegram.org/bots#how-do-i-create-a-bot

## Run with: `python -m jonbot`  🤖❤️✨

## Usage

- Interact with the bot through Discord (text and voice memos)/Telegram(text only for now)
- API endpoints provide programmatic access to core functionality
- `__main__.py` starts up all services

## Architecture

```mermaid
graph TD

    classDef actor fill:#9E9E9E,stroke:#757575,stroke-width:2px,color:#000,rounding:5px;
    classDef interface fill:#90A4AE,stroke:#757575,stroke-width:2px,color:#000,rounding:5px;
    classDef api fill:#A5D6A7,stroke:#757575,stroke-width:2px,color:#000,rounding:5px;
    classDef core fill:#FFE082,stroke:#757575,stroke-width:2px,color:#000,rounding:5px;
    classDef data fill:#FFAB91,stroke:#757575,stroke-width:2px,color:#000,rounding:5px;

    A(["External User"]):::actor

    subgraph JonBot
        subgraph Frontend User Interfaces
            B1["Frontend: Discord"]:::interface
            B2["Frontend: Slack (planned)"]:::interface
            B3["Frontend: Telegram (planned)"]:::interface
        end
    
        subgraph Backend
            subgraph RESTful API
                C["API Interface (FastAPI)"]:::api
            end
            subgraph Controller
                D ["Controller"]:::core
                E ["BackendDatabaseOperations"]:::core
            end
    
            subgraph Processes
                F["Chatbot\nLangChain/OpenAI/Anthropic\nVoiceTranscription - Whisper"]:::core
            end
        
            subgraph Data Layer
                G["Database MongoDB"]:::data
            end
        end

        H(["Pydantic Data Models"]):::data

    
    end
    A --> B1
    A --> B2
    A --> B3
    B1 --> C
    B2 --> C
    B3 --> C
    C --> D
    D --> E
    D --> F
    E --> G
    H --> G
    H --> C
```

**Layer 0 - Frontends**

- `discord_bot`: Discord bot client and event handlers
- `telegram_bot`: Telegram bot client and handlers

**Layer 1 - API Interface**

- `app.py`: FastAPI application with endpoints

**Layer 2 - Core Processes**

- `ai_chatbot`: LangChain setup and processing
- `audio_transcription`: Transcribing audio to text with Whisper

**Layer 3 - Data Layer**

- `database`: MongoDB integration
- `data_models`: Pydantic models for data
- `system`: Configuration and utilities
