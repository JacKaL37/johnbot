# golembot

a friendly machine whats nice to talk to

This is a template repo, clone it and make your own friendo! Or use mine or whatever! No kings, no masters!

this bot helped me build this bot 🤖❤️✨


---

GolemBot is a chatbot with multiple frontends (Discord, Telegram) powered by AI
through [LangChain](https://langchain.readthedocs.io/en/latest/) and connected to a [MongDB](mongodb.com) database.

## Installation

Create and activate an environment:
e.g.

```
conda create -n golembot-env python=3.11
conda activate golembot-env
```

Clone the repo:

```
git clone https://github.com/jonmatthis/golembot
cd golembot
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

## Run with: `python -m golembot`  🤖❤️✨

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

    subgraph Layer 0 - Interface
        B1["Frontend: Discord"]:::interface
        B2["Frontend: Telegram"]:::interface
    end

    subgraph Layer1 - API
        C["API Interface (FastAPI)"]:::api
    end

    subgraph Layer2 - Core Processes
        D["Chatbot\nLangChain/OpenAI/Anthropic\nVoiceTranscription - Whisper"]:::core
    end

    subgraph Layer3 - Data Layer
        E["Database - MongoDB\nDataModels - Pydantic"]:::data
    end

    A --> B1
    A --> B2
    B1 --> C
    B2 --> C
    C --> D
    D --> E
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
