# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LangGraph-based chatbot framework that provides streaming responses and memory management capabilities. The bot integrates with DeepSeek's chat model and includes web search functionality via Tavily.

## Key Commands

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the main chatbot
python chatbot/chatbot.py

# Run tests (when tests directory exists)
pytest tests
```

### Environment Setup
- Requires Python 3.13 (matching CI configuration)
- Set up `.env` file with:
  - `DEEPSEEK_API_KEY`: API key for DeepSeek chat model
  - `TAVILY_API_KEY`: API key for Tavily web search functionality

## Architecture

### Core Components

**State Management (`chatbot/chatbot.py:21-22`)**
- Uses `TypedDict` with `add_messages` annotation for conversation state
- Implements memory with `MemorySaver` checkpointing

**Graph Structure (`chatbot/chatbot.py:26-108`)**
- `START` → `chatbot` node → conditional routing
- Routes to `tools` node if tool calls detected, otherwise to `END`
- `tools` → `chatbot` loop for continued conversation

**Key Nodes:**
- `chatbot`: Handles conversation logic and model inference using DeepSeek
- `tools`: Executes tool calls via `TavilySearchToolNode`

**Tool System (`chatbot/TavilySearchToolNode.py`)**
- Custom tool node for executing Tavily web search
- Converts tool calls to `ToolMessage` responses
- Extensible for additional tools

### Memory and Message Handling
- Message trimming with `trim_messages()` to manage token limits (currently set to 10 tokens)
- Persistent conversation memory via LangGraph's `MemorySaver`
- Thread-based conversations using configurable thread IDs

### Model Configuration
- Uses DeepSeek chat model via `ChatOpenAI` interface
- Custom base URL: `https://api.deepseek.com`
- Tool binding for Tavily search integration

## Development Notes

### Adding New Tools
1. Create tool instances and add them to the tools list in `chatbot.py`
2. Update `TavilySearchToolNode` (or create new tool nodes) to handle new tools
3. The routing logic in `route_tools()` automatically detects tool calls

### Testing
- CI/CD pipeline runs on push/PR to master branch
- Tests run with `pytest tests` (requires tests directory to exist)
- Environment variables for tests configured in GitHub Actions secrets

### Code Structure
- Main chat logic in `chatbot/chatbot.py` (entry point)
- Tool implementations in separate files within `chatbot/`
- Uses LangGraph's `StateGraph` for conversation flow
- Implements streaming responses via `graph.stream()`