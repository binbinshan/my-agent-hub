# Simple Chatbot

This project provides a simple CLI chatbot inspired by the notebook at
https://github.com/DjangoPeng/openai-quickstart/blob/main/langchain/langgraph/chatbot.ipynb

Files:
- `chatbot.py`: CLI script that builds a FAISS vectorstore from text files in `docs/` and runs a conversational retrieval chain using OpenAI and LangChain.
- `requirements.txt`: Python dependencies.
- `docs/sample.txt`: Sample document to index.

Setup:
1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set your OpenAI API key in the environment:

```bash
export OPENAI_API_KEY="sk-..."
```

3. Run the chatbot:

```bash
python my-agent-hub.py --reindex
```

The script will build a FAISS index from files in `docs/` and start an interactive chat loop.

