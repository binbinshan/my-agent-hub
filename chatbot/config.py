"""Configuration module for the chatbot."""
from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ChatConfig:
    """Configuration settings for the chatbot."""

    # Model settings
    model_name: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    max_tokens: int = 4000
    temperature: float = 0.7

    # Memory settings
    max_history_tokens: int = 10
    strategy: str = "last"
    include_system: bool = True
    start_on: str = "human"
    allow_partial: bool = False

    # Tool settings
    max_search_results: int = 5

    # Thread settings
    default_thread_id: str = "default"

    @classmethod
    def from_env(cls) -> "ChatConfig":
        """Create configuration from environment variables."""
        return cls(
            model_name=os.getenv("MODEL_NAME", cls.model_name),
            base_url=os.getenv("BASE_URL", cls.base_url),
            max_tokens=int(os.getenv("MAX_TOKENS", str(cls.max_tokens))),
            temperature=float(os.getenv("TEMPERATURE", str(cls.temperature))),
            max_history_tokens=int(os.getenv("MAX_HISTORY_TOKENS", str(cls.max_history_tokens))),
            max_search_results=int(os.getenv("MAX_SEARCH_RESULTS", str(cls.max_search_results))),
            default_thread_id=os.getenv("DEFAULT_THREAD_ID", cls.default_thread_id)
        )

    def validate(self) -> None:
        """Validate configuration settings."""
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("DEEPSEEK_API_KEY environment variable is required")
        if not os.getenv("TAVILY_API_KEY"):
            raise ValueError("TAVILY_API_KEY environment variable is required")