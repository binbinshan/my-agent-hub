"""Utility functions for the chatbot."""
import json
from typing import Dict, Any, List
from langchain_core.messages import BaseMessage


def format_tool_result(result: Any) -> str:
    """Format tool result for display.

    Args:
        result: Raw result from tool

    Returns:
        Formatted string representation
    """
    if isinstance(result, dict):
        return json.dumps(result, indent=2, ensure_ascii=False)
    elif isinstance(result, list):
        return "\n".join(str(item) for item in result)
    else:
        return str(result)


def count_tokens(messages: List[BaseMessage]) -> int:
    """Simple token counter (approximate).

    Args:
        messages: List of messages

    Returns:
        Approximate token count
    """
    total = 0
    for message in messages:
        # Rough estimation: 1 token ≈ 4 characters for English
        # This is a very rough approximation for Chinese content
        total += len(message.content) // 2
    return total


def safe_extract_content(message: BaseMessage) -> str:
    """Safely extract content from a message.

    Args:
        message: Message instance

    Returns:
        Content string or empty string if extraction fails
    """
    try:
        return message.content if message.content else ""
    except AttributeError:
        return ""


def sanitize_user_input(user_input: str) -> str:
    """Sanitize user input for safety.

    Args:
        user_input: Raw user input

    Returns:
        Sanitized input string
    """
    # Remove potential harmful patterns
    # This is a basic implementation - customize based on your needs
    if not user_input:
        return ""

    # Trim whitespace
    sanitized = user_input.strip()

    # Limit length to prevent abuse
    max_length = 10000
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."

    return sanitized