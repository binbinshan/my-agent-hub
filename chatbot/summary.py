"""Conversation summary generation module."""
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from .logger import logger
from .config import ChatConfig


@dataclass
class ConversationSummary:
    """Represents a conversation summary."""
    thread_id: str
    title: str
    main_topics: List[str]
    key_points: List[str]
    user_goals: List[str]
    created_at: str
    message_count: int
    sentiment: str
    tags: List[str]
    summary_text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSummary":
        """Create from dictionary."""
        return cls(**data)


class SummaryGenerator:
    """Generates conversation summaries using LLM."""

    def __init__(self, config: ChatConfig):
        """Initialize the summary generator.

        Args:
            config: Chatbot configuration
        """
        self.config = config
        self.model = ChatOpenAI(
            model=config.model_name,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=config.base_url,
            temperature=0.3,  # Lower temperature for more consistent summaries
            max_tokens=1000
        )
        self._setup_prompts()

    def _setup_prompts(self) -> None:
        """Set up prompt templates for different summary tasks."""
        # Main summary prompt
        self.summary_prompt = PromptTemplate.from_template("""
        分析以下对话内容，生成结构化摘要：

        对话内容：
        {messages}

        请生成JSON格式的摘要，包含以下字段：
        {{
            "title": "对话标题（不超过20字）",
            "summary_text": "对话摘要（100-200字）",
            "main_topics": ["主要话题1", "主要话题2"],
            "key_points": ["关键点1", "关键点2", "关键点3"],
            "user_goals": ["用户目标1", "用户目标2"],
            "sentiment": "正面/负面/中性",
            "tags": ["标签1", "标签2", "标签3"]
        }}
        """)

        # Title generation prompt
        self.title_prompt = PromptTemplate.from_template("""
        为以下对话生成一个简洁的标题（不超过20字）：

        {messages}

        只返回标题，不要其他内容。
        """)

        # Topic extraction prompt
        self.topics_prompt = PromptTemplate.from_template("""
        从以下对话中提取3-5个主要话题：

        {messages}

        返回JSON格式：{{"topics": ["话题1", "话题2", "话题3"]}}
        """)

    def _format_messages_for_summary(self, messages: List[BaseMessage],
                                   max_messages: int = 10) -> str:
        """Format messages for summary generation.

        Args:
            messages: List of messages to format
            max_messages: Maximum number of messages to include

        Returns:
            Formatted string of messages
        """
        # Take recent messages if too many
        recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages

        formatted = []
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                formatted.append(f"用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"助手: {msg.content}")
            elif isinstance(msg, SystemMessage):
                formatted.append(f"系统: {msg.content}")

        return "\n\n".join(formatted)

    def generate_summary(self, messages: List[BaseMessage],
                        thread_id: str) -> ConversationSummary:
        """Generate a comprehensive conversation summary.

        Args:
            messages: List of conversation messages
            thread_id: Thread ID for the conversation

        Returns:
            ConversationSummary object
        """
        logger.info(f"Generating summary for thread {thread_id}")

        if not messages:
            logger.warning("No messages to summarize")
            return self._create_empty_summary(thread_id)

        try:
            # Format messages
            formatted_messages = self._format_messages_for_summary(messages)

            # Generate main summary
            response = self.model.invoke(
                self.summary_prompt.format(messages=formatted_messages)
            )

            # Parse JSON response
            try:
                summary_data = json.loads(response.content)
            except json.JSONDecodeError:
                logger.error("Failed to parse summary JSON, using fallback")
                summary_data = self._generate_fallback_summary(formatted_messages)

            # Create summary object
            summary = ConversationSummary(
                thread_id=thread_id,
                title=summary_data.get("title", "未命名对话"),
                main_topics=summary_data.get("main_topics", []),
                key_points=summary_data.get("key_points", []),
                user_goals=summary_data.get("user_goals", []),
                created_at=datetime.now().isoformat(),
                message_count=len(messages),
                sentiment=summary_data.get("sentiment", "中性"),
                tags=summary_data.get("tags", []),
                summary_text=summary_data.get("summary_text", "")
            )

            logger.info(f"Generated summary: {summary.title}")
            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return self._create_error_summary(thread_id, str(e))

    def _generate_fallback_summary(self, messages: str) -> Dict[str, Any]:
        """Generate a basic summary when JSON parsing fails.

        Args:
            messages: Formatted messages

        Returns:
            Basic summary dictionary
        """
        return {
            "title": "对话摘要",
            "summary_text": messages[:200] + "..." if len(messages) > 200 else messages,
            "main_topics": ["对话"],
            "key_points": ["用户与助手进行了交流"],
            "user_goals": ["获取信息"],
            "sentiment": "中性",
            "tags": ["未分类"]
        }

    def _create_empty_summary(self, thread_id: str) -> ConversationSummary:
        """Create an empty summary for conversations with no messages."""
        return ConversationSummary(
            thread_id=thread_id,
            title="空对话",
            main_topics=[],
            key_points=[],
            user_goals=[],
            created_at=datetime.now().isoformat(),
            message_count=0,
            sentiment="中性",
            tags=["空对话"],
            summary_text="此对话暂无内容。"
        )

    def _create_error_summary(self, thread_id: str, error: str) -> ConversationSummary:
        """Create an error summary when generation fails."""
        return ConversationSummary(
            thread_id=thread_id,
            title="摘要生成失败",
            main_topics=["错误"],
            key_points=[f"生成摘要时出错: {error}"],
            user_goals=[],
            created_at=datetime.now().isoformat(),
            message_count=0,
            sentiment="负面",
            tags=["错误"],
            summary_text="摘要生成失败，请稍后重试。"
        )

    def generate_title(self, messages: List[BaseMessage]) -> str:
        """Generate a title for the conversation.

        Args:
            messages: List of conversation messages

        Returns:
            Generated title
        """
        try:
            formatted_messages = self._format_messages_for_summary(messages, max_messages=5)
            response = self.model.invoke(
                self.title_prompt.format(messages=formatted_messages)
            )
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating title: {str(e)}")
            return "未命名对话"

    def extract_topics(self, messages: List[BaseMessage]) -> List[str]:
        """Extract main topics from conversation.

        Args:
            messages: List of conversation messages

        Returns:
            List of topics
        """
        try:
            formatted_messages = self._format_messages_for_summary(messages)
            response = self.model.invoke(
                self.topics_prompt.format(messages=formatted_messages)
            )
            data = json.loads(response.content)
            return data.get("topics", [])
        except Exception as e:
            logger.error(f"Error extracting topics: {str(e)}")
            return []


class SummaryStorage:
    """Stores and retrieves conversation summaries."""

    def __init__(self, storage_dir: str = "summaries"):
        """Initialize summary storage.

        Args:
            storage_dir: Directory to store summaries
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def save_summary(self, summary: ConversationSummary) -> bool:
        """Save summary to storage.

        Args:
            summary: Summary to save

        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = os.path.join(self.storage_dir, f"{summary.thread_id}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"Saved summary for thread {summary.thread_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving summary: {str(e)}")
            return False

    def load_summary(self, thread_id: str) -> Optional[ConversationSummary]:
        """Load summary from storage.

        Args:
            thread_id: Thread ID to load

        Returns:
            Summary if found, None otherwise
        """
        try:
            filepath = os.path.join(self.storage_dir, f"{thread_id}.json")
            if not os.path.exists(filepath):
                return None

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ConversationSummary.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading summary: {str(e)}")
            return None

    def list_summaries(self) -> List[ConversationSummary]:
        """List all stored summaries.

        Returns:
            List of all summaries
        """
        summaries = []
        try:
            for filename in os.listdir(self.storage_dir):
                if filename.endswith('.json'):
                    thread_id = filename[:-5]  # Remove .json extension
                    summary = self.load_summary(thread_id)
                    if summary:
                        summaries.append(summary)
        except Exception as e:
            logger.error(f"Error listing summaries: {str(e)}")

        # Sort by creation date (newest first)
        summaries.sort(key=lambda x: x.created_at, reverse=True)
        return summaries

    def search_summaries(self, query: str) -> List[ConversationSummary]:
        """Search summaries by content.

        Args:
            query: Search query

        Returns:
            List of matching summaries
        """
        results = []
        query = query.lower()

        for summary in self.list_summaries():
            # Search in title, topics, and summary text
            if (query in summary.title.lower() or
                query in summary.summary_text.lower() or
                any(query in topic.lower() for topic in summary.main_topics) or
                any(query in tag.lower() for tag in summary.tags)):
                results.append(summary)

        return results

    def delete_summary(self, thread_id: str) -> bool:
        """Delete a summary.

        Args:
            thread_id: Thread ID to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = os.path.join(self.storage_dir, f"{thread_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Deleted summary for thread {thread_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting summary: {str(e)}")
            return False