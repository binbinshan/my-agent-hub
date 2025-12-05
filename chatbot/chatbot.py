"""Main chatbot module with LangGraph workflow."""
import os
import sys
from typing import Annotated, Dict, List, Optional, Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    trim_messages
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from .config import ChatConfig
from .logger import logger, setup_logger
from .utils import count_tokens, sanitize_user_input
from .TavilySearchToolNode import ToolExecutionNode
from .summary import SummaryGenerator, SummaryStorage, ConversationSummary

# Load environment variables
load_dotenv()

# Set up module logger
logger = setup_logger("chatbot", level=os.getenv("LOG_LEVEL", "INFO"))


class ConversationState(TypedDict):
    """State for the conversation graph."""

    messages: Annotated[List[BaseMessage], add_messages]
    thread_id: Optional[str]
    metadata: Optional[Dict[str, Any]]
    summary: Optional[ConversationSummary]  # Conversation summary
    needs_summary: bool  # Flag to trigger summary generation


class ChatBot:
    """Enhanced chatbot with LangGraph workflow and tool support."""

    def __init__(self, config: Optional[ChatConfig] = None):
        """Initialize the chatbot.

        Args:
            config: Configuration settings. If None, loads from environment.
        """
        self.config = config or ChatConfig.from_env()
        self.config.validate()

        # Initialize components
        self._setup_tools()
        self._setup_model()
        self._setup_graph()
        self._setup_summary()

        logger.info(f"Initialized ChatBot with model: {self.config.model_name}")

    def _setup_tools(self) -> None:
        """Set up available tools."""
        self.tools: List[BaseTool] = []

        # Add Tavily search tool
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if tavily_api_key:
            search_tool = TavilySearch(
                max_results=self.config.max_search_results,
                api_key=tavily_api_key
            )
            self.tools.append(search_tool)
            logger.info("Added Tavily search tool")
        else:
            logger.warning("TAVILY_API_KEY not found, search tool disabled")

    def _setup_model(self) -> None:
        """Set up the language model."""
        self.model = ChatOpenAI(
            model=self.config.model_name,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=self.config.base_url,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )

        # Bind tools to model
        if self.tools:
            self.model = self.model.bind_tools(self.tools)
            logger.info(f"Bound {len(self.tools)} tools to model")

    def _setup_graph(self) -> None:
        """Set up the LangGraph workflow."""
        # Create graph builder
        self.graph_builder = StateGraph(state_schema=ConversationState)

        # Add nodes
        self.graph_builder.add_node("chatbot", self._chatbot_node)
        self.graph_builder.add_node("process_response", self._process_response_node)

        if self.tools:
            tool_node = ToolExecutionNode(tools=self.tools)
            self.graph_builder.add_node("tools", tool_node)
            # tools -> process_response (not back to chatbot)
            self.graph_builder.add_edge("tools", "process_response")

        # Add edges
        self.graph_builder.add_edge(START, "chatbot")

        # chatbot -> conditional routing
        if self.tools:
            self.graph_builder.add_conditional_edges(
                "chatbot",
                self._route_tools,
                {
                    "tools": "tools",
                    "process": "process_response"
                }
            )
        else:
            self.graph_builder.add_edge("chatbot", "process_response")

        # process_response -> END
        self.graph_builder.add_edge("process_response", END)

        # Compile with memory
        self.memory = MemorySaver()
        self.graph = self.graph_builder.compile(
            checkpointer=self.memory
        )

        logger.info("Compiled LangGraph workflow")

    def _setup_summary(self) -> None:
        """Set up summary generation and storage."""
        self.summary_generator = SummaryGenerator(self.config)
        self.summary_storage = SummaryStorage()
        logger.info("Initialized summary generator and storage")

    def _chatbot_node(self, state: ConversationState) -> Dict[str, Any]:
        """Process messages through the model.

        Args:
            state: Current conversation state

        Returns:
            Updated state with model response
        """
        messages = state["messages"]

        logger.debug(f"Processing {len(messages)} messages")

        # Trim messages to manage context length
        trimmed_messages = self._trim_messages(messages)

        try:
            response = self.model.invoke(trimmed_messages)
            logger.info("Generated model response")
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            # Return an error message
            error_response = AIMessage(
                content=f"Sorry, I encountered an error: {str(e)}",
                name="error"
            )
            return {"messages": [error_response]}

    def _trim_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Trim messages to fit within token limits.

        Args:
            messages: List of messages to trim

        Returns:
            Trimmed list of messages
        """
        # Don't trim at all for now - this is causing issues with tool calls
        # TODO: Implement proper message trimming that preserves tool context
        return messages

    def _route_tools(self, state: ConversationState) -> Literal["tools", "process"]:
        """Route to tools if the model made tool calls.

        Args:
            state: Current conversation state

        Returns:
            Next node name
        """
        messages = state.get("messages", [])
        if not messages:
            return "process"

        last_message = messages[-1]

        # If the last message is an AI message with tool calls, route to tools
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            logger.debug(f"Routing to tools: {len(last_message.tool_calls)} calls")
            return "tools"

        return "process"

    def _process_response_node(self, state: ConversationState) -> Dict[str, Any]:
        """Process the final response and handle tool results if any.

        Args:
            state: Current conversation state

        Returns:
            Updated state (no changes needed)
        """
        messages = state.get("messages", [])
        if not messages:
            return {}

        # If we have tool results, we need to generate a final response
        last_message = messages[-1]

        # Check if the last message is a tool result
        if isinstance(last_message, ToolMessage):
            logger.info("Processing tool results to generate final response")

            # Get all messages up to and including the tool result
            # Generate a final response based on tool results
            try:
                # Create a prompt that summarizes the tool results
                prompt_messages = []

                # Add the original conversation (excluding the last AI message with tool calls)
                for msg in messages[:-2]:
                    prompt_messages.append(msg)

                # Add a final message asking to summarize the tool results
                result_content = last_message.content
                prompt_messages.append(
                    HumanMessage(content=f"Based on the search results: {result_content}, please provide a helpful response to the user.")
                )

                # Generate final response
                response = self.model.invoke(prompt_messages)

                # Replace the tool result with the actual response
                messages[-1] = response

                logger.info("Generated final response from tool results")

            except Exception as e:
                logger.error(f"Error processing tool results: {str(e)}")
                # Create a fallback response
                fallback_response = AIMessage(
                    content=f"I found some information but encountered an error processing it: {str(e)}",
                    name="error"
                )
                messages[-1] = fallback_response

        return {"messages": messages}

    def chat(self, message: str, thread_id: Optional[str] = None,
              auto_summarize: bool = True) -> Dict[str, Any]:
        """Send a message and get response.

        Args:
            message: User message
            thread_id: Thread ID for conversation continuity
            auto_summarize: Whether to auto-summarize if conversation is long

        Returns:
            Dictionary with response and metadata
        """
        # Sanitize input
        sanitized_message = sanitize_user_input(message)
        if not sanitized_message:
            return {
                "response": "Please provide a valid message.",
                "metadata": {"error": "Invalid input"}
            }

        # Use default thread ID if none provided
        thread_id = thread_id or self.config.default_thread_id

        # Create config
        config = {"configurable": {"thread_id": thread_id}}

        logger.info(f"Processing message for thread {thread_id}")

        try:
            # Stream the response
            events = list(self.graph.stream(
                {"messages": [HumanMessage(content=sanitized_message)]},
                config=config,
                stream_mode="values"
            ))

            # Get the last message
            if events and "messages" in events[-1]:
                last_message = events[-1]["messages"][-1]
                response = last_message.content if hasattr(last_message, 'content') else str(last_message)

                metadata = {
                    "thread_id": thread_id,
                    "message_count": len(events[-1]["messages"]),
                    "tool_calls": len(last_message.tool_calls) if hasattr(last_message, 'tool_calls') else 0
                }

                # Auto-summarize if needed
                if auto_summarize:
                    summary = self.auto_summarize_if_needed(thread_id)
                    if summary:
                        metadata["summary_generated"] = True
                        metadata["summary"] = {
                            "title": summary.title,
                            "topics": summary.main_topics,
                            "sentiment": summary.sentiment
                        }

                return {
                    "response": response,
                    "metadata": metadata
                }
            else:
                return {
                    "response": "No response generated.",
                    "metadata": {"error": "No events"}
                }

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "response": f"Sorry, an error occurred: {str(e)}",
                "metadata": {"error": str(e)}
            }

    def stream_chat(self, message: str, thread_id: Optional[str] = None):
        """Stream chat responses.

        Args:
            message: User message
            thread_id: Thread ID for conversation continuity

        Yields:
            Response chunks
        """
        sanitized_message = sanitize_user_input(message)
        if not sanitized_message:
            yield "Error: Invalid message"
            return

        thread_id = thread_id or self.config.default_thread_id
        config = {"configurable": {"thread_id": thread_id}}

        logger.info(f"Streaming response for thread {thread_id}")

        try:
            for event in self.graph.stream(
                {"messages": [HumanMessage(content=sanitized_message)]},
                config=config,
                stream_mode="values"
            ):
                if "messages" in event and event["messages"]:
                    last_message = event["messages"][-1]
                    if hasattr(last_message, 'content') and last_message.content:
                        yield last_message.content
        except Exception as e:
            logger.error(f"Error in stream: {str(e)}")
            yield f"Error: {str(e)}"

    def get_conversation_history(self, thread_id: Optional[str] = None) -> List[BaseMessage]:
        """Get conversation history for a thread.

        Args:
            thread_id: Thread ID to retrieve history for

        Returns:
            List of messages in the conversation
        """
        thread_id = thread_id or self.config.default_thread_id
        config = {"configurable": {"thread_id": thread_id}}

        try:
            checkpoint = self.memory.get(config)
            if checkpoint and "messages" in checkpoint.get("channel_values", {}):
                return checkpoint["channel_values"]["messages"]
            return []
        except Exception as e:
            logger.error(f"Error retrieving history: {str(e)}")
            return []

    def clear_thread(self, thread_id: Optional[str] = None) -> bool:
        """Clear conversation history for a thread.

        Args:
            thread_id: Thread ID to clear

        Returns:
            True if successful, False otherwise
        """
        thread_id = thread_id or self.config.default_thread_id
        config = {"configurable": {"thread_id": thread_id}}

        try:
            # Delete the checkpoint
            self.memory.delete(config)
            logger.info(f"Cleared conversation for thread {thread_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing thread: {str(e)}")
            return False

    def save_graph_diagram(self, filepath: str = "graph_diagram.png") -> bool:
        """Save the workflow diagram.

        Args:
            filepath: Path to save the diagram

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, "wb") as f:
                f.write(self.graph.get_graph().draw_mermaid_png())
            logger.info(f"Saved graph diagram to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving diagram: {str(e)}")
            return False

    def generate_summary(self, thread_id: Optional[str] = None) -> Optional[ConversationSummary]:
        """Generate a summary for the conversation.

        Args:
            thread_id: Thread ID to summarize. If None, uses default.

        Returns:
            Generated summary or None if failed
        """
        thread_id = thread_id or self.config.default_thread_id

        # Get conversation history
        messages = self.get_conversation_history(thread_id)

        if not messages:
            logger.warning(f"No messages to summarize for thread {thread_id}")
            return None

        # Generate summary
        summary = self.summary_generator.generate_summary(messages, thread_id)

        # Save summary
        self.summary_storage.save_summary(summary)

        logger.info(f"Generated and saved summary for thread {thread_id}")
        return summary

    def get_summary(self, thread_id: Optional[str] = None) -> Optional[ConversationSummary]:
        """Get the summary for a conversation.

        Args:
            thread_id: Thread ID to get summary for

        Returns:
            Summary if exists, None otherwise
        """
        thread_id = thread_id or self.config.default_thread_id
        return self.summary_storage.load_summary(thread_id)

    def list_all_summaries(self) -> List[ConversationSummary]:
        """List all conversation summaries.

        Returns:
            List of all summaries
        """
        return self.summary_storage.list_summaries()

    def search_summaries(self, query: str) -> List[ConversationSummary]:
        """Search summaries by content.

        Args:
            query: Search query

        Returns:
            List of matching summaries
        """
        return self.summary_storage.search_summaries(query)

    def auto_summarize_if_needed(self, thread_id: Optional[str] = None,
                                message_threshold: int = 20) -> Optional[ConversationSummary]:
        """Automatically generate summary if conversation is long enough.

        Args:
            thread_id: Thread ID to check
            message_threshold: Number of messages after which to summarize

        Returns:
            Generated summary if threshold reached, None otherwise
        """
        thread_id = thread_id or self.config.default_thread_id

        # Check if summary already exists
        existing_summary = self.get_summary(thread_id)
        if existing_summary:
            return existing_summary

        # Check message count
        messages = self.get_conversation_history(thread_id)
        if len(messages) >= message_threshold:
            logger.info(f"Auto-generating summary for thread {thread_id} (message count: {len(messages)})")
            return self.generate_summary(thread_id)

        return None

    def update_summary(self, thread_id: Optional[str] = None) -> Optional[ConversationSummary]:
        """Update an existing summary with new conversation content.

        Args:
            thread_id: Thread ID to update summary for

        Returns:
            Updated summary or None if failed
        """
        # Delete old summary
        thread_id = thread_id or self.config.default_thread_id
        self.summary_storage.delete_summary(thread_id)

        # Generate new one
        return self.generate_summary(thread_id)


def main():
    """Main function to run the interactive chatbot."""
    logger.info("Starting chatbot...")

    # Initialize chatbot
    try:
        bot = ChatBot()

        # Save workflow diagram
        bot.save_graph_diagram()

        print("\n🤖 Chatbot started! Type 'quit', 'exit', or 'q' to exit.\n")

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\nGoodbye! 👋")
                    break

                if not user_input:
                    continue

                # Get response
                result = bot.chat(user_input)
                print(f"\nBot: {result['response']}\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except EOFError:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                logger.error(f"Error in conversation loop: {str(e)}")
                print(f"\nError: {str(e)}\n")

    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()