import os
from dotenv import load_dotenv
from typing import Annotated
from datetime import datetime

from langchain_core.messages import BaseMessage, SystemMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import trim_messages
from typing import Literal
from chatbot.TavilySearchToolNode import TavilySearchToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# 定义状态类型，继承自 TypedDict，并使用 add_messages 函数将消息追加到现有列表
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 创建一个状态图对象，传入状态定义
graph_builder = StateGraph(state_schema=State)

# 定义 Tavily 搜索工具，最大搜索结果数设置为 2
tool = TavilySearch(max_results=2)

# 初始化一个 deepseek-chat 模型
chat_model = ChatOpenAI(model="deepseek-chat",
                        api_key=os.environ.get('DEEPSEEK_API_KEY'),
                        base_url="https://api.deepseek.com").bind_tools([tool])


# 定义聊天机器人的节点函数，接收当前状态并返回更新的消息列表
def chatbot(state: State):
    messages = state["messages"]

    print(f"原始消息数量: {len(messages)}")

    # 因记忆原因需限制token数量，按 token 数量限制（假设平均每条消息 100 tokens）
    trimmed_messages = trim_messages(
        messages=messages,
        max_tokens=10,
        token_counter=len,
        strategy="last",
        include_system=True,
        start_on="human",
        allow_partial=False
    )

    print(f"裁剪后消息数量: {len(trimmed_messages)}")

    # 调用模型
    response = chat_model.invoke(trimmed_messages)
    return {"messages": [response]}


# 定义路由函数，检查工具调用
def route_tools(state: State, ) -> Literal["tools", "__end__"]:
    """
    使用条件边来检查最后一条消息中是否有工具调用。
    参数:
    state: 状态字典或消息列表，用于存储当前对话的状态和消息。
    返回:
    如果最后一条消息包含工具调用，返回 "tools" 节点，表示需要执行工具调用；
    否则返回 "__end__"，表示直接结束流程。
    """
    # 检查状态是否是列表类型（即消息列表），取最后一条 AI 消息
    if isinstance(state, list):
        ai_message = state[-1]
    # 否则从状态字典中获取 "messages" 键，取最后一条消息
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    # 如果没有找到消息，则抛出异常
    else:
        raise ValueError(f"输入状态中未找到消息: {state}")

    # 检查最后一条消息是否有工具调用请求
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"  # 如果有工具调用请求，返回 "tools" 节点
    return "__end__"  # 否则返回 "__end__"，流程结束


# 第一个参数是唯一的节点名称，第二个参数是每次节点被调用时的函数或对象
tool_node = TavilySearchToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("chatbot", chatbot)
# 当工具调用完成后，返回到聊天机器人节点以继续对话
graph_builder.add_edge("tools", "chatbot")
# 指定从 START 节点开始，进入聊天机器人节点
graph_builder.add_edge(START, "chatbot")
# 添加条件边，判断是否需要调用工具
graph_builder.add_conditional_edges(
    "chatbot",  # 从聊天机器人节点开始
    route_tools,  # 路由函数，决定下一个节点
    {
        "tools": "tools",
        "__end__": "__end__"
    },  # 定义条件的输出，工具调用走 "tools"，否则走 "__end__"
)

# 创建内存检查点
memory = MemorySaver()

graph = graph_builder.compile(checkpointer=memory,  interrupt_before=["tools"])
with open("graph_diagram.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())
# 开始一个简单的聊天循环

while True:
    # 获取用户输入
    user_input = input("User: ")

    # 如果用户输入 "quit"、"exit" 或 "q"，则退出循环，结束对话
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")  # 打印告别语
        break  # 退出循环

    config = {"configurable": {"thread_id": "1"}}
    events = graph.stream(
        {"messages": [("user", user_input)]},  # 第一个参数传入用户的输入消息，消息格式为 ("user", "输入内容")
        config,  # 第二个参数用于指定线程配置，包含线程 ID
        stream_mode="values"  # stream_mode 设置为 "values"，表示返回流式数据的值
    )

    # 遍历每个事件，并打印最后一条消息的内容
    for event in events:
        # 通过 pretty_print 打印最后一条消息的内容
        event["messages"][-1].pretty_print()

    # 使用 graph.stream 处理用户输入，并生成机器人的回复
    # "messages" 列表中包含用户的输入，传递给对话系统
    # for event in graph.stream({"messages": [("user", user_input)]}):
    #
    #     # 遍历 event 的所有值，检查是否是 BaseMessage 类型的消息
    #     for value in event.values():
    #         if isinstance(value["messages"][-1], BaseMessage):
    #             # 如果消息是 BaseMessage 类型，则打印机器人的回复
    #             print("Assistant:", value["messages"][-1].content)
