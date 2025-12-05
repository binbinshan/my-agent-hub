"""LangGraph 工作流中的工具执行节点模块。"""
import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, ToolMessage, AIMessage
from langchain_core.tools import BaseTool

from .logger import logger
from .utils import format_tool_result


class ToolExecutionNode:
    """工具执行节点类

    该节点负责检查最新的AI消息中的工具调用请求，并执行这些工具。
    支持并行工具执行和完善的错误处理机制。

    属性:
        tools_by_name: 工具名称到工具实例的映射字典
        max_parallel: 并行执行工具的最大数量
    """

    def __init__(self, tools: List[BaseTool], max_parallel: int = 5) -> None:
        """初始化工具执行节点

        参数:
            tools: 可用工具列表
            max_parallel: 并行执行工具的最大数量，默认为5个
        """
        self.tools_by_name: Dict[str, BaseTool] = {tool.name: tool for tool in tools}
        self.max_parallel = max_parallel
        logger.info(f"工具执行节点初始化完成，共加载 {len(tools)} 个工具")

        # 记录所有可用工具
        for tool_name in self.tools_by_name:
            logger.debug(f"可用工具: {tool_name}")

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, List[ToolMessage]]:
        """执行最后一条消息中的工具调用

        参数:
            inputs: 包含对话消息的字典，必须有 "messages" 键

        返回:
            包含工具执行结果的字典，结果以消息形式返回

        异常:
            ValueError: 如果未找到消息或最后一条消息中没有工具调用
        """
        messages = inputs.get("messages", [])
        if not messages:
            logger.error("输入中没有找到消息")
            raise ValueError("输入中没有找到消息")

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            logger.debug("最后一条消息不是AI消息，跳过工具执行")
            return {"messages": []}

        if not last_message.tool_calls:
            logger.debug("最后一条消息中没有工具调用请求")
            return {"messages": []}

        logger.info(f"开始执行 {len(last_message.tool_calls)} 个工具调用")

        outputs = []
        errors = []

        # 逐个执行工具调用
        for tool_call in last_message.tool_calls:
            try:
                result = self._execute_single_tool(tool_call)
                outputs.append(result)
                logger.info(f"工具 '{tool_call['name']}' 执行成功")
            except Exception as e:
                logger.error(f"执行工具 '{tool_call['name']}' 时出错: {str(e)}")
                error_message = self._create_error_message(tool_call, str(e))
                errors.append(error_message)

        all_outputs = outputs + errors
        logger.info(f"工具执行完成：成功 {len(outputs)} 个，失败 {len(errors)} 个")

        return {"messages": all_outputs}

    def _execute_single_tool(self, tool_call: Dict[str, Any]) -> ToolMessage:
        """执行单个工具调用

        参数:
            tool_call: 包含工具调用信息的字典

        返回:
            包含执行结果的 ToolMessage 对象

        异常:
            ValueError: 如果找不到指定的工具
            RuntimeError: 如果工具执行失败
        """
        tool_name = tool_call.get("name")
        if not tool_name or tool_name not in self.tools_by_name:
            raise ValueError(f"找不到工具 '{tool_name}'")

        tool = self.tools_by_name[tool_name]
        args = tool_call.get("args", {})

        logger.debug(f"准备执行工具 '{tool_name}'，参数: {args}")

        try:
            result = tool.invoke(args)
            formatted_result = format_tool_result(result)

            return ToolMessage(
                content=formatted_result,
                name=tool_name,
                tool_call_id=tool_call["id"],
            )
        except Exception as e:
            raise RuntimeError(f"工具执行失败: {str(e)}") from e

    def _create_error_message(self, tool_call: Dict[str, Any], error: str) -> ToolMessage:
        """为工具执行失败创建错误消息

        参数:
            tool_call: 失败的工具调用
            error: 错误信息

        返回:
            包含错误信息的 ToolMessage 对象
        """
        return ToolMessage(
            content=f"执行出错: {error}",
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
            is_error=True
        )


# 为了保持向后兼容性，创建别名
TavilySearchToolNode = ToolExecutionNode


# 导出的类列表
__all__ = ["ToolExecutionNode", "TavilySearchToolNode"]