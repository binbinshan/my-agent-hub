# 🤖 AI Agent Hub

一个基于 LangGraph 的多智能体框架，支持可扩展的 AI 代理开发和部署。

## 📁 项目结构

```
my-agent-hub/
├── README.md              # 项目总体说明（本文件）
├── requirements.txt       # Python 依赖
├── main.py               # 主程序入口
├── CLAUDE.md             # Claude Code 开发指南
│
└── chatbot/              # 💬 智能对话机器人
    ├── README.md         # ChatBot 模块详细说明
    ├── __init__.py
    ├── chatbot.py        # 核心对话逻辑
    ├── TavilySearchToolNode.py  # 工具执行节点
    ├── config.py         # 配置管理
    ├── logger.py         # 日志系统
    ├── utils.py          # 工具函数
    └── summary.py        # 对话摘要功能
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
# DeepSeek API 配置
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Tavily 搜索 API（用于网络搜索功能）
TAVILY_API_KEY=your_tavily_api_key_here

# 可选配置
MODEL_NAME=deepseek-chat
BASE_URL=https://api.deepseek.com
MAX_TOKENS=4000
TEMPERATURE=0.7
```

### 3. 运行程序

```bash
# 运行主程序
python main.py
```

## 📖 模块说明

### 💬 ChatBot - 智能对话机器人

基于 LangGraph 的对话系统，具备以下特性：

- 🔍 **网络搜索**：集成 Tavily 搜索引擎，获取实时信息
- 💾 **记忆管理**：使用 LangGraph MemorySaver 保存对话历史
- 📝 **自动摘要**：智能生成对话摘要，方便回顾
- 🔧 **工具调用**：支持扩展更多工具和功能
- 🌊 **流式响应**：实时显示生成过程

[查看 ChatBot 详细说明 →](chatbot/README.md)

## 🔧 开发指南

详细的开发说明请参考：
- [CLAUDE.md](./CLAUDE.md) - Claude Code 使用指南
- [ChatBot 开发文档](chatbot/README.md) - 模块详细说明

## 🤝 贡献指南

欢迎添加新的 Agent 模块！建议的目录结构：

```
your-agent/
├── README.md         # 模块说明文档
├── __init__.py
├── agent.py          # 核心逻辑
├── config.py         # 配置管理
└── tools/            # 工具目录
```

## 📄 许可证

MIT License

## 🙏 致谢

- [LangGraph](https://python.langchain.com/docs/langgraph) - 构建强大的 AI 工作流
- [DeepSeek](https://www.deepseek.com/) - 强大的语言模型服务
- [Tavily](https://tavily.com/) - 实时搜索 API