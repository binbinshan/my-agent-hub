一个基于大模型的聊天代理框架，支持：
- 大模型返回流式输出（streaming）以降低延迟并实时呈现响应。
- 会话记忆功能（持久/短期记忆）以提升对话连贯性和上下文保持。

## 主要特性
- 流式响应：模型生成的回复以数据块方式逐步输出，便于前端实时渲染。
- 记忆管理：支持会话记忆的读取、写入与清理策略，便于长期任务保持上下文。
- 可扩展工具节点：项目中包含工具节点示例（如 `TavilySearchToolNode.py`）。
- 单元测试与 CI：自带测试目录，GitHub Actions 已配置为在推送与 PR 时运行测试。

## 目录结构（重点）
- `chatbot/`
  - `chatbot.py` — 主入口，包含聊天逻辑、流式输出与记忆调用点
  - `TavilySearchToolNode.py` — 工具节点
  - `__pycache__/` — 字节码缓存
- `tests/` — 测试用例（若存在）
- `.github/workflows/ci.yml` — CI 配置（使用 Python 3.13，运行 pytest）

## 环境与依赖
- 要求：Python 3.13（CI 使用同版本）
- 在 macOS 上开发与在 PyCharm 中运行均已验证。
- 安装依赖：
  ```bash
  pip install -r requirements.txt