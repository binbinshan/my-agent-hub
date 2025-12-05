"""Test script for summary functionality."""
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot import ChatBot, ChatConfig
from chatbot.summary import SummaryGenerator, SummaryStorage


def test_summary_generation():
    """Test the summary generation functionality."""
    print("=" * 60)
    print("测试对话摘要功能")
    print("=" * 60)

    # Initialize chatbot
    try:
        bot = ChatBot()
        print("✓ ChatBot 初始化成功")
    except Exception as e:
        print(f"✗ ChatBot 初始化失败: {str(e)}")
        return

    # Test thread ID
    thread_id = f"test_thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Simulate a conversation
    print(f"\n📝 开始模拟对话 (Thread: {thread_id})")
    print("-" * 40)

    # Sample conversation messages
    test_messages = [
        "你好，我想了解一下机器学习",
        "机器学习是人工智能的一个分支，它让计算机能够从数据中学习规律。",
        "能详细介绍一下监督学习吗？",
        "监督学习是机器学习的一种方法，使用带标签的数据进行训练。",
        "有哪些常见的监督学习算法？",
        "常见的算法包括线性回归、逻辑回归、决策树、随机森林等。",
        "决策树是如何工作的？",
        "决策树通过一系列问题将数据分割成不同的类别。",
        "能给我一个决策树的例子吗？",
        "比如判断是否出去玩：先看天气如何，再看温度，最后看是否有空。",
        "这个例子很清楚！",
        "是的，决策树很直观，易于理解和解释。",
        "那决策树的缺点是什么呢？",
        "容易过拟合，对数据变化敏感，可能需要剪枝。",
        "过拟合是什么意思？",
        "过拟合指模型在训练数据上表现很好，但在新数据上表现差。",
        "如何避免过拟合？",
        "可以通过剪枝、限制树的深度、增加更多数据等方法。",
        "感谢你的详细解释！",
        "不客气！还有其他问题吗？"
    ]

    # Send messages and get responses
    message_count = 0
    for msg in test_messages:
        message_count += 1
        print(f"用户 [{message_count}]: {msg}")

        # Get response (simulated)
        result = bot.chat(msg, thread_id=thread_id, auto_summarize=False)
        response = result["response"][:50] + "..." if len(result["response"]) > 50 else result["response"]
        print(f"助手: {response}\n")

    print(f"\n📊 对话完成，共 {message_count} 条消息")
    print("-" * 40)

    # Generate summary manually
    print("\n🔍 生成对话摘要...")
    try:
        summary = bot.generate_summary(thread_id)
        if summary:
            print("✓ 摘要生成成功！")
            print(f"\n标题: {summary.title}")
            print(f"摘要: {summary.summary_text}")
            print(f"主要话题: {', '.join(summary.main_topics)}")
            print(f"关键点: {len(summary.key_points)} 个")
            print(f"用户目标: {len(summary.user_goals)} 个")
            print(f"情感倾向: {summary.sentiment}")
            print(f"标签: {', '.join(summary.tags)}")
            print(f"消息数量: {summary.message_count}")
        else:
            print("✗ 摘要生成失败")
    except Exception as e:
        print(f"✗ 生成摘要时出错: {str(e)}")

    # Test auto-summarize
    print("\n🤖 测试自动摘要功能...")
    try:
        # Create a new thread for auto-test
        auto_thread_id = f"auto_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Send fewer messages (below threshold)
        for i, msg in enumerate(test_messages[:10], 1):
            bot.chat(msg, thread_id=auto_thread_id, auto_summarize=True)

        # Check if summary was generated
        auto_summary = bot.get_summary(auto_thread_id)
        if auto_summary:
            print("✓ 自动摘要已生成 (消息数: 10)")
        else:
            print("- 自动摘要未生成 (消息数未达到阈值)")

    except Exception as e:
        print(f"✗ 自动摘要测试出错: {str(e)}")

    # Test summary listing
    print("\n📋 测试摘要列表...")
    try:
        summaries = bot.list_all_summaries()
        print(f"✓ 找到 {len(summaries)} 个摘要")
        for s in summaries[:3]:  # Show first 3
            print(f"  - {s.title} ({s.thread_id})")
    except Exception as e:
        print(f"✗ 列出摘要时出错: {str(e)}")

    # Test summary search
    print("\n🔎 测试摘要搜索...")
    try:
        search_results = bot.search_summaries("机器")
        print(f"✓ 搜索 '机器' 找到 {len(search_results)} 个结果")
        for r in search_results:
            print(f"  - {r.title}: {r.main_topics}")
    except Exception as e:
        print(f"✗ 搜索摘要时出错: {str(e)}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


def test_direct_summary_generation():
    """Test summary generation directly."""
    print("\n\n" + "=" * 60)
    print("直接测试摘要生成器")
    print("=" * 60)

    try:
        config = ChatConfig.from_env()
        generator = SummaryGenerator(config)

        from langchain_core.messages import HumanMessage, AIMessage

        # Create test messages
        messages = [
            HumanMessage(content="我想学习Python编程"),
            AIMessage(content="Python是一门很好的编程语言，适合初学者。"),
            HumanMessage(content="我应该从哪里开始？"),
            AIMessage(content="建议从基础语法开始，然后练习简单的项目。"),
            HumanMessage(content="有什么好的学习资源吗？"),
            AIMessage(content= "推荐官方文档、在线教程和实战项目。")
        ]

        summary = generator.generate_summary(messages, "direct_test")

        print(f"标题: {summary.title}")
        print(f"摘要: {summary.summary_text}")
        print(f"话题: {', '.join(summary.main_topics)}")

    except Exception as e:
        print(f"✗ 直接测试失败: {str(e)}")


if __name__ == "__main__":
    # Run tests
    test_summary_generation()
    test_direct_summary_generation()