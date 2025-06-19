"""Script to test ShortTermMemoryMaker functionality."""

import os
from datetime import datetime
from typing import Any

import anyio
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from cogniweave.core.memory_makers.short_memory import ShortTermMemoryMaker
from cogniweave.core.prompts.short_memory import ShortMemoryPromptTemplate

console = Console()

load_dotenv()


def create_test_conversation() -> dict[str, Any]:
    """Create a test conversation."""
    return {
        "name": "言灵",
        "history": [
            HumanMessage("我最近在学习 React，遇到了 useEffect 的问题。"),
            AIMessage("useEffect 确实是 React 中比较复杂的概念。你遇到了什么具体问题？"),  # noqa: RUF001
            HumanMessage("就是不知道什么时候该用依赖数组。"),
            AIMessage(
                "依赖数组决定了 effect 何时重新执行。空数组表示只在组件挂载时执行一次，"
                "有依赖项则在依赖项变化时重新执行。"
            ),
            HumanMessage("原来如此！谢谢你的解释，我明白了。"),
            AIMessage("不客气！如果还有其他 React 相关的问题，随时可以问我。"),
        ],
        "timestamp": datetime.now().timestamp(),
    }


def create_art_conversation() -> dict[str, Any]:
    """Create another test conversation (art topic)."""
    return {
        "name": "小梦",
        "history": [
            HumanMessage("我刚画了一张初音未来，给你看看。"),
            AIMessage("哇，好漂亮的画！你画得真细腻，特别是头发的光泽处理得很好。"),
            HumanMessage("谢谢，我还担心你不喜欢呢。"),
            AIMessage("我真的很喜欢！你的画技太棒了，能看出你在细节上花了很多心思。"),
            HumanMessage("嘿嘿，我练习了好久呢。"),
        ],
        "timestamp": datetime.now().timestamp(),
    }


def print_history(conversation: dict[str, Any], title: str) -> None:
    """Print conversation history as a table."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("角色", min_width=10)
    table.add_column("内容", min_width=40)

    history = conversation.get("history", [])
    for i, msg in enumerate(history, 1):
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        table.add_row(str(i), f"[cyan]{role}[/]", str(msg.content))

    console.print(Panel(table, title="Test Data", border_style="blue"))


def print_memory_result(result: ShortMemoryPromptTemplate, title: str) -> None:
    """Format and display memory result."""
    table = Table(title=title, show_header=False)
    table.add_column("字段", style="cyan")
    table.add_column("值")
    table.add_row("时间戳记", str(result.timestamp))
    table.add_row("标签", ", ".join(result.topic_tags))

    console.print(Panel(table, border_style="blue"))
    console.print(Panel(result.format(), title="Formatted Memory", border_style="magenta"))


def check_api_key() -> bool:
    """Check if the API key is set."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print(
            Panel(
                "[bold red]Error: OPENAI_API_KEY not found[/]\nCreate a .env file in the project root and add:\nOPENAI_API_KEY=your_api_key_here",
                title="Environment Check",
                border_style="red",
            )
        )
        return False

    console.print(f"[bold green]✓ Found API Key: {api_key[:10]}...[/]")
    return True


async def test_sync_memory(
    memory_maker: ShortTermMemoryMaker,
) -> tuple[ShortMemoryPromptTemplate, ShortMemoryPromptTemplate] | None:
    """Test synchronous short-term memory generation."""
    console.print(
        Panel("Step 1: Test synchronous memory", style="bold green", border_style="green")
    )
    try:
        conv1 = create_test_conversation()
        print_history(conv1, "Conversation 1: React Learning")
        result1 = memory_maker.invoke(conv1)
        print_memory_result(result1, "Conversation 1: React Learning")

        conv2 = create_art_conversation()
        print_history(conv2, "Conversation 2: Art Sharing")
        result2 = memory_maker.invoke(conv2)
        print_memory_result(result2, "Conversation 2: Art Sharing")
    except Exception as e:
        console.print(
            Panel(
                f"[bold red]Synchronous test failed: {e}\nError type: {type(e).__name__}[/]",
                title="Error",
                border_style="red",
            )
        )
        import traceback

        console.print(Syntax(traceback.format_exc(), "python", theme="monokai"))
        return None
    else:
        return result1, result2


async def test_async_memory(memory_maker: ShortTermMemoryMaker) -> ShortMemoryPromptTemplate | None:
    """Test asynchronous short-term memory generation."""
    console.print(
        Panel("Step 2: Test asynchronous memory", style="bold green", border_style="green")
    )
    try:
        conv = create_test_conversation()
        print_history(conv, "Async Conversation")
        result = await memory_maker.ainvoke(conv)
        console.print("[bold green]✓ Asynchronous version executed successfully![/]")
        print_memory_result(result, "Async Result")
    except Exception as e:
        console.print(
            Panel(
                f"[bold red]Asynchronous test failed: {e}\nError type: {type(e).__name__}[/]",
                title="Error",
                border_style="red",
            )
        )
        import traceback

        console.print(Syntax(traceback.format_exc(), "python", theme="monokai"))
        return None
    else:
        return result


def print_stats(memory_maker: ShortTermMemoryMaker) -> None:
    """Print performance statistics."""
    table = Table(title="Performance Statistics", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    table.add_row("Language used", memory_maker.lang)
    chain_type = type(memory_maker.memory_chain).__name__ if memory_maker.memory_chain else "None"
    table.add_row("Summary chain type", chain_type)
    tags_chain_type = type(memory_maker.tags_chain).__name__ if memory_maker.tags_chain else "None"
    table.add_row("Tagging chain type", tags_chain_type)

    console.print(Panel(table, border_style="magenta"))


async def test_english_version() -> ShortMemoryPromptTemplate | None:
    """Test English version."""
    console.print(Panel("Step 3: Test English version", style="bold green", border_style="green"))
    english_conv = {
        "name": "James",
        "history": [
            HumanMessage("I just finished a drawing of Hatsune Miku. Want to see?"),
            AIMessage("Wow, it's so beautiful! Your lines are really delicate."),
            HumanMessage("Thanks, I was worried you might not like it."),
            AIMessage("I really love it—your skill is amazing."),
        ],
        "timestamp": datetime.now().timestamp(),
    }
    try:
        updater = ShortTermMemoryMaker(lang="en")
        print_history(english_conv, "English Conversation")
        result = updater.invoke(english_conv)
        console.print("[bold green]✓ English version executed successfully![/]")
        print_memory_result(result, "English Conversation: Art Sharing")
    except Exception as e:
        console.print(
            Panel(
                f"[bold red]English version test failed: {e}\nError type: {type(e).__name__}[/]",
                title="Error",
                border_style="red",
            )
        )
        import traceback

        console.print(Syntax(traceback.format_exc(), "python", theme="monokai"))
        return None
    else:
        return result


def test_error_handling() -> None:
    """Test error handling."""
    console.print(Panel("Step 4: Test error handling", style="bold green", border_style="green"))

    updater = ShortTermMemoryMaker(lang="zh")

    try:
        invalid_input = {
            "name": "测试用户",
            "history": "这不是一个列表",  # Should be a list
            "timestamp": datetime.now().timestamp(),
        }
        updater.invoke(invalid_input)
        console.print("x Expected an error but none was raised")
    except TypeError as e:
        console.print(f"[bold green]✓ Successfully caught type error: {e}[/]")
    except Exception as e:
        console.print(f"[bold red]x Caught unexpected error type: {type(e).__name__}: {e}[/]")

    try:
        missing_field = {
            "name": "测试用户",
            # 缺少 history
            "timestamp": datetime.now().timestamp(),
        }
        updater.invoke(missing_field)
        console.print("x Expected an error but none was raised")
    except TypeError as e:
        console.print(f"[bold green]✓ Successfully caught missing field error: {e}[/]")
    except Exception as e:
        console.print(f"[bold red]x Caught unexpected error type: {type(e).__name__}: {e}[/]")

    console.print("Error handling tests completed")


async def main() -> None:
    """Main test function."""

    console.print(
        Panel.fit(
            "[bold]Start Testing ShortTermMemoryMaker[/]",
            border_style="bold magenta",
            padding=(1, 4),
        )
    )

    if not check_api_key():
        return

    if not os.getenv("SHORT_MEMORY_MODEL"):
        os.environ["SHORT_MEMORY_MODEL"] = "openai/gpt-4.1-mini"
        console.print(f"[bold green]✓ Set default model: {os.environ['SHORT_MEMORY_MODEL']}[/]")

    memory_maker = ShortTermMemoryMaker(lang="zh")
    try:
        sync_results = await test_sync_memory(memory_maker)
        if not sync_results:
            return

        async_result = await test_async_memory(memory_maker)
        if not async_result:
            return

        english_result = await test_english_version()
        if not english_result:
            return

        test_error_handling()

        console.print(
            Panel.fit(
                "[bold green]✓ All tests completed! ShortTermMemoryMaker is working properly.[/]",
                border_style="bold green",
                padding=(1, 4),
            )
        )
        print_stats(memory_maker)
    except Exception as e:
        console.print(
            Panel(
                f"[bold red]Error occurred during tests: {e}[/]",
                title="Error",
                border_style="red",
            )
        )
        import traceback

        console.print(Syntax(traceback.format_exc(), "python", theme="monokai"))


if __name__ == "__main__":
    anyio.run(main)
