import asyncio
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from cogniweave.core.memory_makers.long_memory import LongTermMemoryMaker
from cogniweave.core.prompts.long_memory import LongMemoryPromptTemplate, _format_memory

console = Console()


def check_environment() -> bool:
    """Check required environment variables."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            Panel(
                "[bold red]Error: Please set OPENAI_API_KEY in .env file[/]",
                title="Environment Check",
                border_style="red",
            )
        )
        return False
    return True


def prepare_test_history() -> list[BaseMessage]:
    """Prepare test chat history for memory maker."""
    return [
        HumanMessage(content="我最近開始學習 Python 程式設計"),
        AIMessage(
            content="太好了！Python 是一個很棒的程式語言，適合初學者。你有什麼特別想學的方向嗎？"  # noqa: RUF001
        ),
        HumanMessage(content="我對機器學習很有興趣，想用 Python 來做 AI 專案"),
        AIMessage(
            content="機器學習確實是 Python 的強項！建議你可以從 scikit-learn 開始，然後再學習 TensorFlow 或 PyTorch。"
        ),
        HumanMessage(content="好的，我會先從基礎開始。另外我平常喜歡喝咖啡，特別是手沖咖啡"),
        AIMessage(
            content="手沖咖啡是個很棒的興趣！不同的沖泡方法會帶來不同的風味。你有最喜歡的咖啡豆產區嗎？"  # noqa: RUF001
        ),
        HumanMessage(content="我比較喜歡衣索比亞的豆子，酸味比較明亮"),
        AIMessage(
            content="衣索比亞確實以果香和花香聞名，特別是耶加雪菲。搭配淺烘焙會有很棒的酸質表現。"
        ),
    ]


def print_history(history: list[BaseMessage]) -> None:
    """Print formatted chat history."""
    table = Table(title="Prepared Chat History", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Role", min_width=10)
    table.add_column("Content", min_width=40)

    for i, msg in enumerate(history, 1):
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        table.add_row(str(i), f"[cyan]{role}[/]", str(msg.content))

    console.print(Panel(table, title="Test Data", border_style="blue"))


def test_extraction(memory_maker: LongTermMemoryMaker, history: list[BaseMessage]) -> bool:
    """Test memory extraction functionality."""
    console.print(Panel("Step 1: Test Memory Extraction", style="bold green", border_style="green"))
    try:
        extracted_list = memory_maker._extract(
            {"history": history},
            memory_maker._get_current_timestamp(),
            memory_maker._get_current_date(),
        )

        console.print("[bold]Original extraction result:[/]")
        console.print(Syntax(str(extracted_list), "python", theme="monokai"))

        console.print(
            f"\n[bold green]✓ Successfully extracted {len(extracted_list)} memory items:[/]"
        )
        console.print(
            Panel(_format_memory(extracted_list), title="Extracted Memories", border_style="blue")
        )
    except Exception as e:
        console.print(
            Panel(
                f"[bold red]Extraction failed: {e}\nError type: {type(e).__name__}[/]",
                title="Error",
                border_style="red",
            )
        )
        import traceback

        console.print(Syntax(traceback.format_exc(), "python", theme="monokai"))
        return False
    else:
        return True


def test_complete_process(
    memory_maker: LongTermMemoryMaker, history: list[BaseMessage]
) -> LongMemoryPromptTemplate | None:
    """Test complete memory update process."""
    console.print(
        Panel(
            "Step 2: Test Complete Memory Update Process", style="bold green", border_style="green"
        )
    )
    try:
        input_data = {
            "history": history,
            "current_long_term_memory": [
                "2025-06-17（2天前）得知用戶對程式設計有興趣，想學習新技術",
                "2025-06-16（3天前）得知用戶提到喜歡閱讀技術書籍",
            ],
            "last_update_time": "2025-06-17 15:30",
        }
        result = memory_maker.invoke(input_data)

        console.print("[bold green]✓ Memory updated successfully![/]")
        console.print(
            f"[bold]Number of updated memory items:[/] [cyan]{len(result.updated_memory)}[/]"
        )

        console.print(
            Panel(
                result.format(),
                title="Complete Memory Content",
                border_style="blue",
                subtitle=f"Total {len(result.updated_memory)} items",
            )
        )
    except Exception as e:
        console.print(
            Panel(
                f"[bold red]Complete process failed: {e}\nError type: {type(e).__name__}[/]",
                title="Error",
                border_style="red",
            )
        )
        import traceback

        console.print(Syntax(traceback.format_exc(), "python", theme="monokai"))
        return None
    else:
        return result


def test_async_complete_process(
    memory_maker: LongTermMemoryMaker, history: list[BaseMessage]
) -> LongMemoryPromptTemplate | None:
    """Test asynchronous version of memory update."""
    console.print(
        Panel(
            "Step 3: Test Asynchronous Complete Memory Update Process",
            style="bold green",
            border_style="green",
        )
    )
    try:
        input_data = {
            "history": history,
            "current_long_term_memory": [
                "2025-06-17（2天前）得知用戶對程式設計有興趣，想學習新技術",
                "2025-06-16（3天前）得知用戶提到喜歡閱讀技術書籍",
            ],
            "last_update_time": "2025-06-17 15:30",
        }

        async def test_async() -> LongMemoryPromptTemplate:
            return await memory_maker.ainvoke(input_data)

        async_result = asyncio.run(test_async())
        console.print("[bold green]✓ Asynchronous version executed successfully![/]")
        console.print(
            f"[bold]Number of asynchronous result memory items:[/] [cyan]{len(async_result.updated_memory)}[/]"
        )

        console.print(
            Panel(
                async_result.format(),
                title="Asynchronous complete Memory Content",
                border_style="blue",
                subtitle=f"Total {len(async_result.updated_memory)} items",
            )
        )
    except Exception as e:
        console.print(
            Panel(
                f"[bold red]Asynchronous complete process failed: {e}\nError type: {type(e).__name__}[/]",
                title="Error",
                border_style="red",
            )
        )
        import traceback

        console.print(Syntax(traceback.format_exc(), "python", theme="monokai"))
        return None
    else:
        return async_result


def print_stats(memory_maker: LongTermMemoryMaker) -> None:
    """Print performance statistics."""
    table = Table(title="Performance Statistics", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    table.add_row("Language used", memory_maker.lang)
    table.add_row("Extraction chain type", type(memory_maker.extract_chain).__name__)
    table.add_row("Merging chain type", type(memory_maker.chat_chain).__name__)

    console.print(Panel(table, border_style="magenta"))


def main() -> None:
    """Real LLM test LongTermMemoryMaker's complete process."""
    if not check_environment():
        return

    console.print(
        Panel.fit(
            "[bold]Start Testing LongTermMemoryMaker[/]",
            border_style="bold magenta",
            padding=(1, 4),
        )
    )

    history = prepare_test_history()
    memory_maker = LongTermMemoryMaker(lang="zh")
    print_history(history)

    if not test_extraction(memory_maker, history):
        return

    result = test_complete_process(memory_maker, history)
    if not result:
        return

    async_result = test_async_complete_process(memory_maker, history)
    if not async_result:
        return

    console.print(
        Panel.fit(
            "[bold green]✓ All tests completed! LongTermMemoryMaker is working properly.[/]",
            border_style="bold green",
            padding=(1, 4),
        )
    )
    print_stats(memory_maker)


if __name__ == "__main__":
    main()
