import os
import time
from typing import Any

import anyio
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from cogniweave.core.memory_maker.summary import SummaryMemoryMaker
from cogniweave.core.prompts.long_memory import LongMemoryPromptTemplate
from cogniweave.core.prompts.short_memory import ShortMemoryPromptTemplate
from cogniweave.core.vector_stores import TagsVectorStore
from cogniweave.history_stores import BaseHistoryStore
from cogniweave.llms import OpenAIEmbeddings

console = Console()

load_dotenv()


def check_api_key() -> bool:
    """Ensure OPENAI_API_KEY exists."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print(
            Panel(
                "[bold red]Error: OPENAI_API_KEY not found[/]\nCreate a .env file and set OPENAI_API_KEY=your_api_key_here",
                title="Environment Check",
                border_style="red",
            )
        )
        return False
    console.print(f"[bold green]✓ Found API Key: {api_key[:10]}...[/]")
    return True


def create_block1() -> list[Any]:
    return [
        HumanMessage("我最近在学习 React，遇到了 useEffect 的问题。"),
        AIMessage("useEffect 确实是 React 中比较复杂的概念。你遇到了什么具体问题?"),
        HumanMessage("就是不知道什么时候该用依赖数组。"),
        AIMessage(
            "依赖数组决定了 effect 何时重新执行。空数组表示只在组件挂载时执行一次，有依赖项则在依赖项变化时重新执行。"
        ),
    ]


def create_block2() -> list[Any]:
    return [
        HumanMessage("我刚画了一张初音未来，给你看看。"),
        AIMessage("哇，好漂亮的画！你画得真细腻，特别是头发的光泽处理得很好。"),
        HumanMessage("谢谢，我还担心你不喜欢呢。"),
        AIMessage("我真的很喜欢！你的画技太棒了，能看出你在细节上花了很多心思。"),
    ]


def create_block3() -> list[Any]:
    return [
        HumanMessage("我最近開始學習 Python 程式設計"),
        AIMessage("太好了! Python 是一個很棒的程式語言，適合初學者。你有什麼特別想學的方向嗎?"),
        HumanMessage("我對機器學習很有興趣，想用 Python 來做 AI 專案"),
        AIMessage(
            "機器學習確實是 Python 的強項！建議你可以從 scikit-learn 開始，然後再學習 TensorFlow 或 PyTorch。"
        ),
    ]


def create_block4() -> list[Any]:
    return [
        HumanMessage("最近在看一部日剧，非常好看。"),
        AIMessage("是什么剧? 推荐我看看呗。"),
    ]


def _add_block(
    store: BaseHistoryStore, session_id: str, block_id: str, start_ts: float, messages: list[Any]
) -> None:
    pairs = [(msg, start_ts + i * 0.1) for i, msg in enumerate(messages)]
    store.add_messages(pairs, block_id=block_id, block_ts=start_ts, session_id=session_id)


def prepare_history(store: BaseHistoryStore, session_id: str) -> None:
    """Insert test messages into history store."""
    store.add_user_name("演示用户", session_id=session_id)
    ts = time.time()
    _add_block(store, session_id, "b1", ts, create_block1())
    _add_block(store, session_id, "b2", ts + 100, create_block2())
    _add_block(store, session_id, "b3", ts + 200, create_block3())


def print_memory(title: str, memory: ShortMemoryPromptTemplate | LongMemoryPromptTemplate) -> None:
    table = Table(title=title, show_header=False)
    table.add_column("字段", style="cyan")
    table.add_column("值")
    if isinstance(memory, ShortMemoryPromptTemplate):
        table.add_row("时间戳记", str(memory.timestamp))
        table.add_row("标签", ", ".join(memory.topic_tags))
    else:
        table.add_row("更新时间", memory.updated_time)
        table.add_row("更新块ID", memory.updated_block_id)
    console.print(Panel(table, border_style="blue"))
    console.print(Panel(memory.format(), title="Formatted", border_style="magenta"))


def test_sync(maker: SummaryMemoryMaker, session_id: str) -> bool:
    console.print(
        Panel("Step 1: Test synchronous summary", style="bold green", border_style="green")
    )
    try:
        maker.invoke({"session_id": session_id})
        console.print("[bold green]✓ Synchronous version executed successfully![/]")
        block_ids = maker.history_store.get_session_block_ids(session_id)
        if len(block_ids) >= 2:
            short = maker.history_store.get_short_memory(block_ids[-2])
            if short:
                print_memory(f"Short Memory of {block_ids[-2]}", short)
        long_mem = maker.history_store.get_long_memory(session_id)
        if long_mem:
            print_memory("Long Memory", long_mem)
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
        return False
    return True


async def test_async(maker: SummaryMemoryMaker, session_id: str) -> bool:
    console.print(
        Panel("Step 2: Test asynchronous summary", style="bold green", border_style="green")
    )
    try:
        await maker.ainvoke({"session_id": session_id})
        console.print("[bold green]✓ Asynchronous version executed successfully![/]")
        block_ids = await maker.history_store.aget_session_block_ids(session_id)
        if len(block_ids) >= 2:
            short = await maker.history_store.aget_short_memory(block_ids[-2])
            if short:
                print_memory(f"Short Memory of {block_ids[-2]}", short)
        long_mem = await maker.history_store.aget_long_memory(session_id)
        if long_mem:
            print_memory("Long Memory", long_mem)
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
        return False
    return True


def print_stats(maker: SummaryMemoryMaker) -> None:
    table = Table(title="Performance Statistics", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("Language used", maker.lang)
    short_chain = type(maker.short_maker.memory_chain).__name__ if maker.short_maker else "None"
    long_chain = type(maker.long_maker.extract_chain).__name__ if maker.long_maker else "None"
    table.add_row("Short memory chain", short_chain)
    table.add_row("Long memory chain", long_chain)
    console.print(Panel(table, border_style="magenta"))


async def main() -> None:
    if not check_api_key():
        return
    console.print(
        Panel.fit(
            "[bold]Start Testing SummaryMemoryMaker[/]", border_style="bold magenta", padding=(1, 4)
        )
    )
    history_store = BaseHistoryStore(db_url="sqlite:///./.cache/history_cache/summary_demo.sqlite")
    vector_store = TagsVectorStore(
        folder_path="./.cache/model_cache",
        index_name="summary_demo",
        embeddings=OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,
        auto_save=True,
    )
    session_id = "demo_session"
    prepare_history(history_store, session_id)
    maker = SummaryMemoryMaker(history_store=history_store, vector_store=vector_store, lang="zh")
    if not test_sync(maker, session_id):
        return
    _add_block(history_store, session_id, "b4", time.time() + 300, create_block4())
    if not await test_async(maker, session_id):
        return
    console.print(
        Panel.fit(
            "[bold green]✓ All tests completed! SummaryMemoryMaker is working properly.[/]",
            border_style="bold green",
            padding=(1, 4),
        )
    )
    print_stats(maker)


if __name__ == "__main__":
    anyio.run(main)
