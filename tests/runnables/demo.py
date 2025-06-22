import warnings
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.tools import tool
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

from cogniweave.core.end_detector import EndDetector
from cogniweave.core.time_splitter import TimeSplitter
from cogniweave.history_store import BaseHistoryStore as HistoryStore
from cogniweave.llms import AgentBase
from cogniweave.runnables.end_detector import RunnableWithEndDetector
from cogniweave.runnables.history_store import RunnableWithHistoryStore

warnings.filterwarnings("ignore")


@tool(description="get datetime")
def get_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


db_path = Path("test.sqlite")

agent = AgentBase(
    lang="zh",
    model="gpt-4.1-mini",
    contexts=[MessagesPlaceholder(variable_name="history", optional=True)],
    tools=[get_datetime],
)
runnable_with_end_detector = RunnableWithEndDetector(
    agent,
    end_detector=EndDetector(),
    default={"output": []},
    history_messages_key="history",
)
runnable_with_history = RunnableWithHistoryStore(
    runnable_with_end_detector,
    history_store=HistoryStore(db_url=f"sqlite:///{db_path}"),
    time_splitter=TimeSplitter(),
    input_messages_key="input",
    history_messages_key="history",
)
console = Console()


while True:
    input_msg = Prompt.ask("[bold cyan]Input[/bold cyan]")
    if input_msg.lower() == "exit":
        break
    chunks: Iterator[dict[str, Any]] = runnable_with_history.stream(
        {"input": input_msg},
        config={"configurable": {"session_id": "foo"}},
    )

    next_conversation = False
    for chunk in chunks:
        text: str = chunk if isinstance(chunk, str) else chunk.get("output", "")
        if text:
            if not next_conversation:
                console.print("[bold green]Output[/bold green]: ", end="", style="bright_white")
                next_conversation = True
            console.print(Markdown(text), end="", soft_wrap=True, highlight=False)

    if next_conversation:
        console.print()
        console.print("[dim]────────────────────────────[/dim]")
