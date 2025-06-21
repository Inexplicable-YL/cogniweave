import warnings
from datetime import datetime
from pathlib import Path

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.tools import tool
from rich.console import Console
from rich.prompt import Prompt

from cogniweave.core.end_detector import EndDetector
from cogniweave.core.time_splitter import TimeSplitter
from cogniweave.history_store import BaseHistoryStore as HistoryStore
from cogniweave.llms import AgentBase
from cogniweave.runnables.end_detector import RunnableWithEndDetector
from cogniweave.runnables.history import RunnableWithHistory

warnings.filterwarnings("ignore")


@tool(description="get datetime")
def get_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


db_path = Path("test.sqlite")

chat = AgentBase(
    lang="zh",
    model="gpt-4.1-mini",
    contexts=[MessagesPlaceholder(variable_name="history")],
    tools=[get_datetime],
)
runnable_with_end_detector = RunnableWithEndDetector(
    chat,
    end_detector=EndDetector(),
    default=[],
    history_messages_key="history",
)
runnable = RunnableWithHistory(
    runnable_with_end_detector,
    history_store=HistoryStore(db_url=f"sqlite:///{db_path}"),
    time_splitter=TimeSplitter(),
    input_messages_key="input",
    history_messages_key="history",
    auto_package=True,
)

console = Console()

while True:
    input_msg = Prompt.ask("[bold cyan]Input[/bold cyan]")
    if input_msg.lower() == "exit":
        break

    result = runnable.invoke({"input": input_msg}, config={"configurable": {"session_id": "foo"}})
    if result:
        console.print("[bold green]Output[/bold green]:", result["output"], style="bright_white")
        console.print("[dim]────────────────────────────[/dim]")
