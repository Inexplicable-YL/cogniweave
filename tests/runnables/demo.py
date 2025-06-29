import shutil
import sys
import textwrap
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from cogniweave.core.end_detector import EndDetector
from cogniweave.core.history_stores import BaseHistoryStore as HistoryStore
from cogniweave.core.time_splitter import TimeSplitter
from cogniweave.core.vector_stores import TagsVectorStore
from cogniweave.llms import OpenAIEmbeddings, StringSingleTurnChat
from cogniweave.prompts import MessageSegmentsPlaceholder, RichSystemMessagePromptTemplate
from cogniweave.runnables.end_detector import RunnableWithEndDetector
from cogniweave.runnables.history_store import RunnableWithHistoryStore
from cogniweave.runnables.memory_maker import RunnableWithMemoryMaker

warnings.filterwarnings("ignore")

embeddings = OpenAIEmbeddings()

db_path = Path("./.cache/history_cache/test1.sqlite")
history_store = HistoryStore(db_url=f"sqlite:///{db_path}")

vector_store = TagsVectorStore(
    folder_path="./.cache/model_cache",
    index_name="test1",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
    auto_save=True,
)

agent = StringSingleTurnChat(
    lang="zh",
    provider="deepseek",
    model="deepseek-chat",
    contexts=[
        RichSystemMessagePromptTemplate.from_template(
            [
                "你是一个AI助手，你叫CogniWeave，你的任务是回答用户的问题。\n",
                MessageSegmentsPlaceholder(variable_name="long_memory"),
            ]
        ),
        MessagesPlaceholder(variable_name="history", optional=True),
    ],
)
runnable_with_memory_maker = RunnableWithMemoryMaker(
    agent,
    history_store=history_store,
    vector_store=vector_store,
    input_messages_key="input",
    history_messages_key="history",
    short_memory_key="short_memory",
    long_memory_key="long_memory",
)
runnable_with_end_detector = RunnableWithEndDetector(
    runnable_with_memory_maker,
    end_detector=EndDetector(),
    default={"output": []},
    history_messages_key="history",
)
runnable_with_history = RunnableWithHistoryStore(
    runnable_with_end_detector,
    history_store=history_store,
    time_splitter=TimeSplitter(),
    input_messages_key="input",
    history_messages_key="history",
)


console = Console()


def get_input(prompt: str = "> ") -> str | None:
    console.print(f"{prompt}", style="bold blink bright_cyan", end="")
    try:
        input_msg = input()
    except (KeyboardInterrupt, EOFError):
        sys.stdout.write("\033[K")
        return None

    if input_msg.strip().lower() == "exit":
        return None

    # 计算总显示行数（包括 prompt + 内容，自动换行考虑）
    term_width = shutil.get_terminal_size().columns
    wrapped_lines = textwrap.wrap(prompt + input_msg, width=term_width)
    num_lines = len(wrapped_lines)

    # 回退光标
    sys.stdout.write("\033[F" * num_lines)

    return input_msg


def print_input(input_val: str) -> None:
    user_bubble = Panel(input_val, style="bold #63bbd0", expand=False)
    console.print(Align.right(user_bubble))


def print_output(output_val: str) -> None:
    assistant_bubble = Panel(Markdown(output_val), style="#83cbac", expand=False)
    console.print(Align.left(assistant_bubble))


nearly_history = history_store.get_session_history("foo", limit=10)
for hist in nearly_history:
    if isinstance(hist, HumanMessage):
        print_input(str(hist.content))
    elif isinstance(hist, AIMessage):
        print_output(str(hist.content))

while True:
    input_msg = get_input()

    if not input_msg:
        break

    print_input(input_msg)

    chunks: Iterator[dict[str, Any]] = runnable_with_history.stream(
        {"input": input_msg},
        config={"configurable": {"session_id": "foo"}},
    )

    text_buffer = ""

    with Live("", console=console, refresh_per_second=8, transient=True) as live:
        for chunk in chunks:
            text = chunk if isinstance(chunk, str) else chunk.get("output", "")
            if text:
                text_buffer += text
                assistant_bubble = Panel(Markdown(text_buffer), style="#83cbac", expand=False)
                live.update(Align.left(assistant_bubble))

    if text_buffer:
        print_output(text_buffer)
