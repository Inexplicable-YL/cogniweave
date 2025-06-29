from __future__ import annotations

import argparse
import shutil
import sys
import textwrap
import warnings
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from cogniweave.core.history_stores import BaseHistoryStore as HistoryStore
from cogniweave.quickstart import DEF_FOLDER_PATH, build_pipeline

if TYPE_CHECKING:
    from collections.abc import Iterator


warnings.filterwarnings("ignore")


def _get_input(prompt: str = "> ") -> str | None:
    console = Console()
    console.print(f"{prompt}", style="bold blink bright_cyan", end="")
    try:
        input_msg = input()
    except (KeyboardInterrupt, EOFError):
        sys.stdout.write("\033[K")
        return None

    if input_msg.strip().lower() == "exit":
        return None

    term_width = shutil.get_terminal_size().columns
    wrapped = textwrap.wrap(prompt + input_msg, width=term_width)
    sys.stdout.write("\033[F" * len(wrapped))
    return input_msg


def _print_input(console: Console, message: str) -> None:
    bubble = Panel(message, style="bold #63bbd0", expand=False)
    console.print(Align.right(bubble))


def _print_output(console: Console, message: str) -> None:
    bubble = Panel(Markdown(message), style="#83cbac", expand=False)
    console.print(Align.left(bubble))


def demo(session_id: str) -> None:
    pipeline = build_pipeline()
    history_store = HistoryStore(db_url=f"sqlite:///{DEF_FOLDER_PATH}")
    console = Console()

    nearly_history = history_store.get_session_history(session_id, limit=10)
    for hist in nearly_history:
        if isinstance(hist, HumanMessage):
            _print_input(console, str(hist.content))
        elif isinstance(hist, AIMessage):
            _print_output(console, str(hist.content))

    while True:
        input_msg = _get_input()
        if not input_msg:
            break

        _print_input(console, input_msg)
        chunks: Iterator[dict[str, Any]] = pipeline.stream(
            {"input": input_msg},
            config={"configurable": {"session_id": session_id}},
        )

        text_buffer = ""
        with Live("", console=console, refresh_per_second=8, transient=True) as live:
            for chunk in chunks:
                text = chunk if isinstance(chunk, str) else chunk.get("output", "")
                if text:
                    text_buffer += text
                    bubble = Panel(Markdown(text_buffer), style="#83cbac", expand=False)
                    live.update(Align.left(bubble))

        if text_buffer:
            _print_output(console, text_buffer)


def main() -> None:
    parser = argparse.ArgumentParser(description="CogniWeave CLI")
    sub = parser.add_subparsers(dest="command")

    demo_cmd = sub.add_parser("demo", help="Run interactive demo")
    demo_cmd.add_argument("session", nargs="?", default="demo")

    args = parser.parse_args()

    if args.command == "demo":
        demo(args.session)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
