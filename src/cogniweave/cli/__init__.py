from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from cogniweave.config import init_config
from cogniweave.quickstart import DEF_FOLDER_PATH, build_pipeline

if TYPE_CHECKING:
    from collections.abc import Iterator


warnings.filterwarnings("ignore")


def _get_input() -> str | None:
    console = Console()
    console.print("> ", style="bold blink bright_cyan", end="")
    try:
        input_msg = input()
    except (KeyboardInterrupt, EOFError):
        sys.stdout.write("\033[K")
        return None

    if input_msg.strip().lower() == "exit":
        return None
    return input_msg


def _print_input(console: Console, message: str) -> None:
    console.print("> ", style="bold blink bright_cyan", end="")
    console.print(message)


def _print_output(console: Console, message: str) -> None:
    bubble = Panel(Markdown(message), style="#83cbac", expand=False)
    console.print(Align.left(bubble))


def demo(
    session_id: str,
    *,
    index: str = "demo",
    folder: str | Path = DEF_FOLDER_PATH,
    config_file: str | Path | None = None,
) -> None:
    """Run the interactive demo."""

    if config_file:
        init_config(_config_file=config_file)
    else:
        init_config()

    pipeline = build_pipeline(index_name=index, folder_path=folder)
    history_store = pipeline.history_store
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

        with console.status("[#83cbac]Processing...", spinner_style="#83cbac"):
            time.sleep(3)
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
    demo_cmd.add_argument("session", nargs="?", default="demo", help="Session identifier")
    demo_cmd.add_argument(
        "--index",
        default="demo",
        help="Index name for history and vector store",
    )
    demo_cmd.add_argument(
        "--folder",
        default=str(DEF_FOLDER_PATH),
        help="Folder used to store cache files",
    )
    demo_cmd.add_argument(
        "--config-file",
        help="Path to a file containing the system prompt",
        default=None,
    )

    args = parser.parse_args()

    if args.command == "demo":
        demo(
            args.session,
            index=args.index,
            folder=Path(args.folder),
            config_file=args.config_file,
        )
    else:
        parser.print_help()
