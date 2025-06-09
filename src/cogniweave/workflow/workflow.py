from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from cogniweave.core.end_detector import ConversationEndDetector
from cogniweave.core.memory.updater import (
    LongTermMemoryMaker,
    ShortTermMemoryMaker,
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


class ChatState(BaseModel):
    name: str
    messages: list[Any]
    timestamp: float
    short_memory: str | None = None
    long_memory: str | None = None
    is_complete: bool = False
    short_summary: str | None = None
    short_tags: list[str] | None = None
    short_memory_persisted: bool = False
    new_long_memory_items: str | None = None
    updated_long_memory: str | None = None


def build_memory_graph(lang: Literal["zh", "en"] = "zh") -> CompiledStateGraph:
    end_detector = ConversationEndDetector(lang=lang)
    short_term_maker = ShortTermMemoryMaker(lang=lang)
    long_term_maker = LongTermMemoryMaker(lang=lang)

    def check_complete(state: ChatState) -> ChatState:
        result = end_detector.invoke({"messages": state.messages})
        state.is_complete = result
        return state

    def make_short_summary(state: ChatState) -> ChatState:
        prompt = short_term_maker.invoke(
            {"name": state.name, "history": state.messages, "timestamp": state.timestamp}
        )
        state.short_summary = prompt.chat_summary
        return state

    def make_short_tags(state: ChatState) -> ChatState:
        prompt = short_term_maker.invoke(
            {"name": state.name, "history": state.messages, "timestamp": state.timestamp}
        )
        state.short_tags = prompt.topic_tags
        return state

    def write_dummy_short(state: ChatState) -> ChatState:
        """Dummy persistence step for short-term memory."""
        state.short_memory_persisted = True
        return state

    def extract_long_memory(state: ChatState) -> ChatState:
        """Extract new long-term memory items without merging."""
        state.new_long_memory_items = long_term_maker.extract({"history": state.messages})
        return state

    def update_long_memory(state: ChatState) -> ChatState:
        """Merge and update long-term memory."""
        state.updated_long_memory = long_term_maker.invoke({"history": state.messages})
        return state

    builder = StateGraph(ChatState)
    builder.add_node("check_complete", check_complete)
    builder.add_node("short_summary_node", make_short_summary)
    builder.add_node("short_tags_node", make_short_tags)
    builder.add_node("write_short_node", write_dummy_short)
    builder.add_node("extract_long_memory_node", extract_long_memory)
    builder.add_node("update_long_memory_node", update_long_memory)

    builder.set_entry_point("check_complete")
    # parallel short-term and long-term memory pipelines
    builder.add_edge("check_complete", "short_summary_node")
    builder.add_edge("check_complete", "short_tags_node")
    builder.add_edge("check_complete", "extract_long_memory_node")

    # short-term flow: summary and tags in parallel -> write -> END
    builder.add_edge("short_summary_node", "write_short_node")
    builder.add_edge("short_tags_node", "write_short_node")
    builder.add_edge("write_short_node", END)

    # long-term flow: extract -> update -> END
    builder.add_edge("extract_long_memory_node", "update_long_memory_node")
    builder.add_edge("update_long_memory_node", END)

    return builder.compile()
