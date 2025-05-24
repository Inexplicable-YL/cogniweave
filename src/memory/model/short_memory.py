from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.load import Serializable

from src.utils import format_datetime_relative


class ChatSnippet(Serializable):
    timestamp: datetime
    chat_summary: str
    topic_tags: list[str] = Field(default_factory=list)

    def format(self, time: datetime | None = None) -> str:
        """Format the chat snippet for display."""
        time_str = format_datetime_relative(self.timestamp, time)
        return f"[{time_str}]\n{self.chat_summary}"


"""class RecentSchedule(BaseModel):
    timestamp: datetime
    description: str  # “你刚刚去跑步”
    status: Literal["finished", "ongoing", "cancelled"]
    activity_type: str | None = None  # “运动”，“会议”等


class HotTopic(BaseModel):
    topic: str  # 关键词，如“考研”、“AI大模型”
    last_mentioned: datetime
    local_relevance: bool = False  # 是否与某人聊天时频繁提及
    global_trending: bool = False  # 外部热点（可结合news API）"""


class ShortTermMemory(BaseModel):
    chat_snippets: list[ChatSnippet] = Field(default_factory=list)
    """recent_schedules: list[RecentSchedule] = Field(default_factory=list)
    hot_topics: list[HotTopic] = Field(default_factory=list)"""
