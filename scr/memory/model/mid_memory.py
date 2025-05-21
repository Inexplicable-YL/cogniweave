from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class InteractionEvent(BaseModel):
    timestamp: datetime
    user_id: str
    event_summary: str  # 例如：“他分享了生病的消息”
    emotional_impact: Literal["positive", "neutral", "negative"]
    affect_strength: float = Field(ge=0, le=1.0)  # 情绪强度：0无感 → 1强烈


class TemporaryEmotion(BaseModel):
    timestamp: datetime
    dominant_emotion: str  # 如“开心、困惑、愤怒、孤独”
    cause_summary: str  # 如“因为昨天和B争吵”
    fade_rate: float = 0.1  # 情绪衰减率（用于记忆管理器定期更新）


class MidTermMemory(BaseModel):
    recent_interactions: list[InteractionEvent] = Field(default_factory=list)
    temporary_emotions: list[TemporaryEmotion] = Field(default_factory=list)
