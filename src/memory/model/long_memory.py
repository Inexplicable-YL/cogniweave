from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SocialBond(BaseModel):
    user_id: str
    relationship_type: Literal["friend", "close_friend", "acquaintance", "stranger"]
    closeness_score: float = Field(ge=0, le=1.0)
    trust_score: float = Field(ge=0, le=1.0)
    last_updated: datetime
    notes: str | None = None  # 人设印象，如“他喜欢分析问题，但情绪不稳定”


class LongTermEvent(BaseModel):
    timestamp: datetime
    description: str  # 例如：“去年和C旅游经历非常愉快”
    involved_users: list[str] = Field(default_factory=list)
    long_term_effect: str  # “加深关系”、“角色变化”、“情绪记忆”


class KnowledgeItem(BaseModel):
    acquired_at: datetime
    domain: str  # “人际关系”、“生活经验”、“编程技巧”
    content: str  # 例如：“原来很多人只是在倾诉时需要倾听”
    source: str | None = None  # “从用户B的聊天中学到的”


class LongTermMemory(BaseModel):
    social_bonds: list[SocialBond] = Field(default_factory=list)
    impactful_events: list[LongTermEvent] = Field(default_factory=list)
    acquired_knowledge: list[KnowledgeItem] = Field(default_factory=list)
