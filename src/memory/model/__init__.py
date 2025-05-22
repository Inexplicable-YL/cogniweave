from pydantic import BaseModel, Field

from .long_memory import LongTermMemory
from .mid_memory import MidTermMemory
from .short_memory import ShortTermMemory


class UserMemory(BaseModel):
    user_id: str
    short_term: ShortTermMemory = Field(default_factory=ShortTermMemory)
    mid_term: MidTermMemory = Field(default_factory=MidTermMemory)
    long_term: LongTermMemory = Field(default_factory=LongTermMemory)
