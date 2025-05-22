from typing import Any, Literal

from langchain.schema import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
)
from pydantic import Field, model_validator

from src.llms import SingleTurnChatBase
from src.prompts.summary import (
    SHORT_TERM_MEMORY_SUMMARY_EN,
    SHORT_TERM_MEMORY_SUMMARY_ZH,
)
from src.utils import get_model_from_env, get_provider_from_env


class ShortTermMemoryUpdater(SingleTurnChatBase):
    """Short-term memory updater for chat models."""

    lang: Literal["en", "zh"] = Field(default="zh")

    provider: str = Field(
        default_factory=get_provider_from_env("SHORT_MEMORY_MODEL", default="openai")
    )
    model_name: str = Field(
        default_factory=get_model_from_env("SHORT_MEMORY_MODEL", default="gpt-4.1-mini")
    )
    temperature: float = 0.7

    @model_validator(mode="before")
    @classmethod
    def process_prompt_with_lang(
        cls,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        """Set the prompt based on the language."""
        if values.get("prompt") is None:
            if values["lang"] == "zh":
                values["prompt"] = SystemMessagePromptTemplate.from_template(
                    SHORT_TERM_MEMORY_SUMMARY_ZH
                )
            else:
                values["prompt"] = SystemMessagePromptTemplate.from_template(
                    SHORT_TERM_MEMORY_SUMMARY_EN
                )
        return values

    def _format_history(
        self,
        history: list[BaseMessage],
    ) -> str:
        """Format the chat history for the model."""
        chat_history: list[HumanMessage | AIMessage] = [
            msg for msg in history if isinstance(msg, (HumanMessage, AIMessage))
        ]
        return "\n".join(
            [
                f"[User]: {msg.content}"
                if isinstance(msg, HumanMessage)
                else f"[Assistant]: {msg.content}"
                for msg in chat_history
            ]
        )

    def get_memory(self, name: str, history: list[BaseMessage], **kwargs: Any) -> str:
        """Get the short-term memory from the model."""
        message = f"UserName：{name}\nChatHistory:\n" + self._format_history(history)
        return self.invoke({"input": message} | kwargs)

    async def aget_memory(
        self, name: str, history: list[BaseMessage], **kwargs: Any
    ) -> str:
        """Asynchronously get the short-term memory from the model."""
        message = f"UserName：{name}\nChatHistory:\n" + self._format_history(history)
        return await self.ainvoke({"input": message} | kwargs)
