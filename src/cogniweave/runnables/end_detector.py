from __future__ import annotations

import inspect
from collections.abc import Sequence
from itertools import takewhile
from types import GenericAlias
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.runnables.base import Runnable, RunnableBindingBase
from langchain_core.utils.pydantic import create_model_v2
from pydantic import BaseModel, PrivateAttr

from cogniweave.core.end_detector import EndDetector  # noqa: TC001

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig

MessageLikeType = str | BaseMessage | Sequence[BaseMessage] | dict[str, Any]
MessageLikeWithTimeType = (
    tuple[str, float]
    | tuple[BaseMessage, float]
    | Sequence[tuple[BaseMessage, float]]
    | dict[str, Any]
)


class RunnableWithEndDetector(RunnableBindingBase):
    end_detector: EndDetector

    input_messages_key: str | None = None
    history_messages_key: str | None = None
    _default: MessageLikeType | MessageLikeWithTimeType = PrivateAttr()

    def __init__(
        self,
        runnable: Runnable[dict[str, Any], MessageLikeType | MessageLikeWithTimeType],
        end_detector: EndDetector,
        default: MessageLikeType | MessageLikeWithTimeType,
        *,
        input_messages_key: str | None = None,
        history_messages_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        lambda_end_detector = RunnableLambda(self._end_detect, self._a_end_detect)
        bound = RunnableBranch(
            (lambda_end_detector, runnable),
            lambda _: default,
        )
        super().__init__(
            bound=bound,
            end_detector=end_detector,
            input_messages_key=input_messages_key,
            history_messages_key=history_messages_key,
            **kwargs,
        )
        self._default = default

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        from langchain_core.messages import BaseMessage

        fields: dict = {}
        if self.input_messages_key and self.history_messages_key:
            fields[self.input_messages_key] = (
                tuple[str, float] | tuple[BaseMessage, float] | Sequence[tuple[BaseMessage, float]],
                ...,
            )
        elif self.input_messages_key:
            fields[self.input_messages_key] = (Sequence[tuple[BaseMessage, float]], ...)
        else:
            return create_model_v2(
                "RunnableWithChatHistoryInput",
                module_name=self.__class__.__module__,
                root=(Sequence[tuple[BaseMessage, float]], ...),
            )
        return create_model_v2(
            "RunnableWithChatHistoryInput",
            field_definitions=fields,
            module_name=self.__class__.__module__,
        )

    @override
    def get_output_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        """Get a pydantic model that can be used to validate output to the Runnable.

        Runnables that leverage the configurable_fields and configurable_alternatives
        methods will have a dynamic output schema that depends on which
        configuration the Runnable is invoked with.

        This method allows to get an output schema for a specific configuration.

        Args:
            config: A config to use when generating the schema.

        Returns:
            A pydantic model that can be used to validate output.
        """
        root_type = self.OutputType

        if (
            inspect.isclass(root_type)
            and not isinstance(root_type, GenericAlias)
            and issubclass(root_type, BaseModel)
        ):
            return root_type

        return create_model_v2(
            "RunnableWithChatHistoryOutput",
            root=root_type,
            module_name=self.__class__.__module__,
        )

    def _get_messages_with_timestamps(
        self,
        input_val: tuple[str, float]
        | tuple[BaseMessage, float]
        | Sequence[tuple[BaseMessage, float]],
    ) -> list[tuple[BaseMessage, float]]:
        # If value is a list or tuple...
        result: list[tuple[BaseMessage, float]] = []
        if isinstance(input_val, list) and len(input_val) != 0:
            # If is a list of list, then return the first value
            # This occurs for chat models - since we batch inputs
            if isinstance(input_val[0], list):
                if len(input_val) != 1:
                    msg = f"Expected a single list of messages. Got {input_val}."
                    raise ValueError(msg)
                input_val = input_val[0]
            if any(
                isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], float)  # noqa: PLR2004
                for item in input_val
            ):
                result = list(input_val)
        elif (
            isinstance(input_val, tuple)
            and len(input_val) == 2  # noqa: PLR2004
            and isinstance(input_val[1], (float, int))
        ):
            if isinstance(input_val[0], str):
                from langchain_core.messages import HumanMessage

                result = [(HumanMessage(content=input_val[0]), float(input_val[1]))]
            if isinstance(input_val[0], BaseMessage):
                result = [(input_val[0], float(input_val[1]))]
        if result:
            return sorted(result, key=lambda x: x[1])
        msg = f"Expected tuple[str, float], tuple[BaseMessage, float], or list[tuple[BaseMessage, float]]. Got {input_val}."
        raise ValueError(msg)

    def _get_input_messages(
        self,
        input_val: MessageLikeType | MessageLikeWithTimeType,
    ) -> list[BaseMessage]:
        # If dictionary, try to pluck the single key representing messages
        result: list[BaseMessage] = []
        if isinstance(input_val, dict):
            if self.input_messages_key:
                key = self.input_messages_key
            elif len(input_val) == 1:
                key = next(iter(input_val.keys()))
            else:
                key = "input"
            input_val = input_val[key]
        if isinstance(input_val, str):
            from langchain_core.messages import HumanMessage

            result = [HumanMessage(content=input_val)]
        # If value is a single message, convert to a list
        if isinstance(input_val, BaseMessage):
            result = [input_val]
        if isinstance(input_val, (list, tuple)):
            if len(input_val) == 0:
                return []
            if any(isinstance(item, BaseMessage) for item in input_val):
                result = list(input_val)  # type: ignore
            else:
                return [item for item, _ in self._get_messages_with_timestamps(input_val)]  # type: ignore
        if result:
            return result
        msg = (
            f"Expected str, BaseMessage, list[BaseMessage], or tuple[BaseMessage]. Got {input_val}."
        )
        raise ValueError(msg)

    def _get_user_messages(
        self,
        input_val: MessageLikeType | MessageLikeWithTimeType,
    ) -> list[BaseMessage]:
        # If value is a single message, convert to a list
        input_messages = self._get_input_messages(input_val)
        # If dictionary, try to pluck the single key representing messages
        if isinstance(input_val, dict) and self.history_messages_key:
            input_messages = (
                self._get_input_messages(input_val.get(self.history_messages_key, []))
                + input_messages
            )
        rev = input_messages[::-1]
        users_tail = list(takewhile(lambda m: isinstance(m, HumanMessage), rev))
        if not users_tail:
            return []
        k = len(users_tail)
        ai_block = list(takewhile(lambda m: isinstance(m, AIMessage), rev[k:]))
        k += len(ai_block)
        users_prev = list(takewhile(lambda m: isinstance(m, HumanMessage), rev[k:]))
        seq = users_tail if not users_prev else users_tail + ai_block + users_prev
        return seq[::-1]

    def _end_detect(
        self,
        input: dict[str, Any],
    ) -> bool:
        user_messages = self._get_user_messages(input)
        return self.end_detector.invoke({"messages": user_messages})

    async def _a_end_detect(
        self,
        input: dict[str, Any],
    ) -> bool:
        user_messages = self._get_user_messages(input)
        return await self.end_detector.ainvoke({"messages": user_messages})
