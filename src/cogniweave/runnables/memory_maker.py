from __future__ import annotations

import hashlib
import inspect
import pickle
from collections.abc import Generator, Sequence
from types import GenericAlias
from typing import TYPE_CHECKING, Any, Literal, cast
from typing_extensions import override

import anyio
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages.utils import convert_to_openai_messages
from langchain_core.runnables.base import Runnable, RunnableBindingBase, RunnableLambda
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    Output,
    get_unique_config_specs,
)
from langchain_core.utils.pydantic import create_model_v2
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from cogniweave.core.memory_maker import SummaryMemoryMaker
from cogniweave.core.prompt_values.long_memory import LongTermMemoryPromptValue
from cogniweave.core.prompt_values.short_memory import ShortTermMemoryPromptValue
from cogniweave.core.vector_stores import TagsVectorStore  # noqa: TC001
from cogniweave.history_stores import BaseHistoryStore  # noqa: TC001
from cogniweave.prompts import RichHumanMessagePromptTemplate

if TYPE_CHECKING:
    from langchain_core.prompts.string import StringPromptTemplate
    from langchain_core.runnables.config import RunnableConfig
    from langchain_core.tracers.schemas import Run

MessageLikeType = str | BaseMessage | Sequence[BaseMessage] | dict[str, Any]
MessageLikeWithTimeType = (
    tuple[str, float]
    | tuple[BaseMessage, float]
    | Sequence[tuple[BaseMessage, float]]
    | dict[str, Any]
)


class RunnableWithMemoryMaker(RunnableBindingBase):
    lang: Literal["zh", "en"] = Field(default="zh")

    history_store: BaseHistoryStore
    vector_store: TagsVectorStore[str]

    input_messages_key: str | None = None
    history_messages_key: str | None = None
    short_memory_key: str | None = None
    long_memory_key: str | None = None
    session_factory_config: Sequence[ConfigurableFieldSpec]

    _memory_maker: SummaryMemoryMaker = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        runnable: Runnable[dict[str, Any], MessageLikeType | MessageLikeWithTimeType],
        history_store: BaseHistoryStore,
        vector_store: TagsVectorStore,
        *,
        input_messages_key: str | None = None,
        history_messages_key: str | None = None,
        short_memory_key: str | None = None,
        long_memory_key: str | None = None,
        session_factory_config: Sequence[ConfigurableFieldSpec] | None = None,
        auto_package: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize a RunnableWithMessageHistory."""
        if auto_package:
            runnable = (
                runnable
                if isinstance(runnable, RunnableLambda)
                else RunnableLambda(runnable.invoke, runnable.ainvoke)
            )
        short_memory_chain: Runnable = RunnableLambda(
            self._enter_short_memory, self._aenter_short_memory
        ).with_config(run_name="load_short_memory")
        if input_messages_key:
            short_memory_chain = RunnablePassthrough.assign(
                **{input_messages_key: short_memory_chain}
            ).with_config(run_name="insert_short_memory")

        long_memory_chain: Runnable = RunnableLambda(
            self._enter_long_memory, self._aenter_long_memory
        ).with_config(run_name="load_long_memory")
        if long_memory_key:
            long_memory_chain = RunnablePassthrough.assign(
                **{long_memory_key: long_memory_chain}
            ).with_config(run_name="insert_long_memory")

        runnable_sync: Runnable = runnable.with_listeners(on_end=self._exit_memory)
        runnable_async: Runnable = runnable.with_alisteners(on_end=self._aexit_memory)

        def _call_runnable_sync(_input: Any) -> Runnable:
            return runnable_sync

        async def _call_runnable_async(_input: Any) -> Runnable:
            return runnable_async

        bound: Runnable = (
            short_memory_chain
            | long_memory_chain
            | RunnableLambda(
                _call_runnable_sync,
                _call_runnable_async,
            ).with_config(run_name="check_sync_or_async")
        ).with_config(run_name="RunnableWithMessageHistory")

        if session_factory_config:
            _config_specs = session_factory_config
        else:
            # If not provided, then we'll use the default session_id field
            _config_specs = [
                ConfigurableFieldSpec(
                    id="session_id",
                    annotation=str,
                    name="Session ID",
                    description="Unique identifier for a session.",
                    default="",
                    is_shared=True,
                ),
            ]

        super().__init__(
            history_store=history_store,
            vector_store=vector_store,
            bound=bound,
            input_messages_key=input_messages_key,
            history_messages_key=history_messages_key,
            short_memory_key=short_memory_key,
            long_memory_key=long_memory_key,
            session_factory_config=_config_specs,
            **kwargs,
        )
        self._history_chain = short_memory_chain
        self._memory_maker = SummaryMemoryMaker(
            lang=self.lang, history_store=self.history_store, vector_store=self.vector_store
        )

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        """Get the configuration specs for the RunnableWithMessageHistory."""
        return get_unique_config_specs(super().config_specs + list(self.session_factory_config))

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

    @property
    @override
    def OutputType(self) -> type[Output]:
        return self._history_chain.OutputType

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

    def _get_input_messages(
        self, input_val: str | BaseMessage | Sequence[BaseMessage] | dict
    ) -> list[BaseMessage]:
        from langchain_core.messages import BaseMessage

        # If dictionary, try to pluck the single key representing messages
        if isinstance(input_val, dict):
            if self.input_messages_key:
                key = self.input_messages_key
            elif len(input_val) == 1:
                key = next(iter(input_val.keys()))
            else:
                key = "input"
            input_val = input_val[key]

        # If value is a string, convert to a human message
        if isinstance(input_val, str):
            from langchain_core.messages import HumanMessage

            return [HumanMessage(content=input_val)]
        # If value is a single message, convert to a list
        if isinstance(input_val, BaseMessage):
            return [input_val]
        # If value is a list or tuple...
        if isinstance(input_val, (list, tuple)):
            # Handle empty case
            if len(input_val) == 0:
                return list(input_val)
            # If is a list of list, then return the first value
            # This occurs for chat models - since we batch inputs
            if isinstance(input_val[0], list):
                if len(input_val) != 1:
                    msg = f"Expected a single list of messages. Got {input_val}."
                    raise ValueError(msg)
                return input_val[0]
            return list(input_val)
        msg = (
            f"Expected str, BaseMessage, list[BaseMessage], or tuple[BaseMessage]. Got {input_val}."
        )
        raise ValueError(msg)

    def _get_message_content(self, message: BaseMessage) -> str | list[dict[str, Any]]:
        content = cast("dict", convert_to_openai_messages(message))["content"]
        return cast("str | list[dict[str, Any]]", content)

    def _enter_short_memory(self, value: Any, config: RunnableConfig) -> list[BaseMessage]:
        session_id: str = config["configurable"]["_unique_session_id"]  # type: ignore

        input_val = value if not self.input_messages_key else value[self.input_messages_key]
        input_messages = raw_messages = self._get_input_messages(input_val)

        if not self.history_messages_key:
            historic_messages = self.history_store.get_session_history(session_id)
            input_messages = raw_messages[len(historic_messages) :]

        messages: list[BaseMessage] = []
        for input_message in input_messages:
            prompt_template: RichHumanMessagePromptTemplate | None = None
            if isinstance(input_message, HumanMessage) and isinstance(input_message.content, str):
                short_memory_ids = [
                    data.content
                    for data in self.vector_store.similarity_search(
                        input_message.content,
                        k=2,
                        filter={"session_id": session_id},
                        score_threshold=2,
                        extract_high_score=True,
                    )
                ]
                short_memorys = [
                    *ShortTermMemoryPromptValue().to_messages(lang=self.lang),
                    "<ChatMemory>",
                    *[
                        result
                        for block_id in short_memory_ids
                        if (result := self.history_store.get_short_memory(block_id)) is not None
                    ],
                    "</ChatMemory>",
                ]
                content = self._get_message_content(input_message)
                if isinstance(content, str):
                    prompt_template = RichHumanMessagePromptTemplate.from_template(
                        [*short_memorys, content]
                    )
                if isinstance(content, list):
                    prompt_template = RichHumanMessagePromptTemplate.from_template(
                        [*short_memorys, *content]
                    )
            if prompt_template is not None:
                messages.append(prompt_template.format(**value))
            else:
                messages.append(input_message)

        if not self.history_messages_key:
            messages = raw_messages[: len(historic_messages)] + messages  # type: ignore

        return messages

    async def _aenter_short_memory(
        self, value: dict[str, Any], config: RunnableConfig
    ) -> list[BaseMessage]:
        session_id: str = config["configurable"]["_unique_session_id"]  # type: ignore

        input_val = value if not self.input_messages_key else value[self.input_messages_key]
        input_messages = raw_messages = self._get_input_messages(input_val)

        if not self.history_messages_key:
            historic_messages = await self.history_store.aget_session_history(session_id)
            input_messages = raw_messages[len(historic_messages) :]

        processed_message_map: dict[int, BaseMessage] = {}

        async def _process_one(idx: int, input_msg: BaseMessage) -> None:
            if isinstance(input_msg, HumanMessage) and isinstance(input_msg.content, str):
                short_memory_ids = [
                    data.content
                    for data in await self.vector_store.asimilarity_search(
                        input_msg.content,
                        k=2,
                        filter={"session_id": session_id},
                        score_threshold=2,
                        extract_high_score=True,
                    )
                ]
                if not short_memory_ids:
                    return
                short_memorys = [
                    *ShortTermMemoryPromptValue().to_messages(lang=self.lang),
                    "<ChatMemory>",
                    *[
                        result
                        for block_id in short_memory_ids
                        if (result := self.history_store.get_short_memory(block_id)) is not None
                    ],
                    "</ChatMemory>",
                ]
                content = self._get_message_content(input_message)
                prompt_template: RichHumanMessagePromptTemplate | None = None
                if isinstance(content, str):
                    prompt_template = RichHumanMessagePromptTemplate.from_template(
                        [*short_memorys, content]
                    )
                if isinstance(content, list):
                    prompt_template = RichHumanMessagePromptTemplate.from_template(
                        [*short_memorys, *content]
                    )
                if prompt_template is not None:
                    processed_message_map[idx] = prompt_template.format(**value)

        async with anyio.create_task_group() as tg:
            for idx, input_msg in enumerate(input_messages):
                tg.start_soon(_process_one, idx, input_msg)

        messages: list[BaseMessage] = []
        for idx, input_message in enumerate(input_messages):
            if idx in processed_message_map:
                input_message = processed_message_map[idx]
            messages.append(input_message)

        if not self.history_messages_key:
            messages = raw_messages[: len(historic_messages)] + messages  # type: ignore

        return messages

    def _enter_long_memory(
        self, _: dict[str, Any], config: RunnableConfig
    ) -> list[str | StringPromptTemplate]:
        session_id: str = config["configurable"]["_unique_session_id"]  # type: ignore

        long_memory = self.history_store.get_long_memory(session_id)
        if long_memory is not None:
            return [
                *cast(
                    "Generator[str | StringPromptTemplate]",
                    LongTermMemoryPromptValue().to_messages(lang=self.lang),
                ),
                "<LongMemory>",
                long_memory,
                "</LongMemory>",
            ]
        return []

    async def _aenter_long_memory(
        self, _: dict[str, Any], config: RunnableConfig
    ) -> list[str | StringPromptTemplate]:
        session_id: str = config["configurable"]["_unique_session_id"]  # type: ignore

        long_memory = await self.history_store.aget_long_memory(session_id)
        if long_memory is not None:
            return [
                *cast(
                    "Generator[str | StringPromptTemplate]",
                    LongTermMemoryPromptValue().to_messages(lang=self.lang),
                ),
                "<LongMemory>",
                long_memory,
                "</LongMemory>",
            ]
        return []

    def _exit_memory(self, _: Run, config: RunnableConfig) -> None:
        session_id: str = config["configurable"]["_unique_session_id"]  # type: ignore
        self._memory_maker.invoke({"session_id": session_id})

    async def _aexit_memory(self, _: Run, config: RunnableConfig) -> None:
        session_id: str = config["configurable"]["_unique_session_id"]  # type: ignore
        await self._memory_maker.ainvoke({"session_id": session_id})

    @override
    def _merge_configs(self, *configs: RunnableConfig | None) -> RunnableConfig:
        config = super()._merge_configs(*configs)
        expected_keys = [field_spec.id for field_spec in self.session_factory_config]

        configurable = config.get("configurable", {})

        missing_keys = set(expected_keys) - set(configurable.keys())

        if missing_keys:
            example_input = {self.input_messages_key: "foo"}
            example_configurable = dict.fromkeys(missing_keys, "[your-value-here]")
            example_config = {"configurable": example_configurable}
            msg = (
                f"Missing keys {sorted(missing_keys)} in config['configurable'] "
                f"Expected keys are {sorted(expected_keys)}."
                f"When using via .invoke() or .stream(), pass in a config; "
                f"e.g., chain.invoke({example_input}, {example_config})"
            )
            raise ValueError(msg)

        session_id: str = ""
        if len(expected_keys) == 1:
            session_id = str(configurable[expected_keys[0]])
        else:
            # otherwise verify that names of keys patch and invoke by named arguments
            session_id = hashlib.sha256(
                pickle.dumps(tuple(sorted(configurable[key] for key in expected_keys)))
            ).hexdigest()
        config["configurable"]["_unique_session_id"] = session_id  # type: ignore
        return config
