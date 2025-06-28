from __future__ import annotations

import hashlib
import inspect
import pickle
import time
import uuid
from collections.abc import Sequence
from types import GenericAlias
from typing import (
    TYPE_CHECKING,
    Any,
)
from typing_extensions import override

from langchain_core.load.load import load
from langchain_core.messages.base import BaseMessage
from langchain_core.runnables.base import Runnable, RunnableBindingBase, RunnableLambda
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    Output,
    get_unique_config_specs,
)
from langchain_core.utils.pydantic import create_model_v2
from pydantic import BaseModel, PrivateAttr

from cogniweave.history_store import BaseHistoryStore  # noqa: TC001
from cogniweave.time_splitter.base import BaseTimeSplitter  # noqa: TC001

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig
    from langchain_core.tracers.schemas import Run

MessageLikeType = str | BaseMessage | Sequence[BaseMessage] | dict[str, Any]
MessageLikeWithTimeType = (
    tuple[str, float]
    | tuple[BaseMessage, float]
    | Sequence[tuple[BaseMessage, float]]
    | dict[str, Any]
)


class RunnableWithHistoryStore(RunnableBindingBase):
    history_store: BaseHistoryStore
    time_splitter: BaseTimeSplitter

    history_limit: int | None = None
    input_messages_key: str | None = None
    output_messages_key: str | None = None
    history_messages_key: str | None = None
    history_factory_config: Sequence[ConfigurableFieldSpec]

    _input_messages_cache: dict[str, list[tuple[BaseMessage, float]]] = PrivateAttr(
        default_factory=dict
    )

    def __init__(
        self,
        runnable: Runnable[dict[str, Any], MessageLikeType | MessageLikeWithTimeType],
        history_store: BaseHistoryStore,
        time_splitter: BaseTimeSplitter,
        *,
        history_limit: int | None = None,
        input_messages_key: str | None = None,
        output_messages_key: str | None = None,
        history_messages_key: str | None = None,
        history_factory_config: Sequence[ConfigurableFieldSpec] | None = None,
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
        history_chain: Runnable = RunnableLambda(
            self._enter_history, self._aenter_history
        ).with_config(run_name="load_history")
        messages_key = history_messages_key or input_messages_key
        if messages_key:
            history_chain = RunnablePassthrough.assign(**{messages_key: history_chain}).with_config(
                run_name="insert_history"
            )

        runnable_sync: Runnable = runnable.with_listeners(on_end=self._exit_history)
        runnable_async: Runnable = runnable.with_alisteners(on_end=self._aexit_history)

        def _call_runnable_sync(_input: Any) -> Runnable:
            return runnable_sync

        async def _call_runnable_async(_input: Any) -> Runnable:
            return runnable_async

        bound: Runnable = (
            history_chain
            | RunnableLambda(
                _call_runnable_sync,
                _call_runnable_async,
            ).with_config(run_name="check_sync_or_async")
        ).with_config(run_name="RunnableWithMessageHistory")

        if history_factory_config:
            _config_specs = history_factory_config
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
            time_splitter=time_splitter,
            history_limit=history_limit,
            input_messages_key=input_messages_key,
            output_messages_key=output_messages_key,
            bound=bound,
            history_messages_key=history_messages_key,
            history_factory_config=_config_specs,
            **kwargs,
        )
        self._history_chain = history_chain

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        """Get the configuration specs for the RunnableWithMessageHistory."""
        return get_unique_config_specs(super().config_specs + list(self.history_factory_config))

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
    ) -> list[tuple[BaseMessage, float]]:
        # If dictionary, try to pluck the single key representing messages
        result: list[tuple[BaseMessage, float]] = []
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

            result = [(HumanMessage(content=input_val), time.time())]
        # If value is a single message, convert to a list
        if isinstance(input_val, BaseMessage):
            result = [(input_val, time.time())]
        if isinstance(input_val, (list, tuple)):
            if len(input_val) == 0:
                return []
            if any(isinstance(item, BaseMessage) for item in input_val):
                result = [(item, time.time()) for item in input_val]  # type: ignore
            else:
                return self._get_messages_with_timestamps(input_val)  # type: ignore
        if result:
            return sorted(result, key=lambda x: x[1])
        msg = (
            f"Expected str, BaseMessage, list[BaseMessage], or tuple[BaseMessage]. Got {input_val}."
        )
        raise ValueError(msg)

    def _get_output_messages(
        self,
        output_val: MessageLikeType | MessageLikeWithTimeType,
    ) -> list[tuple[BaseMessage, float]]:
        result: list[tuple[BaseMessage, float]] = []
        # If dictionary, try to pluck the single key representing messages
        if isinstance(output_val, dict):
            if self.output_messages_key:
                key = self.output_messages_key
            elif len(output_val) == 1:
                key = next(iter(output_val.keys()))
            else:
                key = "output"
            # If you are wrapping a chat model directly
            # The output is actually this weird generations object
            if key not in output_val and "generations" in output_val:
                output_val = output_val["generations"][0][0]["message"]
            else:
                output_val = output_val[key]

        if isinstance(output_val, str):
            from langchain_core.messages import AIMessage

            result = [(AIMessage(content=output_val), time.time())]
        # If value is a single message, convert to a list
        if isinstance(output_val, BaseMessage):
            result = [(output_val, time.time())]
        if isinstance(output_val, (list, tuple)):
            if len(output_val) == 0:
                return []
            if any(isinstance(item, BaseMessage) for item in output_val):
                result = [(item, time.time()) for item in output_val]  # type: ignore
            else:
                return self._get_messages_with_timestamps(output_val)  # type: ignore
        if result:
            return sorted(result, key=lambda x: x[1])
        msg = (
            f"Expected str, BaseMessage, list[BaseMessage], or tuple[BaseMessage]. "
            f"Got {output_val}."
        )
        raise ValueError(msg)

    def _enter_history(self, value: Any, config: RunnableConfig) -> list[BaseMessage]:
        session_id: str = config["configurable"]["_unique_session_id"]  # type: ignore
        cache_id: str = config["configurable"]["_input_messages_cache_id"]  # type: ignore
        messages = self.history_store.get_session_history(session_id, limit=self.history_limit)

        input_val = value if not self.input_messages_key else value[self.input_messages_key]
        input_messages = self._get_input_messages(input_val)
        if len(input_messages) != 1:
            raise ValueError(
                "We currently do not support the input of multiple messages to avoid unpredictable results. "
                "Please ensure that the input is a single message."
            )
        self._input_messages_cache[cache_id] = input_messages

        if not self.history_messages_key:
            # return all messages
            messages += [input_messages[0][0]]
        return messages

    async def _aenter_history(
        self, value: dict[str, Any], config: RunnableConfig
    ) -> list[BaseMessage]:
        session_id: str = config["configurable"]["_unique_session_id"]  # type: ignore
        cache_id: str = config["configurable"]["_input_messages_cache_id"]  # type: ignore
        messages = await self.history_store.aget_session_history(
            session_id, limit=self.history_limit
        )

        input_val = value if not self.input_messages_key else value[self.input_messages_key]
        input_messages = self._get_input_messages(input_val)
        if len(input_messages) != 1:
            raise ValueError(
                "We currently do not support the input of multiple messages to avoid unpredictable results. "
                "Please ensure that the input is a single message."
            )
        self._input_messages_cache[cache_id] = input_messages

        if not self.history_messages_key:
            # return all messages
            messages += [input_messages[0][0]]
        return messages

    def _exit_history(self, run: Run, config: RunnableConfig) -> None:
        session_id: str = config["configurable"]["_unique_session_id"]  # type: ignore
        cache_id: str = config["configurable"]["_input_messages_cache_id"]  # type: ignore

        # Get the input messages
        input_messages = self._input_messages_cache.pop(cache_id)

        assert len(input_messages) == 1
        block_id, block_ts = self.time_splitter.invoke(
            input={"timestamp": input_messages[0][1]},
            config={"configurable": {"session_id": session_id}},
        )

        # Get the output messages
        output_val = load(run.outputs)
        output_messages = self._get_output_messages(output_val)
        self.history_store.add_messages(
            input_messages + output_messages,
            block_id=block_id,
            block_ts=block_ts,
            session_id=session_id,
        )

    async def _aexit_history(self, run: Run, config: RunnableConfig) -> None:
        session_id: str = config["configurable"]["_unique_session_id"]  # type: ignore
        cache_id: str = config["configurable"]["_input_messages_cache_id"]  # type: ignore

        # Get the input messages
        input_messages = self._input_messages_cache.pop(cache_id)

        assert len(input_messages) == 1
        block_id, block_ts = self.time_splitter.invoke(
            input={"timestamp": input_messages[0][1]},
            config={"configurable": {"session_id": session_id}},
        )

        # Get the output messages
        output_val = load(run.outputs)
        output_messages = self._get_output_messages(output_val)
        await self.history_store.aadd_messages(
            input_messages + output_messages,
            block_id=block_id,
            block_ts=block_ts,
            session_id=session_id,
        )

    @override
    def _merge_configs(self, *configs: RunnableConfig | None) -> RunnableConfig:
        config = super()._merge_configs(*configs)
        expected_keys = [field_spec.id for field_spec in self.history_factory_config]

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
        config["configurable"]["_input_messages_cache_id"] = uuid.uuid1().hex  # type: ignore
        return config
