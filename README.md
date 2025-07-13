# CogniWeave

CogniWeave is an experimental agent framework built on top of [LangChain](https://github.com/langchain-ai/langchain). The repository showcases how to combine short‑term memory, persistent chat history and a long‑term vector store with end‑of‑conversation detection. The code base mainly serves as a set of runnable components used by the demonstration scripts and tests.

<p align="left">
<img src="https://github.com/Inexplicable-YL/CogniWeave/blob/main/docs/flow.png" width="600px">
</p>

## Features

- **Extensible chat agent** – defaults to OpenAI models but can be switched to other providers via environment variables.
- **Persistent chat history** – messages are stored in a SQLite database for later analysis and memory generation.
- **Vectorised long‑term memory** – FAISS indexes store tagged long‑term memory and allow retrieval as the conversation evolves.
- **Automatic memory creation** – short and long‑term memories are generated when a session ends and merged into the history.
- **Interactive CLI** – run `python -m cogniweave demo` to try the full pipeline from the terminal.

Additional helper functions for building the pipeline are available in the `cogniweave.quickstart` module.

## Environment variables

The agent relies on several environment variables. Reasonable defaults are used when a variable is not provided.

| Variable | Purpose | Default |
|----------|---------|---------|
| `CHAT_MODEL` | Chat model in the form `provider/model` | `openai/gpt-4.1` |
| `AGENT_MODEL` | Agent model in the form `provider/model` | `openai/gpt-4.1` |
| `EMBEDDINGS_MODEL` | Embedding model in the form `provider/model` | `openai/text-embedding-ada-002` |
| `SHORT_MEMORY_MODEL` | Model used to summarise recent messages | `openai/gpt-4.1-mini` |
| `LONG_MEMORY_MODEL` | Model used for long‑term memory extraction | `openai/gpt-o3` |
| `END_DETECTOR_MODEL` | Model that decides when a conversation is over | `openai/gpt-4.1-mini` |

Model providers usually require credentials such as `*_API_KEY` and `*_API_BASE`. These can be supplied via a `.env` file in the project root.

Environment variables are **case-insensitive** and override any value defined in
`config.toml`. All settings can be provided entirely through environment
variables. Nested options use `__` to separate levels, for example:

```bash
PROMPT_VALUES__CHAT__EN="You are a helpful assistant."
```

is equivalent to the configuration file section:

```toml
[prompt_values.chat]
en = "You are a helpful assistant."
```

## Configuration file

In addition to environment variables, settings can be defined in a `config.toml` (or
JSON/YAML) file. The CLI automatically loads this file when present, or you can
explicitly provide a path with `--config-file` or by calling
`cogniweave.init_config(_config_file=...)` in your own code.

```toml
index_name = "demo"
folder_path = "./.cache"
language = "en"
chat_model = "openai/gpt-4.1"
chat_temperature = 0.8
```

Any fields matching the keys of the :class:`cogniweave.config.Config` model are
accepted, including nested `prompt_values` sections for overriding system
prompts.

All `prompt_values` strings support the f-string placeholder `{default}`. The
placeholder is replaced with CogniWeave's built-in prompt so you can extend it
easily:

```toml
[prompt_values.end_detector]
en = "The agent's name is CogniWeave. {default}"
```

which becomes `"The agent's name is CogniWeave. You are a "message completeness detector. ..."`.

If you supply a configuration file or define nested options via environment
variables, make sure to call `cogniweave.init_config()` before invoking
`build_pipeline()` so the settings take effect.

## Multi-language support

The built-in prompt templates only include Chinese and English text. To use
another language, define the prompt in the `prompt_values` section and set the
`language` key to match. For Japanese using a TOML config:

```toml
language = "jp"

[prompt_values.chat]
jp = "あなたは役に立つアシスタントです。"
```

When a configuration file or environment variables include nested values like
this, remember to call `cogniweave.init_config()` before creating the
pipeline so the custom prompts are applied.

## CLI usage

After installing the dependencies (see `pyproject.toml`), start the interactive demo with:

```bash
python -m cogniweave demo
```

You can specify a session identifier to keep conversations separate:

```bash
python -m cogniweave demo my_session
```

Additional options control where history and vector data are stored:

```bash
python -m cogniweave demo my_session --index my_index --folder /tmp/cache
```

You can load custom configuration from a file using the --config-file argument:

```bash
python -m cogniweave demo my_session --config-file config.toml
```

The `--index` argument sets the file names for the SQLite database and FAISS index, while `--folder` chooses the directory used to store them. The optional `--config-file` points to a toml, json or yaml file that contains all the necessary settings for the demo.

## Quick build

The `quickstart.py` module assembles the entire pipeline for you:

```python
from cogniweave import init_config, build_pipeline

init_config()
pipeline = build_pipeline(index_name="demo")
```

The pipeline object exposes a LangChain `Runnable` that contains the agent, history store and vector store ready to use.

## Manual build

For full control you can construct the components step by step.

1. **Create embeddings**

   ```python
   from cogniweave.quickstart import create_embeddings

   embeddings = create_embeddings()
   ```

2. **Create history store**

   ```python
   from cogniweave.quickstart import create_history_store

   history_store = create_history_store(index_name="demo")
   ```

3. **Create vector store**

   ```python
   from cogniweave.quickstart import create_vector_store

   vector_store = create_vector_store(embeddings, index_name="demo")
   ```

4. **Create chat agent**

   ```python
   from cogniweave.quickstart import create_agent

   agent = create_agent()
   ```

5. **Wire up memory and end detection**

   ```python
   from cogniweave.core.runnables.memory_maker import RunnableWithMemoryMaker
   from cogniweave.core.runnables.end_detector import RunnableWithEndDetector
   from cogniweave.core.runnables.history_store import RunnableWithHistoryStore
   from cogniweave.core.end_detector import EndDetector
   from cogniweave.core.time_splitter import TimeSplitter

   pipeline = RunnableWithMemoryMaker(
       agent,
       history_store=history_store,
       vector_store=vector_store,
       input_messages_key="input",
       history_messages_key="history",
       short_memory_key="short_memory",
       long_memory_key="long_memory",
   )
   pipeline = RunnableWithEndDetector(
       pipeline,
       end_detector=EndDetector(),
       default={"output": []},
       history_messages_key="history",
   )
   pipeline = RunnableWithHistoryStore(
       pipeline,
       history_store=history_store,
       time_splitter=TimeSplitter(),
       input_messages_key="input",
       history_messages_key="history",
   )
   ```

6. **Stream messages**

   ```python
   for chunk in pipeline.stream({"input": "Hello"}, config={"configurable": {"session_id": "demo"}}):
       print(chunk, end="")
   ```

With these steps you can tailor the pipeline to your own requirements.
