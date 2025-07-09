# CogniWeave

CogniWeave is an experimental agent framework built on top of [LangChain](https://github.com/langchain-ai/langchain). The repository showcases how to combine short‑term memory, persistent chat history and a long‑term vector store with end‑of‑conversation detection. The code base mainly serves as a set of runnable components used by the demonstration scripts and tests.

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

The `--index` argument sets the file names for the SQLite database and FAISS index, while `--folder` chooses the directory used to store them.

## Quick build

The `quickstart.py` module assembles the entire pipeline for you:

```python
from cogniweave.quickstart import build_pipeline

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
