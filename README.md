# CogniWeave

CogniWeave is an experimental agent framework. The repository includes a
few utilities and runnable components used in the tests.  The small helper
module ``cogniweave.quickstart`` exposes functions that simplify building
the demonstration pipeline used in ``tests/runnables/demo.py``.

## Quick demo

Install the project dependencies (see ``pyproject.toml``) and then run the
CLI to start an interactive session:

```bash
python -m cogniweave demo
```

You can optionally specify a custom session identifier which controls
the history database and vector index used:

```bash
python -m cogniweave demo my_session
```

Additional options allow you to customise where history and vector
index data is stored:

```bash
python -m cogniweave demo my_session --index my_index --folder /tmp/cache
```

The ``--index`` argument determines the SQLite database and vector store
filenames.  ``--folder`` controls the folder in which these files are
created.

The ``cogniweave.quickstart`` module also provides utilities to programmatically
construct the demo pipeline:

```python
from cogniweave.quickstart import build_pipeline

pipeline = build_pipeline(index_name="my_session")
```

The agent and embedding models are configured via environment variables.
``AGENT_MODEL`` specifies ``<provider>/<model>`` for the chat agent while
``EMBEDDINGS_MODEL`` controls the embedding model.  If unset, sensible
defaults are used.
