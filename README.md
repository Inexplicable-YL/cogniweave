# CogniWeave

CogniWeave is an experimental agent framework.  The repository includes
utilities and runnable components used in the tests.  A small helper
module ``cogniweave.quickstart`` exposes functions that simplify
building the demonstration pipeline used in ``tests/runnables/demo.py``.

## Quick demo

Install the project dependencies (see ``pyproject.toml``). Then run the
CLI to start an interactive session:

```bash
python scripts/cli.py demo
```

You can optionally specify a custom session identifier which controls
the history database and vector index used:

```bash
python scripts/cli.py demo my_session
```

The ``cogniweave.quickstart`` module also provides utilities to programmatically
construct the demo pipeline:

```python
from cogniweave.quickstart import build_pipeline

pipeline = build_pipeline(session_id="my_session")
```
