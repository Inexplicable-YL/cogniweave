import os
os.environ.setdefault("OPENAI_API_KEY", "test")
from cogniweave.workflow import ChatState, build_memory_graph


def test_graph_compile() -> None:
    graph = build_memory_graph()
    assert graph is not None

    assert hasattr(graph, "invoke"), "Compiled graph should have an invoke method"
    # verify expected workflow nodes exist
    # some Graph implementations expose 'nodes' attribute
    nodes = getattr(graph, 'nodes', None) or []
    expected_nodes = {
        'check_complete',
        'short_summary_node',
        'short_tags_node',
        'write_short_node',
        'extract_long_memory_node',
        'update_long_memory_node',
    }
    for node in expected_nodes:
        assert node in nodes, f"Expected node '{node}' in compiled graph, got: {{nodes}}"