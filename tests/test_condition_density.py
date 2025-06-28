import pathlib
import sys
from typing import Any

from cogniweave.core.time_splitter import TimeSplitter

sys.path.append(str(pathlib.Path(__file__).parent))


def _format_input(session_id: str, current_time: float) -> dict[str, Any]:
    return {
        "input": {"timestamp": current_time},
        "config": {"configurable": {"session_id": session_id}},
    }


def test_segment_switch_after_gap() -> None:
    manager = TimeSplitter()
    t0 = 1000.0
    seg1 = manager.invoke(**_format_input("user1", current_time=t0))
    seg2 = manager.invoke(**_format_input("user1", current_time=t0 + 15))
    seg3 = manager.invoke(**_format_input("user1", current_time=t0 + 40))
    seg4 = manager.invoke(**_format_input("user1", current_time=t0 + 120))
    seg5 = manager.invoke(**_format_input("user1", current_time=t0 + 140))
    seg6 = manager.invoke(**_format_input("user1", current_time=t0 + 160))

    assert {seg1, seg2, seg3} == {seg1}
    assert {seg4, seg5, seg6} == {seg4}
    assert seg1 != seg4


def test_sessions_are_independent() -> None:
    manager = TimeSplitter()
    base = 2000.0

    seg_a1 = manager.invoke(**_format_input("A", current_time=base))
    seg_c1 = manager.invoke(**_format_input("C", current_time=base + 1))
    seg_b1 = manager.invoke(**_format_input("B", current_time=base + 2))
    seg_a2 = manager.invoke(**_format_input("A", current_time=base + 3))
    seg_b2 = manager.invoke(**_format_input("B", current_time=base + 4))
    seg_a3 = manager.invoke(**_format_input("A", current_time=base + 7))
    seg_b3 = manager.invoke(**_format_input("B", current_time=base + 9))
    seg_d1 = manager.invoke(**_format_input("D", current_time=base + 10))
    seg_d2 = manager.invoke(**_format_input("D", current_time=base + 15))
    seg_a4 = manager.invoke(**_format_input("A", current_time=base + 120))
    seg_a5 = manager.invoke(**_format_input("A", current_time=base + 125))
    seg_b4 = manager.invoke(**_format_input("B", current_time=base + 150))
    seg_d3 = manager.invoke(**_format_input("D", current_time=base + 300))

    # A 会话断言（两个聚类）
    assert {seg_a1, seg_a2, seg_a3} == {seg_a1}
    assert {seg_a4, seg_a5} == {seg_a4}
    assert seg_a1 != seg_a4

    # B 会话断言（两个聚类）
    assert {seg_b1, seg_b2, seg_b3} == {seg_b1}
    assert seg_b4 != seg_b1

    # C 会话断言（单独聚类）
    assert {seg_c1} == {seg_c1}  # 仅自身

    # D 会话断言（两组聚类）
    assert {seg_d1, seg_d2} == {seg_d1}
    assert seg_d3 != seg_d1


async def test_segment_switch_after_gap_async() -> None:
    manager = TimeSplitter()
    t0 = 1000.0
    seg1 = await manager.ainvoke(**_format_input("user1", current_time=t0))
    seg2 = await manager.ainvoke(**_format_input("user1", current_time=t0 + 15))
    seg3 = await manager.ainvoke(**_format_input("user1", current_time=t0 + 40))
    seg4 = await manager.ainvoke(**_format_input("user1", current_time=t0 + 120))
    seg5 = await manager.ainvoke(**_format_input("user1", current_time=t0 + 140))
    seg6 = await manager.ainvoke(**_format_input("user1", current_time=t0 + 160))

    assert {seg1, seg2, seg3} == {seg1}
    assert {seg4, seg5, seg6} == {seg4}
    assert seg1 != seg4


async def test_sessions_are_independent_async() -> None:
    manager = TimeSplitter()
    base = 2000.0

    seg_a1 = await manager.ainvoke(**_format_input("A", current_time=base))
    seg_c1 = await manager.ainvoke(**_format_input("C", current_time=base + 1))
    seg_b1 = await manager.ainvoke(**_format_input("B", current_time=base + 2))
    seg_a2 = await manager.ainvoke(**_format_input("A", current_time=base + 3))
    seg_b2 = await manager.ainvoke(**_format_input("B", current_time=base + 4))
    seg_a3 = await manager.ainvoke(**_format_input("A", current_time=base + 7))
    seg_b3 = await manager.ainvoke(**_format_input("B", current_time=base + 9))
    seg_d1 = await manager.ainvoke(**_format_input("D", current_time=base + 10))
    seg_d2 = await manager.ainvoke(**_format_input("D", current_time=base + 15))
    seg_a4 = await manager.ainvoke(**_format_input("A", current_time=base + 120))
    seg_a5 = await manager.ainvoke(**_format_input("A", current_time=base + 125))
    seg_b4 = await manager.ainvoke(**_format_input("B", current_time=base + 150))
    seg_d3 = await manager.ainvoke(**_format_input("D", current_time=base + 300))

    # A 会话断言（两个聚类）
    assert {seg_a1, seg_a2, seg_a3} == {seg_a1}
    assert {seg_a4, seg_a5} == {seg_a4}
    assert seg_a1 != seg_a4

    # B 会话断言（两个聚类）
    assert {seg_b1, seg_b2, seg_b3} == {seg_b1}
    assert seg_b4 != seg_b1

    # C 会话断言（单独聚类）
    assert {seg_c1} == {seg_c1}  # 仅自身

    # D 会话断言（两组聚类）
    assert {seg_d1, seg_d2} == {seg_d1}
    assert seg_d3 != seg_d1
