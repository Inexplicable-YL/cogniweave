import pathlib
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).parent))
from grouping import AsyncConditionDensityManager


@pytest.mark.asyncio
async def test_segment_switch_after_gap() -> None:
    manager = AsyncConditionDensityManager()
    t0 = 1000.0
    seg1 = await manager.update_condition_density("user1", current_time=t0)
    seg2 = await manager.update_condition_density("user1", current_time=t0 + 1)
    seg3 = await manager.update_condition_density("user1", current_time=t0 + 120)

    assert seg1 == seg2
    assert seg3 != seg2
    assert manager.get_density_weight("user1") > 0


@pytest.mark.asyncio
async def test_sessions_are_independent() -> None:
    manager = AsyncConditionDensityManager()
    base = 2000.0

    seg_a1 = await manager.update_condition_density("A", current_time=base)
    seg_b1 = await manager.update_condition_density("B", current_time=base + 0.5)
    seg_a2 = await manager.update_condition_density("A", current_time=base + 1)
    seg_b2 = await manager.update_condition_density("B", current_time=base + 1.5)
    seg_a3 = await manager.update_condition_density("A", current_time=base + 130)
    seg_b3 = await manager.update_condition_density("B", current_time=base + 200)

    assert seg_a1 == seg_a2
    assert seg_a3 != seg_a2
    assert seg_b1 == seg_b2
    assert seg_b3 != seg_b2
    assert manager.get_density_weight("A") > 0
    assert manager.get_density_weight("B") > 0
