import asyncio
import time

import matplotlib.pyplot as plt
from grouping import AsyncConditionDensityManager

# assume AsyncConditionDensityManager 已经在当前作用域中可用


async def simulate_and_plot() -> None:
    """simulate events for a single user and plot density over time."""
    manager = AsyncConditionDensityManager()

    # simulate message timestamps with varying间隔
    base = time.time()
    intervals = [1, 2, 2, 50, 1, 1, 100, 1, 2, 2, 500, 2]
    timestamps = [base]
    for delta in intervals:
        timestamps.append(timestamps[-1] + delta)

    densities = []

    for ts in timestamps:
        await manager.update_condition_density("user1", current_time=ts)
        densities.append(manager.get_density_weight("user1"))

    # plot density over event index
    plt.figure()
    plt.plot(range(len(densities)), densities)
    plt.xlabel("Event Index")
    plt.ylabel("Density Weight")
    plt.title("Density Weight over Simulated Events")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

async def test_session_splitting():
    """test segmentation: assign each message to a session index and print."""
    manager = AsyncConditionDensityManager(
        segment_min=10.0,        # 最小拆分阈值 10 秒
        segment_max=1000.0,      # 最大拆分阈值 1000 秒（够大即可）
        segment_factor=1.5,      # 放缩系数 1.5
        std_multiplier=0.0,      # 不加标准差
        min_messages_per_session=2
    )

    # 模拟一串消息时间戳（单位：秒），间隔变化较大，以触发拆分
    base = time.time()
    intervals = [1, 2, 2, 50, 1, 1, 100, 1, 2, 2, 500, 2]
    timestamps = [base]
    for delta in intervals:
        timestamps.append(timestamps[-1] + delta)

    session_indices = []
    current_session = 0

    for ts in timestamps:
        # 获取调用前的消息计数（默认为 0）
        previous_count = manager._message_count_per_key.get("user1", 0)

        # 实时更新密度（内部会根据delta动态判断是否新会话并在需要时重置计数）
        await manager.update_condition_density("user1", current_time=ts)

        # 获取调用后的新消息计数
        new_count = manager._message_count_per_key["user1"]

        # 如果调用后变为 1，且调用前不为 0，说明重置了，会话编号 +1
        if new_count == 1 and previous_count != 0:
            current_session += 1

        session_indices.append(current_session)

    # 打印每条消息的偏移时间（相对于 base）和所属会话编号
    print("Index | Timestamp Offset | Session")
    for idx, (ts, sess) in enumerate(zip(timestamps, session_indices)):
        offset = ts - base
        print(f"{idx:5d} | {offset:15.1f}s | {sess:7d}")
# run the async simulation and visualization
asyncio.run(test_session_splitting())
