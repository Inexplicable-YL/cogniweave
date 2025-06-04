import asyncio
import math
import time
from collections import defaultdict, deque
from enum import Enum
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator


class Sigmoid:
    """provide sigmoid function for growth and decay calculations."""

    @staticmethod
    def f(x: float, lower: float, upper: float, midpoint: float, steepness: float) -> float:
        """compute a sigmoid value."""
        return lower + (upper - lower) / (1 + math.exp(-steepness * (x - midpoint)))


class DensityStrategy(str, Enum):
    """enumerate supported density calculation strategies."""

    AUTO = "auto"
    EXPONENTIAL_MOVING_AVERAGE = "ema"
    SIMPLE_MOVING_AVERAGE = "sma"
    WEIGHTED_MOVING_AVERAGE = "wma"
    EXPONENTIALLY_WEIGHTED_MOVING_AVERAGE = "ewma"


class TimeWheel:
    """time wheel data structure for efficient time-window management."""

    def __init__(self, window_size: int, time_window: float) -> None:
        self.slots: dict[float, deque[float]] = defaultdict(deque)
        self.window_size = window_size
        self.time_window = time_window

    def add(self, timestamp: float) -> None:
        """add a timestamp to the appropriate slot."""
        slot_key = math.floor(timestamp / self.time_window) if self.time_window else 0
        self.slots[slot_key].append(timestamp)
        if len(self.slots[slot_key]) > self.window_size:
            self.slots[slot_key].popleft()

    def prune(self, current_time: float) -> None:
        """remove slots that are older than the time window."""
        expired = [
            key for key in self.slots if (current_time - key * self.time_window) > self.time_window
        ]
        for key in expired:
            del self.slots[key]

    def get_all(self) -> deque[float]:
        """retrieve all timestamps in ascending order."""
        all_ts = deque()
        for slot in sorted(self.slots.keys()):
            all_ts.extend(self.slots[slot])
        return all_ts


class WeightedAverageCalculator:
    """compute exponentially weighted moving average of timestamps."""

    _cached_weights: dict

    def __init__(self, adaptive_strength: float = 0.5) -> None:
        self._cached_weights = {}
        self.adaptive_strength = adaptive_strength

    def compute(self, timestamps: deque[float], smoothing_factor: float = 0.9) -> float:
        """
        compute ewma of time intervals.

        timestamps: deque of timestamps (must be sorted ascending)
        smoothing_factor: controls decay of older intervals
        """
        num_intervals = len(timestamps) - 1
        if num_intervals <= 0:
            return 100.0

        if num_intervals not in self._cached_weights:
            weights = np.exp(
                -smoothing_factor * np.linspace(0, num_intervals - 1, num_intervals)[::-1]
            )
            self._cached_weights[num_intervals] = weights

        weights = self._cached_weights[num_intervals]
        intervals = np.diff(list(timestamps))
        weighted_avg = np.dot(weights, intervals) / np.sum(weights)
        return max(weighted_avg, 1e-9)


class DynamicDecayCalculator:
    """compute a dynamic decay factor based on average interval."""

    @staticmethod
    def compute(avg_interval: float, decay_factor: float) -> float:
        """
        compute decay value.

        avg_interval: current average time interval
        decay_factor: base decay coefficient
        """
        dynamic_decay = decay_factor * Sigmoid.f(
            avg_interval, lower=0.5, upper=1.5, midpoint=6, steepness=1.5
        )
        return math.exp(-0.7 * dynamic_decay * avg_interval)


class DensityCalculator:
    """calculate density weight using various strategies."""

    def __init__(
        self,
        *,
        strategy: DensityStrategy,
        ema_alpha: float,
        decay_factor: float,
        auto_threshold_low: float = 4,
        auto_threshold_high: float = 10,
    ) -> None:
        self.strategy = strategy
        self.ema_alpha = ema_alpha
        self.decay_factor = decay_factor
        self.auto_threshold_low = auto_threshold_low
        self.auto_threshold_high = auto_threshold_high
        self.recent_avg_intervals: deque[float] = deque(maxlen=5)

    def _auto_select_strategy(self, avg_interval: float) -> DensityStrategy:
        """select best strategy based on recent interval trends."""
        self.recent_avg_intervals.append(avg_interval)
        if len(self.recent_avg_intervals) < 2:
            trend = avg_interval
        else:
            trend = np.mean(self.recent_avg_intervals)
            std_dev = np.std(self.recent_avg_intervals)
            if std_dev > 2:
                return DensityStrategy.EXPONENTIALLY_WEIGHTED_MOVING_AVERAGE

        if trend < self.auto_threshold_low:
            return DensityStrategy.EXPONENTIALLY_WEIGHTED_MOVING_AVERAGE
        if trend < self.auto_threshold_high:
            return DensityStrategy.WEIGHTED_MOVING_AVERAGE
        return DensityStrategy.SIMPLE_MOVING_AVERAGE

    def calculate(
        self,
        *,
        prev_weight: float,
        density_increment: float,
        decay_factor: float,
        avg_interval: float,
    ) -> float:
        """
        compute new density weight.

        prev_weight: previous density weight
        density_increment: growth component
        decay_factor: decay component
        avg_interval: average timestamp interval
        """
        if self.strategy == DensityStrategy.AUTO:
            strategy = self._auto_select_strategy(avg_interval)
        else:
            strategy = self.strategy

        if strategy == DensityStrategy.EXPONENTIAL_MOVING_AVERAGE:
            result = (
                self.ema_alpha * prev_weight
                + (1 - self.ema_alpha) * density_increment * decay_factor
            )
        elif strategy == DensityStrategy.SIMPLE_MOVING_AVERAGE:
            result = np.mean([prev_weight, density_increment * decay_factor])
        elif strategy == DensityStrategy.WEIGHTED_MOVING_AVERAGE:
            result = 0.6 * prev_weight + 0.4 * density_increment * decay_factor
        elif strategy == DensityStrategy.EXPONENTIALLY_WEIGHTED_MOVING_AVERAGE:
            result = self.ema_alpha * prev_weight + (1 - self.ema_alpha) * density_increment
        else:
            raise ValueError("unsupported density calculation strategy")

        return float(result) * DynamicDecayCalculator.compute(avg_interval, self.decay_factor)


class AsyncConditionDensityManager(BaseModel):
    """manage per-user density with adaptive session segmentation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    window_size: int = 20
    decay_factor: float = 0.1
    scaling_factor: int = 10
    avg_smoothing_factor: float = 0.9
    ema_alpha: float = 0.8
    time_window: float | None = None
    density_strategy: DensityStrategy = DensityStrategy.AUTO
    adaptive_strength: float = 0.9
    auto_threshold_low: float = 4
    auto_threshold_high: float = 10
    segment_factor: float = 2.0
    segment_min: float = 60.0
    segment_max: float = 3600.0
    std_multiplier: float = 1.0
    min_messages_per_session: int = 2

    # non-init internal state
    condition_timestamps: dict[str, TimeWheel] = {}
    condition_weights: dict[str, float] = {}
    _last_timestamp_per_key: dict[str, float] = {}
    _intervals_per_key: dict[str, deque[float]] = {}
    _message_count_per_key: dict[str, int] = {}
    density_calculator: DensityCalculator | None = None
    weighted_avg_calc: WeightedAverageCalculator | None = None
    lock: asyncio.Lock | None = None
    prune_task: asyncio.Task | None = None

    @model_validator(mode="after")
    def build_state(self) -> Self:
        self.condition_timestamps = defaultdict(
            lambda: TimeWheel(self.window_size, self.time_window or 0)
        )
        self.condition_weights = defaultdict(float)
        self._last_timestamp_per_key = {}
        self._intervals_per_key = defaultdict(lambda: deque(maxlen=5))
        self._message_count_per_key = defaultdict(int)
        self.density_calculator = DensityCalculator(
            strategy=self.density_strategy,
            ema_alpha=self.ema_alpha,
            decay_factor=self.decay_factor,
            auto_threshold_low=self.auto_threshold_low,
            auto_threshold_high=self.auto_threshold_high,
        )
        self.weighted_avg_calc = WeightedAverageCalculator(self.adaptive_strength)
        self.lock = asyncio.Lock()
        if self.time_window:
            self.prune_task = asyncio.create_task(self.auto_prune(interval=60.0))
        return self

    async def auto_prune(self, interval: float = 60.0) -> None:
        """periodically prune stale user state."""
        assert self.lock
        while True:
            await asyncio.sleep(interval)
            async with self.lock:
                now = time.time()
                for key in list(self.condition_timestamps.keys()):
                    self.condition_timestamps[key].prune(now)
                    if not self.condition_timestamps[key].get_all():
                        del self.condition_timestamps[key]
                        del self.condition_weights[key]
                        self._last_timestamp_per_key.pop(key, None)
                        self._intervals_per_key.pop(key, None)
                        self._message_count_per_key.pop(key, None)

    async def update_condition_density(
        self, condition_key: str, current_time: float | None = None
    ) -> None:
        """
        process a new message event for a user.

        condition_key: unique identifier for the user or session
        current_time: timestamp of the new message
        """
        assert self.lock
        assert self.weighted_avg_calc
        assert self.density_calculator

        async with self.lock:
            now = current_time or time.time()
            last_ts = self._last_timestamp_per_key.get(condition_key)

            if last_ts is not None:
                delta = now - last_ts
                intervals = self._intervals_per_key[condition_key]
                intervals.append(delta)

                # reconstruct fake timestamps for ewma computation
                fake_timestamps = deque()
                cur = last_ts
                fake_timestamps.append(cur)
                for d in reversed(intervals):
                    cur -= d
                    fake_timestamps.appendleft(cur)

                avg_interval = self.weighted_avg_calc.compute(
                    fake_timestamps, self.avg_smoothing_factor
                )
                std_dev = float(np.std(list(intervals))) if len(intervals) > 1 else 0.0

                # compute dynamic threshold with std multiplier
                raw_threshold = self.segment_factor * avg_interval + self.std_multiplier * std_dev
                dynamic_threshold = min(max(raw_threshold, self.segment_min), self.segment_max)

                msg_count = self._message_count_per_key.get(condition_key, 0)
                if delta > dynamic_threshold and msg_count >= self.min_messages_per_session:
                    # new session: reset state
                    self.condition_timestamps[condition_key] = TimeWheel(
                        self.window_size, self.time_window or 0
                    )
                    self.condition_weights[condition_key] = 0.0
                    self._intervals_per_key[condition_key].clear()
                    self._message_count_per_key[condition_key] = 0

            else:
                # first message for this user
                avg_interval = 100.0  # fallback for first message

            # update last timestamp and increment message count
            self._last_timestamp_per_key[condition_key] = now
            self._message_count_per_key[condition_key] += 1

            # add current timestamp to time wheel
            tw = self.condition_timestamps[condition_key]
            tw.add(now)
            if self.time_window:
                tw.prune(now)

            timestamps = tw.get_all()
            if len(timestamps) < 2:
                avg_interval_calc = 100.0
            else:
                avg_interval_calc = self.weighted_avg_calc.compute(
                    timestamps, self.avg_smoothing_factor
                )

            growth = Sigmoid.f(avg_interval_calc, 0.275, 0.61, 8.6, 0.3)
            decay = Sigmoid.f(avg_interval_calc, 0.045, 0.155, 10, 0.3)

            growth = 0.9 * growth + 0.1 * self.condition_weights[condition_key]
            decay = max(0.05, 0.9 * decay + 0.1 * (1 / (avg_interval_calc + 1)))

            density_increment = growth / (avg_interval_calc + 1)
            decay_factor = math.exp(-decay * avg_interval_calc)

            new_weight = self.density_calculator.calculate(
                prev_weight=self.condition_weights[condition_key],
                density_increment=density_increment,
                decay_factor=decay_factor,
                avg_interval=avg_interval_calc,
            )
            self.condition_weights[condition_key] = new_weight

    def get_density_weight(self, condition_key: str) -> float:
        """
        retrieve current density weight for a user.

        condition_key: unique identifier for the user
        returns: density weight mapped to [-0.2, 0.5]
        """
        w = max(1e-9, self.condition_weights.get(condition_key, 1e-9) * 10)
        return 0.05 * max(-4, math.log(self.scaling_factor * w))
