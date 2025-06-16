import itertools
import math
import statistics
import time
import uuid
from collections import defaultdict, deque
from enum import Enum
from typing import Any

import anyio
from pydantic import BaseModel, PrivateAttr


def Sigmoid(x: float, lower: float, upper: float, midpoint: float, steepness: float) -> float:
    """provide sigmoid function for growth and decay calculations."""
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

    def _prune(self, current_time: float) -> None:
        if not self.time_window:
            return
        expired = [
            key for key in self.slots if (current_time - key * self.time_window) > self.time_window
        ]
        for key in expired:
            del self.slots[key]

    def add(self, timestamp: float) -> None:
        """add a timestamp and prune expired slots."""
        self._prune(timestamp)
        slot_key = math.floor(timestamp / self.time_window) if self.time_window else 0
        self.slots[slot_key].append(timestamp)
        if len(self.slots[slot_key]) > self.window_size:
            self.slots[slot_key].popleft()

    def get_all(self, current_time: float | None = None) -> deque[float]:
        """retrieve all timestamps in ascending order."""
        if current_time is not None:
            self._prune(current_time)
        all_ts = deque()
        for slot in sorted(self.slots.keys()):
            all_ts.extend(self.slots[slot])
        return all_ts

    def is_empty(self, current_time: float) -> bool:
        """Return ``True`` if the wheel contains no valid timestamps."""
        self._prune(current_time)
        return not self.slots


class WeightedAverageCalculator:
    """Compute exponentially weighted moving average of intervals."""

    _cached_weights: dict

    def __init__(self, adaptive_strength: float = 0.5) -> None:
        self._cached_weights = {}
        self.adaptive_strength = adaptive_strength

    def compute(self, intervals: deque[float], smoothing_factor: float = 0.9) -> float:
        """Return EWMA of `intervals`."""
        num = len(intervals)
        if num <= 0:
            return 100.0

        if num not in self._cached_weights:
            weights = [math.exp(-smoothing_factor * i) for i in range(num - 1, -1, -1)]
            self._cached_weights[num] = weights

        weights = self._cached_weights[num]
        weighted_sum = sum(w * val for w, val in zip(weights, intervals, strict=False))
        weight_total = sum(weights)
        weighted_avg = weighted_sum / weight_total
        return max(float(weighted_avg), 1e-9)


class DynamicDecayCalculator:
    """compute a dynamic decay factor based on average interval."""

    @staticmethod
    def compute(avg_interval: float, decay_factor: float) -> float:
        """
        compute decay value.

        avg_interval: current average time interval
        decay_factor: base decay coefficient
        """
        dynamic_decay = decay_factor * Sigmoid(
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
        if len(self.recent_avg_intervals) < 2:  # noqa: PLR2004
            trend = avg_interval
        else:
            trend = statistics.fmean(self.recent_avg_intervals)
            if len(self.recent_avg_intervals) > 1:
                std_dev = statistics.pstdev(self.recent_avg_intervals)
            else:
                std_dev = 0.0
            if std_dev > 2:  # noqa: PLR2004
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
            result = statistics.fmean([prev_weight, density_increment * decay_factor])
        elif strategy == DensityStrategy.WEIGHTED_MOVING_AVERAGE:
            result = 0.6 * prev_weight + 0.4 * density_increment * decay_factor
        elif strategy == DensityStrategy.EXPONENTIALLY_WEIGHTED_MOVING_AVERAGE:
            result = self.ema_alpha * prev_weight + (1 - self.ema_alpha) * density_increment
        else:
            raise ValueError("unsupported density calculation strategy")

        return float(result) * DynamicDecayCalculator.compute(avg_interval, self.decay_factor)


class ConditionDensityManager(BaseModel):
    """Manage density weights with adaptive segmentation."""

    # exposed configuration
    time_window: float | None = None
    density_strategy: DensityStrategy = DensityStrategy.AUTO
    segment_factor: float = 0.5
    segment_min: float = 60.0
    segment_max: float = 3600.0
    std_multiplier: float = 0.5

    # internal constants
    _WINDOW_SIZE: int = 20
    _DECAY_FACTOR: float = 0.1
    _SCALING_FACTOR: int = 10
    _AVG_SMOOTHING: float = 0.9
    _EMA_ALPHA: float = 0.8
    _ADAPTIVE_STRENGTH: float = 0.9
    _AUTO_LOW: float = 4.0
    _AUTO_HIGH: float = 10.0
    _MIN_MSGS: int = 2

    # non-init internal state
    _session_timestamps: dict[str, TimeWheel] = PrivateAttr(default_factory=dict)
    _session_weights: dict[str, float] = PrivateAttr(
        default_factory=lambda: defaultdict(lambda: 1.0)
    )
    _last_timestamp: dict[str, float] = PrivateAttr(default_factory=dict)
    _intervals: dict[str, deque[float]] = PrivateAttr(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=5))
    )
    _message_count: dict[str, int] = PrivateAttr(default_factory=lambda: defaultdict(int))
    _segment_id_per_key: dict[str, str] = PrivateAttr(default_factory=dict)
    _density_calculator: DensityCalculator | None = PrivateAttr(default=None)
    _weighted_avg_calc: WeightedAverageCalculator | None = PrivateAttr(default=None)
    _alocks: dict[str, anyio.Lock] = PrivateAttr(default_factory=lambda: defaultdict(anyio.Lock))

    def __init__(
        self,
        *,
        time_window: float | None = None,
        density_strategy: DensityStrategy = DensityStrategy.AUTO,
        segment_factor: float = 0.5,
        segment_min: float = 60.0,
        segment_max: float = 3600.0,
        std_multiplier: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            time_window=time_window,
            density_strategy=density_strategy,
            segment_factor=segment_factor,
            segment_min=segment_min,
            segment_max=segment_max,
            std_multiplier=std_multiplier,
            **kwargs,
        )
        self._session_timestamps = defaultdict(
            lambda: TimeWheel(self._WINDOW_SIZE, self.time_window or 0)
        )
        self._density_calculator = DensityCalculator(
            strategy=self.density_strategy,
            ema_alpha=self._EMA_ALPHA,
            decay_factor=self._DECAY_FACTOR,
            auto_threshold_low=self._AUTO_LOW,
            auto_threshold_high=self._AUTO_HIGH,
        )
        self._weighted_avg_calc = WeightedAverageCalculator(self._ADAPTIVE_STRENGTH)

    def update_condition_density(self, session_id: str, current_time: float | None = None) -> str:
        """Update density for a session.

        Args:
            session_id: Identifier of the conversation.
            current_time: Unix timestamp of the new event.

        Returns:
            Current segment identifier.
        """
        assert self._weighted_avg_calc
        assert self._density_calculator

        now = float(current_time) if current_time is not None else time.time()
        last_ts = self._last_timestamp.get(session_id)

        segment_id = self._segment_id_per_key.get(session_id)
        if segment_id is None:
            segment_id = uuid.uuid4().hex
            self._segment_id_per_key[session_id] = segment_id

        if last_ts is not None:
            delta = now - last_ts
            intervals = self._intervals[session_id]
            intervals.append(delta)

            avg_interval = self._weighted_avg_calc.compute(intervals, self._AVG_SMOOTHING)
            std_dev = float(statistics.pstdev(list(intervals))) if len(intervals) > 1 else 0.0

            # compute dynamic threshold with std multiplier
            raw_threshold = self.segment_factor * avg_interval + self.std_multiplier * std_dev
            dynamic_threshold = min(max(raw_threshold, self.segment_min), self.segment_max)

            msg_count = self._message_count.get(session_id, 0)
            if delta > dynamic_threshold and msg_count >= self._MIN_MSGS:
                # new session: reset state and return new segment id
                self._session_timestamps[session_id] = TimeWheel(
                    self._WINDOW_SIZE, self.time_window or 0
                )
                self._session_weights[session_id] = 1.0
                self._intervals[session_id].clear()
                self._message_count[session_id] = 0
                segment_id = uuid.uuid4().hex
                self._segment_id_per_key[session_id] = segment_id
                self._last_timestamp[session_id] = now
                self._message_count[session_id] += 1
                tw = self._session_timestamps[session_id]
                tw.add(now)
                return segment_id

        else:
            # first message for this user
            avg_interval = 100.0  # fallback for first message

        # update last timestamp and increment message count
        self._last_timestamp[session_id] = now
        self._message_count[session_id] += 1

        # add current timestamp to time wheel
        tw = self._session_timestamps[session_id]
        tw.add(now)

        timestamps = tw.get_all(now if self.time_window else None)
        if len(timestamps) < 2:  # noqa: PLR2004
            avg_interval_calc = 100.0
        else:
            seq = list(timestamps)
            diffs = [b - a for a, b in itertools.pairwise(seq)]
            avg_interval_calc = float(statistics.fmean(diffs)) if diffs else 100.0

        growth = Sigmoid(avg_interval_calc, 0.275, 0.61, 8.6, 0.3)
        decay = Sigmoid(avg_interval_calc, 0.045, 0.155, 10, 0.3)

        growth = 0.9 * growth + 0.1 * self._session_weights[session_id]
        decay = max(0.05, 0.9 * decay + 0.1 * (1 / (avg_interval_calc + 1)))

        density_increment = growth / (avg_interval_calc + 1)
        decay_factor = math.exp(-decay * avg_interval_calc)

        new_weight = self._density_calculator.calculate(
            prev_weight=self._session_weights[session_id],
            density_increment=density_increment,
            decay_factor=decay_factor,
            avg_interval=avg_interval_calc,
        )
        self._session_weights[session_id] = new_weight

        return segment_id

    async def aupdate_condition_density(
        self, session_id: str, current_time: float | None = None
    ) -> str:
        async with self._alocks[session_id]:
            return self.update_condition_density(session_id, current_time)

    def get_density_weight(self, session_id: str) -> float:
        """Return current density weight for a session."""
        w = max(1e-9, self._session_weights.get(session_id, 1e-9) * 10)
        return 0.05 * max(-4, math.log(self._SCALING_FACTOR * w))
