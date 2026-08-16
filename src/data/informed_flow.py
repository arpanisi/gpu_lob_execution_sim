from __future__ import annotations

from dataclasses import dataclass
from statistics import quantiles

from lob.events import BookEvent
from lob.matching import OrderBook, apply_event, mid_price


@dataclass(frozen=True)
class FlowSignal:
    event_index: int
    mature_index: int
    impact_per_unit: float
    informed_heavy: bool


def compute_informed_flow_signals(
    events: list[BookEvent],
    *,
    horizon_events: int = 50,
    trailing_matured: int = 500,
) -> dict[int, FlowSignal]:
    mids = _mid_after_each_event(events)
    signals: dict[int, FlowSignal] = {}
    prior_impacts: list[float] = []
    for idx, event in enumerate(events):
        if event.event_type != "EXECUTE" or event.quantity <= 0:
            continue
        mature_idx = idx + horizon_events
        if mature_idx >= len(events):
            continue
        start_mid = mids[idx]
        end_mid = mids[mature_idx]
        direction = 1.0 if event.side == "ask" else -1.0
        impact = direction * (end_mid - start_mid) / event.quantity
        threshold = _top_quartile_threshold(prior_impacts[-trailing_matured:])
        informed = threshold is not None and impact >= threshold
        signal = FlowSignal(idx, mature_idx, float(impact), informed)
        signals[mature_idx] = signal
        prior_impacts.append(float(impact))
    return signals


def current_flow_value(signals_by_mature_index: dict[int, FlowSignal], replay_position: int) -> float:
    matured = [idx for idx in signals_by_mature_index if idx <= replay_position]
    if not matured:
        return 0.0
    return signals_by_mature_index[max(matured)].impact_per_unit


def current_flow_flag(signals_by_mature_index: dict[int, FlowSignal], replay_position: int) -> bool:
    matured = [idx for idx in signals_by_mature_index if idx <= replay_position]
    if not matured:
        return False
    return signals_by_mature_index[max(matured)].informed_heavy


def flow_value_series(signals_by_mature_index: dict[int, FlowSignal], length: int) -> list[float]:
    """Forward-filled per-unit impact values for replay positions 0 .. length-1.

    ``series[p]`` equals ``current_flow_value(signals, p)``: the per-unit impact of the most
    recently matured execution as of replay position ``p`` (0.0 before the first maturation). This
    lets per-environment replay positions in the batched environment be looked up in O(1) without
    rescanning the signal dict on every step.
    """
    series = [0.0] * length
    current = 0.0
    for pos in range(length):
        signal = signals_by_mature_index.get(pos)
        if signal is not None:
            current = signal.impact_per_unit
        series[pos] = current
    return series


def _mid_after_each_event(events: list[BookEvent]) -> list[float]:
    book = OrderBook.empty()
    mids: list[float] = []
    last_mid = 0.0
    for event in events:
        apply_event(book, event, validation=False)
        try:
            last_mid = mid_price(book)
        except ValueError:
            pass
        mids.append(last_mid)
    return mids


def _top_quartile_threshold(values: list[float]) -> float | None:
    if len(values) < 4:
        return None
    return float(quantiles(values, n=4, method="inclusive")[2])
