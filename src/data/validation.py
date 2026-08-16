from __future__ import annotations

from dataclasses import dataclass

from lob.events import BookEvent, DepthRecord
from lob.matching import OrderBook, aggregate_depth, apply_event


@dataclass(frozen=True)
class ReplayValidation:
    checked_records: int
    passed: bool
    first_error: str | None = None


def validate_replay(events: list[BookEvent], checkpoints: list[DepthRecord], *, capacity: int = 1000) -> ReplayValidation:
    ordered_events = sorted(events, key=lambda event: event.timestamp)
    ordered_checkpoints = sorted(checkpoints, key=lambda record: record.timestamp)
    book = OrderBook.empty(capacity=capacity)
    expected_bid: dict[float, float] = {}
    expected_ask: dict[float, float] = {}
    event_pos = 0
    checked = 0
    for checkpoint in ordered_checkpoints:
        expected_bid = _apply_depth_record(expected_bid, checkpoint.bids, checkpoint.is_snapshot)
        expected_ask = _apply_depth_record(expected_ask, checkpoint.asks, checkpoint.is_snapshot)
        while event_pos < len(ordered_events) and ordered_events[event_pos].timestamp <= checkpoint.timestamp:
            try:
                apply_event(book, ordered_events[event_pos], validation=True)
            except OverflowError as exc:
                return ReplayValidation(checked, False, str(exc))
            event_pos += 1
        actual_bid = aggregate_depth(book, "bid")
        actual_ask = aggregate_depth(book, "ask")
        if not _depth_equal(actual_bid, expected_bid):
            return ReplayValidation(checked, False, f"bid depth mismatch at {checkpoint.timestamp}: expected={expected_bid} actual={actual_bid}")
        if not _depth_equal(actual_ask, expected_ask):
            return ReplayValidation(checked, False, f"ask depth mismatch at {checkpoint.timestamp}: expected={expected_ask} actual={actual_ask}")
        checked += 1
    return ReplayValidation(checked, True)


def validate_trade_consumption(trade_quantity: dict[str, float], consumed: dict[str, float]) -> ReplayValidation:
    for trade_id, used in consumed.items():
        available = trade_quantity.get(trade_id, 0.0)
        if used - available > 1e-12:
            return ReplayValidation(0, False, f"trade {trade_id} consumed {used} > available {available}")
    return ReplayValidation(len(consumed), True)


def _levels_to_depth(levels: tuple[tuple[float, float], ...]) -> dict[float, float]:
    return {float(price): float(quantity) for price, quantity in levels if quantity > 1e-12}


def _apply_depth_record(
    prior: dict[float, float],
    levels: tuple[tuple[float, float], ...],
    is_snapshot: bool,
) -> dict[float, float]:
    depth = {} if is_snapshot else dict(prior)
    for price, quantity in levels:
        price = float(price)
        quantity = float(quantity)
        if quantity <= 1e-12:
            depth.pop(price, None)
        else:
            depth[price] = quantity
    return depth


def _depth_equal(left: dict[float, float], right: dict[float, float]) -> bool:
    if set(left) != set(right):
        return False
    return all(abs(left[price] - right[price]) <= 1e-9 for price in left)
