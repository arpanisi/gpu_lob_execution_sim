from __future__ import annotations

from data.informed_flow import compute_informed_flow_signals, current_flow_flag, current_flow_value, flow_value_series
from lob.events import BookEvent


def _impact_events() -> list[BookEvent]:
    events = [
        BookEvent("ADD", "bid", 100.0, 5.0, 1, 0, 0),
        BookEvent("ADD", "ask", 101.0, 5.0, 2, 0, 0),
    ]
    order_id = 3
    for idx in range(4):
        events.append(BookEvent("EXECUTE", "ask", 101.0, 0.5, 0, 90 + idx, idx + 1))
        for step in range(3):
            order_id += 1
            price = 100.0 + idx * 0.1 + step * 0.01
            events.append(BookEvent("ADD", "bid", price, 0.1, order_id, 0, idx * 10 + step + 2))
    events.append(BookEvent("EXECUTE", "ask", 101.0, 0.1, 0, 99, 100))
    for idx in range(5):
        order_id += 1
        events.append(BookEvent("ADD", "bid", 103.0 + idx, 0.1, order_id, 0, 101 + idx))
    return events


def test_flow_signal_does_not_expose_pending_execution_before_maturity() -> None:
    events = _impact_events()
    signals = compute_informed_flow_signals(events, horizon_events=3, trailing_matured=4)

    mature_index = min(signals)
    assert current_flow_value(signals, mature_index - 1) == 0.0
    assert current_flow_value(signals, mature_index) == signals[mature_index].impact_per_unit


def test_large_price_moving_execution_is_flagged_only_when_matured() -> None:
    events = _impact_events()
    signals = compute_informed_flow_signals(events, horizon_events=3, trailing_matured=4)
    flagged = [signal for signal in signals.values() if signal.informed_heavy]

    assert flagged
    first_flag = flagged[-1]
    assert current_flow_flag(signals, first_flag.mature_index - 1) is False
    assert current_flow_flag(signals, first_flag.mature_index) is True


def test_flow_value_series_forward_fills_most_recent_matured_execution() -> None:
    events = _impact_events()
    signals = compute_informed_flow_signals(events, horizon_events=3, trailing_matured=4)
    series = flow_value_series(signals, len(events) + 1)

    assert len(series) == len(events) + 1
    for position in range(len(events) + 1):
        assert series[position] == current_flow_value(signals, position)
    # Neutral sentinel before the first maturation, then the value is carried forward
    assert series[0] == 0.0
    first_mature = min(signals)
    assert series[first_mature - 1] == 0.0
    assert series[first_mature] == signals[first_mature].impact_per_unit
    assert series[first_mature + 1] == signals[first_mature].impact_per_unit
