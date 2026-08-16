from __future__ import annotations

from env.windows import slice_non_overlapping_windows
from lob.events import BookEvent


def test_slice_non_overlapping_windows_preserves_time_order_and_bounds() -> None:
    events = [
        BookEvent("ADD", "bid", 99.0, 1.0, 1, 0, 0),
        BookEvent("ADD", "ask", 101.0, 1.0, 2, 0, 0),
        BookEvent("ADD", "bid", 98.0, 1.0, 3, 0, 10),
        BookEvent("ADD", "bid", 97.0, 1.0, 4, 0, 25),
        BookEvent("ADD", "ask", 102.0, 1.0, 5, 0, 25),
        BookEvent("ADD", "bid", 96.0, 1.0, 6, 0, 40),
    ]

    windows = slice_non_overlapping_windows(events, episode_length_ms=20)

    assert len(windows) == 2
    assert windows[0].start_ts == 0
    assert windows[0].end_ts == 10
    assert [event.timestamp for event in windows[0].events] == [10]
    assert windows[1].start_ts == 25
    assert windows[1].events[0].timestamp == 40
