from __future__ import annotations

from env.episode import EpisodeWindow, make_window_from_events
from lob.events import BookEvent


def slice_non_overlapping_windows(events: list[BookEvent], *, episode_length_ms: int) -> list[EpisodeWindow]:
    ordered = sorted(events, key=lambda event: event.timestamp)
    windows: list[EpisodeWindow] = []
    start = 0
    while start < len(ordered):
        start_ts = ordered[start].timestamp
        end_ts = start_ts + episode_length_ms
        stop = start
        while stop < len(ordered) and ordered[stop].timestamp <= end_ts:
            stop += 1
        if stop > start:
            window = make_window_from_events(
                ordered[start:stop],
                episode_length_ms=episode_length_ms,
                flat_event_offset=start,
            )
            if window.events:
                windows.append(window)
        start = stop
    return windows
