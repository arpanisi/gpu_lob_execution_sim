from __future__ import annotations

import json
from pathlib import Path

from lob.events import BookEvent, event_from_dict, event_to_dict


def write_events_jsonl(events: list[BookEvent], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event_to_dict(event), separators=(",", ":")) + "\n")
    return output


def read_events_jsonl(path: str | Path) -> list[BookEvent]:
    source = Path(path)
    events: list[BookEvent] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(event_from_dict(json.loads(line)))
    return events

