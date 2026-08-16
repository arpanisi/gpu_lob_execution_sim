from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from pathlib import Path

from lob.events import DepthRecord


def iter_bybit_orderbook_zip(path: str | Path, *, max_records: int | None = None) -> Iterator[DepthRecord]:
    source = Path(path)
    count = 0
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise ValueError(f"{source} must contain exactly one .data member")
        with archive.open(names[0]) as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue
                payload = json.loads(raw_line)
                data = payload["data"]
                yield DepthRecord(
                    timestamp=int(payload["ts"]),
                    bids=_parse_levels(data.get("b", [])),
                    asks=_parse_levels(data.get("a", [])),
                    is_snapshot=payload.get("type") == "snapshot",
                )
                count += 1
                if max_records is not None and count >= max_records:
                    break


def _parse_levels(levels: list[list[str]]) -> tuple[tuple[float, float], ...]:
    return tuple((float(price), float(quantity)) for price, quantity in levels)

