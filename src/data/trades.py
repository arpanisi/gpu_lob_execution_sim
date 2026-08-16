from __future__ import annotations

import csv
import gzip
from pathlib import Path

from lob.events import TradePrint


def load_bybit_trade_gzip(path: str | Path) -> list[TradePrint]:
    rows: list[TradePrint] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            rows.append(_parse_trade_row(row, fallback_id=str(idx)))
    return rows


def _parse_trade_row(row: dict[str, str], fallback_id: str) -> TradePrint:
    timestamp = int(round(float(row["timestamp"]) * 1000.0))
    side = row["side"].strip().lower()
    if side not in {"buy", "sell"}:
        raise ValueError(f"unsupported Bybit trade side {row['side']!r}")
    return TradePrint(
        timestamp=timestamp,
        price=float(row["price"]),
        quantity=float(row["size"]),
        aggressor_side=side,  # type: ignore[arg-type]
        trade_id=row.get("trdMatchID") or fallback_id,
    )

