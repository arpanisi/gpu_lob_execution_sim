from __future__ import annotations

import json
import zipfile

from data.orderbook import iter_bybit_orderbook_zip


def test_bybit_orderbook_zip_parser_streams_depth_records(tmp_path) -> None:
    path = tmp_path / "sample.zip"
    rows = [
        {"topic": "orderbook.500.BTCUSDT", "type": "snapshot", "ts": 1000, "data": {"s": "BTCUSDT", "b": [["99.0", "1.0"]], "a": [["101.0", "2.0"]]}},
        {"topic": "orderbook.500.BTCUSDT", "type": "delta", "ts": 1100, "data": {"s": "BTCUSDT", "b": [["99.0", "0.5"]], "a": []}},
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sample.data", "\n".join(json.dumps(row) for row in rows) + "\n")

    parsed = list(iter_bybit_orderbook_zip(path))
    assert parsed[0].is_snapshot
    assert parsed[0].timestamp == 1000
    assert parsed[0].bids == ((99.0, 1.0),)
    assert parsed[0].asks == ((101.0, 2.0),)
    assert not parsed[1].is_snapshot
    assert parsed[1].bids == ((99.0, 0.5),)
