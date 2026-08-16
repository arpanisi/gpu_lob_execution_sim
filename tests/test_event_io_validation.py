from __future__ import annotations

from data.event_io import read_events_jsonl, write_events_jsonl
from data.reconstruct import reconstruct_events
from data.validation import validate_replay, validate_trade_consumption
from lob.events import DepthRecord, TradePrint


def _records() -> tuple[list[DepthRecord], list[TradePrint]]:
    depth = [
        DepthRecord(1_000, bids=((99.0, 2.0),), asks=((101.0, 1.0),), is_snapshot=True),
        DepthRecord(1_100, bids=((99.0, 0.5),), asks=((101.0, 1.5),)),
        DepthRecord(1_200, bids=((99.0, 0.0),), asks=((101.0, 1.5),)),
    ]
    trades = [TradePrint(timestamp=1_050, price=99.0, quantity=1.0, aggressor_side="sell", trade_id="t1")]
    return depth, trades


def test_event_jsonl_roundtrip_and_replay_validation(tmp_path) -> None:
    depth, trades = _records()
    result = reconstruct_events(depth, trades)
    path = write_events_jsonl(result.events, tmp_path / "events.jsonl")
    loaded = read_events_jsonl(path)

    assert loaded == result.events
    validation = validate_replay(loaded, depth)
    assert validation.passed, validation.first_error


def test_replay_validation_expands_delta_checkpoints_to_full_reference_depth() -> None:
    depth = [
        DepthRecord(1_000, bids=((99.0, 2.0), (98.0, 1.0)), asks=((101.0, 1.0),), is_snapshot=True),
        DepthRecord(1_100, bids=((99.0, 1.5),), asks=()),
    ]
    result = reconstruct_events(depth, [TradePrint(timestamp=1_050, price=99.0, quantity=0.5, aggressor_side="sell", trade_id="t1")])
    validation = validate_replay(result.events, depth)
    assert validation.passed, validation.first_error


def test_trade_consumption_validation_rejects_overuse() -> None:
    ok = validate_trade_consumption({"t1": 1.0}, {"t1": 1.0})
    bad = validate_trade_consumption({"t1": 1.0}, {"t1": 1.1})
    assert ok.passed
    assert not bad.passed
    assert "consumed" in (bad.first_error or "")
