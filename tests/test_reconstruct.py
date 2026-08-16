from __future__ import annotations

from data.reconstruct import reconstruct_events
from lob.events import DepthRecord, TradePrint
from lob.matching import OrderBook, aggregate_depth, apply_event


def test_reconstruction_splits_decrease_between_execute_and_cancel_without_reusing_trade() -> None:
    depth = [
        DepthRecord(1_000, bids=((99.0, 2.0),), asks=((101.0, 1.0),), is_snapshot=True),
        DepthRecord(1_100, bids=((99.0, 0.5),), asks=()),
        DepthRecord(1_200, bids=((99.0, 0.0),), asks=()),
    ]
    trades = [TradePrint(timestamp=1_050, price=99.0, quantity=1.0, aggressor_side="sell", trade_id="t1")]

    result = reconstruct_events(depth, trades, tolerance_ms=1_000)
    assert result.trade_consumed["t1"] == 1.0
    assert [(event.event_type, event.side, event.price, event.quantity) for event in result.events] == [
        ("ADD", "bid", 99.0, 2.0),
        ("ADD", "ask", 101.0, 1.0),
        ("EXECUTE", "bid", 99.0, 1.0),
        ("CANCEL", "bid", 99.0, 0.5),
        ("CANCEL", "bid", 99.0, 0.5),
    ]

    book = OrderBook.empty(capacity=10)
    for event in result.events:
        apply_event(book, event)
    assert aggregate_depth(book, "bid") == {}
    assert aggregate_depth(book, "ask") == {101.0: 1.0}

