from __future__ import annotations

from lob.events import BookEvent
from lob.matching import OrderBook, aggregate_depth, apply_event, batched_apply_event, clone_book, match_marketable_limit, stacked_arrays


def test_add_cancel_execute_fifo() -> None:
    book = OrderBook.empty(capacity=4)
    apply_event(book, BookEvent("ADD", "ask", 100.0, 1.0, 1, 0, 10))
    apply_event(book, BookEvent("ADD", "ask", 100.0, 2.0, 2, 0, 11))
    assert aggregate_depth(book, "ask") == {100.0: 3.0}

    apply_event(book, BookEvent("CANCEL", "ask", 100.0, 0.5, 0, 0, 12))
    assert aggregate_depth(book, "ask") == {100.0: 2.5}

    fills = apply_event(book, BookEvent("EXECUTE", "ask", 100.0, 1.0, 0, 99, 13))
    assert [fill.resting_order_id for fill in fills] == [1, 2]
    assert [fill.quantity for fill in fills] == [0.5, 0.5]
    assert aggregate_depth(book, "ask") == {100.0: 1.5}


def test_validation_overflow_raises_training_overflow_counts() -> None:
    book = OrderBook.empty(capacity=1)
    apply_event(book, BookEvent("ADD", "bid", 99.0, 1.0, 1, 0, 10), validation=True)
    try:
        apply_event(book, BookEvent("ADD", "bid", 98.0, 1.0, 2, 0, 11), validation=True)
    except OverflowError:
        pass
    else:
        raise AssertionError("validation overflow must raise")

    apply_event(book, BookEvent("ADD", "bid", 98.0, 1.0, 2, 0, 11), validation=False)
    assert book.bid_overflows == 1


def test_batched_apply_matches_independent_unbatched_books() -> None:
    base_a = OrderBook.empty(capacity=4)
    base_b = OrderBook.empty(capacity=4)
    apply_event(base_a, BookEvent("ADD", "bid", 99.0, 1.0, 1, 0, 10))
    apply_event(base_b, BookEvent("ADD", "ask", 101.0, 1.0, 2, 0, 10))

    independent_a = clone_book(base_a)
    independent_b = clone_book(base_b)
    batched_a = clone_book(base_a)
    batched_b = clone_book(base_b)
    events = [
        BookEvent("CANCEL", "bid", 99.0, 0.25, 0, 0, 11),
        BookEvent("EXECUTE", "ask", 101.0, 0.5, 0, 99, 11),
    ]

    apply_event(independent_a, events[0])
    apply_event(independent_b, events[1])
    batched_apply_event([batched_a, batched_b], events)

    left_bids, left_asks = stacked_arrays([independent_a, independent_b])
    right_bids, right_asks = stacked_arrays([batched_a, batched_b])
    assert (left_bids == right_bids).all()
    assert (left_asks == right_asks).all()


def test_reconstructed_execute_consumes_exact_price_not_better_prices() -> None:
    book = OrderBook.empty(capacity=4)
    apply_event(book, BookEvent("ADD", "bid", 101.0, 1.0, 1, 0, 10))
    apply_event(book, BookEvent("ADD", "bid", 100.0, 1.0, 2, 0, 11))
    fills = apply_event(book, BookEvent("EXECUTE", "bid", 100.0, 0.25, 0, 99, 12))
    assert fills[0].resting_order_id == 2
    assert aggregate_depth(book, "bid") == {100.0: 0.75, 101.0: 1.0}


def test_cancel_uses_order_id_when_present() -> None:
    book = OrderBook.empty(capacity=4)
    apply_event(book, BookEvent("ADD", "bid", 100.0, 1.0, 1, 0, 10))
    apply_event(book, BookEvent("ADD", "bid", 100.0, 1.0, 2, 0, 11))
    apply_event(book, BookEvent("CANCEL", "bid", 100.0, 0.25, 2, 0, 12))
    fills = apply_event(book, BookEvent("EXECUTE", "bid", 100.0, 1.25, 0, 99, 13))

    assert [fill.resting_order_id for fill in fills] == [1, 2]
    assert [fill.quantity for fill in fills] == [1.0, 0.25]


def test_marketable_limit_walks_best_price_first() -> None:
    book = OrderBook.empty(capacity=4)
    apply_event(book, BookEvent("ADD", "ask", 100.0, 1.0, 1, 0, 10))
    apply_event(book, BookEvent("ADD", "ask", 101.0, 1.0, 2, 0, 11))
    fills = match_marketable_limit(book, "ask", limit_price=101.0, quantity=1.5, incoming_trader_id=99, timestamp=12)
    assert [fill.resting_order_id for fill in fills] == [1, 2]
    assert [fill.quantity for fill in fills] == [1.0, 0.5]
