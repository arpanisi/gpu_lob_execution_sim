from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from lob.constants import EVENT_ADD, EVENT_CANCEL, SIDE_ASK, SIDE_BID
from lob.events import BookEvent
from lob.jax_matching import require_jax
from lob.matching import OrderBook, apply_event, stacked_arrays


def test_jax_matching_import_is_guarded_when_jax_is_missing() -> None:
    if importlib.util.find_spec("jax") is not None:
        require_jax()
    else:
        with pytest.raises(ModuleNotFoundError):
            require_jax()


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX is not installed")
def test_jax_batched_cancel_matches_unbatched_books() -> None:
    import jax.numpy as jnp

    from lob.jax_matching import batched_apply_event_arrays

    books = []
    for idx in range(3):
        book = OrderBook.empty(capacity=8)
        apply_event(book, BookEvent("ADD", "bid", 100.0, 1.0 + idx, idx + 1, 0, 1))
        apply_event(book, BookEvent("ADD", "ask", 101.0, 1.0, idx + 10, 0, 1))
        books.append(book)

    events = [BookEvent("CANCEL", "bid", 100.0, 0.25, 0, 0, 2) for _ in books]
    expected = []
    for book, event in zip(books, events):
        apply_event(book, event, validation=False)
        expected.append(book)

    bids, asks = stacked_arrays(expected)
    before_books = []
    for idx in range(3):
        book = OrderBook.empty(capacity=8)
        apply_event(book, BookEvent("ADD", "bid", 100.0, 1.0 + idx, idx + 1, 0, 1))
        apply_event(book, BookEvent("ADD", "ask", 101.0, 1.0, idx + 10, 0, 1))
        before_books.append(book)
    before_bids, before_asks = stacked_arrays(before_books)
    event_array = np.array([[EVENT_CANCEL, SIDE_BID, 100.0, 0.25, 0.0, 0.0, 2.0]] * 3, dtype=float)

    actual_bids, actual_asks, fill_quantities, _, _ = batched_apply_event_arrays(
        jnp.asarray(before_bids),
        jnp.asarray(before_asks),
        jnp.asarray(event_array),
    )

    np.testing.assert_allclose(np.asarray(actual_bids), bids)
    np.testing.assert_allclose(np.asarray(actual_asks), asks)
    np.testing.assert_allclose(np.asarray(fill_quantities), np.zeros(3))


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX is not installed")
def test_batched_add_reports_capacity_overflow_drop_flag() -> None:
    import jax.numpy as jnp

    from lob.jax_matching import batched_apply_event_arrays

    # Two envs: env 0's bid side is full (no empty row), env 1's bid side has one empty row.
    bids = jnp.zeros((2, 1, 5), dtype=jnp.float32)
    bids = bids.at[0, :, 0].set(100.0)
    bids = bids.at[1, :, 0].set(-1.0)
    asks = jnp.zeros((2, 1, 5), dtype=jnp.float32).at[:, :, 0].set(-1.0)
    events = jnp.asarray(
        [
            [EVENT_ADD, SIDE_BID, 99.0, 1.0, 5.0, 0.0, 1.0],
            [EVENT_ADD, SIDE_BID, 99.0, 1.0, 5.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )

    out_bids, _, _, bid_drops, ask_drops = batched_apply_event_arrays(bids, asks, events)

    assert [float(item) for item in bid_drops] == [1.0, 0.0]
    assert [float(item) for item in ask_drops] == [0.0, 0.0]
    # Env 1 accepted the add into its empty row; env 0's full book is unchanged.
    assert float(out_bids[1, 0, 0]) == pytest.approx(99.0)
    assert float(out_bids[0, 0, 0]) == pytest.approx(100.0)


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX is not installed")
def test_batched_add_with_capacity_one_overflow_matches_numpy_counter() -> None:
    import jax.numpy as jnp

    from lob.jax_matching import batched_apply_event_arrays
    from lob.constants import CAPACITY

    # Mirror the numpy reference behavior: an Add on a full side increments that side's counter.
    bids = jnp.zeros((1, CAPACITY, 5), dtype=jnp.float32).at[:, :, 0].set(-1.0)
    asks = jnp.zeros((1, CAPACITY, 5), dtype=jnp.float32).at[:, :, 0].set(100.0)
    event = jnp.asarray([[EVENT_ADD, SIDE_ASK, 101.0, 1.0, 5.0, 0.0, 1.0]], dtype=jnp.float32)
    _, _, _, bid_drops, ask_drops = batched_apply_event_arrays(bids, asks, event)
    assert float(bid_drops[0]) == 0.0
    assert float(ask_drops[0]) == 1.0
