from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lob.constants import CAPACITY, EMPTY_PRICE, EVENT_ADD, EVENT_CANCEL, EVENT_EXECUTE, SIDE_ASK, SIDE_BID
from lob.events import BookEvent, Side


@dataclass(frozen=True)
class Fill:
    resting_order_id: int
    resting_trader_id: int
    incoming_trader_id: int
    side: Side
    price: float
    quantity: float
    timestamp: int


@dataclass
class OrderBook:
    bids: np.ndarray
    asks: np.ndarray
    bid_overflows: int = 0
    ask_overflows: int = 0

    @classmethod
    def empty(cls, capacity: int = CAPACITY) -> "OrderBook":
        bids = np.full((capacity, 5), 0.0, dtype=float)
        asks = np.full((capacity, 5), 0.0, dtype=float)
        bids[:, 0] = EMPTY_PRICE
        asks[:, 0] = EMPTY_PRICE
        return cls(bids=bids, asks=asks)

    def side_array(self, side: Side) -> np.ndarray:
        return self.bids if side == "bid" else self.asks


def apply_event(book: OrderBook, event: BookEvent, *, validation: bool = True) -> list[Fill]:
    if event.event_type == "ADD":
        add_order(book, event.side, event.price, event.quantity, event.order_id, event.trader_id, event.timestamp, validation=validation)
        return []
    if event.event_type == "CANCEL":
        if event.order_id:
            cancel_by_order_id(book, event.side, event.order_id, event.quantity)
        else:
            cancel_fifo(book, event.side, event.price, event.quantity)
        return []
    if event.event_type == "EXECUTE":
        return execute_fifo(book, event.side, event.price, event.quantity, event.trader_id, event.timestamp)
    raise ValueError(f"unknown event type {event.event_type}")


def add_order(
    book: OrderBook,
    side: Side,
    price: float,
    quantity: float,
    order_id: int,
    trader_id: int,
    timestamp: int,
    *,
    validation: bool,
) -> None:
    arr = book.side_array(side)
    empty = np.flatnonzero(arr[:, 0] == EMPTY_PRICE)
    if empty.size == 0:
        if validation:
            raise OverflowError(f"{side} capacity overflow at price={price} timestamp={timestamp}")
        if side == "bid":
            book.bid_overflows += 1
        else:
            book.ask_overflows += 1
        return
    row = int(empty[0])
    arr[row] = [float(price), float(quantity), float(order_id), float(trader_id), float(timestamp)]


def cancel_fifo(book: OrderBook, side: Side, price: float, quantity: float) -> float:
    return _consume_fifo(book.side_array(side), float(price), float(quantity))


def cancel_by_order_id(book: OrderBook, side: Side, order_id: int, quantity: float) -> float:
    arr = book.side_array(side)
    rows = np.flatnonzero((arr[:, 0] != EMPTY_PRICE) & (arr[:, 2] == float(order_id)))
    if rows.size == 0:
        return 0.0
    row = int(rows[0])
    take = min(float(quantity), float(arr[row, 1]))
    arr[row, 1] -= take
    if arr[row, 1] <= 1e-12:
        arr[row] = [EMPTY_PRICE, 0.0, 0.0, 0.0, 0.0]
    return take


def execute_fifo(book: OrderBook, resting_side: Side, price: float, quantity: float, incoming_trader_id: int, timestamp: int) -> list[Fill]:
    arr = book.side_array(resting_side)
    remaining = float(quantity)
    fills: list[Fill] = []
    while remaining > 1e-12:
        rows = _exact_price_time_rows(arr, float(price))
        if rows.size == 0:
            break
        row = int(rows[0])
        fill_qty = min(remaining, float(arr[row, 1]))
        fills.append(
            Fill(
                resting_order_id=int(arr[row, 2]),
                resting_trader_id=int(arr[row, 3]),
                incoming_trader_id=int(incoming_trader_id),
                side=resting_side,
                price=float(arr[row, 0]),
                quantity=float(fill_qty),
                timestamp=int(timestamp),
            )
        )
        arr[row, 1] -= fill_qty
        remaining -= fill_qty
        if arr[row, 1] <= 1e-12:
            arr[row] = [EMPTY_PRICE, 0.0, 0.0, 0.0, 0.0]
    return fills


def match_marketable_limit(
    book: OrderBook,
    resting_side: Side,
    limit_price: float,
    quantity: float,
    incoming_trader_id: int,
    timestamp: int,
) -> list[Fill]:
    arr = book.side_array(resting_side)
    remaining = float(quantity)
    fills: list[Fill] = []
    while remaining > 1e-12:
        rows = _price_time_rows(arr, resting_side, limit_price=float(limit_price))
        if rows.size == 0:
            break
        row = int(rows[0])
        fill_qty = min(remaining, float(arr[row, 1]))
        fills.append(
            Fill(
                resting_order_id=int(arr[row, 2]),
                resting_trader_id=int(arr[row, 3]),
                incoming_trader_id=int(incoming_trader_id),
                side=resting_side,
                price=float(arr[row, 0]),
                quantity=float(fill_qty),
                timestamp=int(timestamp),
            )
        )
        arr[row, 1] -= fill_qty
        remaining -= fill_qty
        if arr[row, 1] <= 1e-12:
            arr[row] = [EMPTY_PRICE, 0.0, 0.0, 0.0, 0.0]
    return fills


def cancel_trader_orders(book: OrderBook, trader_id: int) -> None:
    _cancel_trader_side(book.bids, trader_id)
    _cancel_trader_side(book.asks, trader_id)


def aggregate_depth(book: OrderBook, side: Side) -> dict[float, float]:
    arr = book.side_array(side)
    active = arr[arr[:, 0] != EMPTY_PRICE]
    depth: dict[float, float] = {}
    for price, quantity, *_ in active:
        depth[float(price)] = depth.get(float(price), 0.0) + float(quantity)
    return {price: qty for price, qty in sorted(depth.items()) if qty > 1e-12}


def top_levels(book: OrderBook, side: Side, levels: int = 20) -> np.ndarray:
    depth = aggregate_depth(book, side)
    ordered = sorted(depth.items(), key=lambda item: item[0], reverse=(side == "bid"))[:levels]
    out = np.full((levels, 2), np.nan, dtype=float)
    for idx, (price, quantity) in enumerate(ordered):
        out[idx] = [price, quantity]
    return out


def best_price(book: OrderBook, side: Side) -> float:
    levels = top_levels(book, side, levels=1)
    price = levels[0, 0]
    if np.isnan(price):
        raise ValueError(f"no {side} price available")
    return float(price)


def mid_price(book: OrderBook) -> float:
    return (best_price(book, "bid") + best_price(book, "ask")) / 2.0


def event_to_array(event: BookEvent) -> np.ndarray:
    return np.array([event.type_code, event.side_code, event.price, event.quantity, event.order_id, event.trader_id, event.timestamp], dtype=float)


def clone_book(book: OrderBook) -> OrderBook:
    return OrderBook(book.bids.copy(), book.asks.copy(), book.bid_overflows, book.ask_overflows)


def batched_apply_event(books: list[OrderBook], events: list[BookEvent], *, validation: bool = True) -> list[list[Fill]]:
    if len(books) != len(events):
        raise ValueError("books and events must have identical lengths")
    return [apply_event(book, event, validation=validation) for book, event in zip(books, events)]


def stacked_arrays(books: list[OrderBook]) -> tuple[np.ndarray, np.ndarray]:
    return np.stack([book.bids for book in books]), np.stack([book.asks for book in books])


def _consume_fifo(arr: np.ndarray, price: float, quantity: float) -> float:
    remaining = float(quantity)
    rows = np.flatnonzero((arr[:, 0] == price) & (arr[:, 1] > 0))
    rows = rows[np.argsort(arr[rows, 4], kind="stable")]
    for row in rows:
        if remaining <= 1e-12:
            break
        take = min(remaining, float(arr[row, 1]))
        arr[row, 1] -= take
        remaining -= take
        if arr[row, 1] <= 1e-12:
            arr[row] = [EMPTY_PRICE, 0.0, 0.0, 0.0, 0.0]
    return float(quantity) - remaining


def _cancel_trader_side(arr: np.ndarray, trader_id: int) -> None:
    rows = np.flatnonzero((arr[:, 0] != EMPTY_PRICE) & (arr[:, 3] == float(trader_id)))
    for row in rows:
        arr[row] = [EMPTY_PRICE, 0.0, 0.0, 0.0, 0.0]


def _exact_price_time_rows(arr: np.ndarray, price: float) -> np.ndarray:
    rows = np.flatnonzero((arr[:, 0] == price) & (arr[:, 1] > 0))
    return rows[np.argsort(arr[rows, 4], kind="stable")]


def _price_time_rows(arr: np.ndarray, side: Side, limit_price: float) -> np.ndarray:
    active = np.flatnonzero(arr[:, 0] != EMPTY_PRICE)
    if side == "ask":
        marketable = active[arr[active, 0] <= limit_price]
        price_key = arr[marketable, 0]
    else:
        marketable = active[arr[active, 0] >= limit_price]
        price_key = -arr[marketable, 0]
    if marketable.size == 0:
        return marketable
    order = np.lexsort((arr[marketable, 4], price_key))
    return marketable[order]
