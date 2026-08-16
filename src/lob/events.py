from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Literal

from lob.constants import EVENT_ADD, EVENT_CANCEL, EVENT_EXECUTE, MARKET_ID, SIDE_ASK, SIDE_BID

Side = Literal["bid", "ask"]
EventType = Literal["ADD", "CANCEL", "EXECUTE"]
AggressorSide = Literal["buy", "sell"]


@dataclass(frozen=True)
class BookEvent:
    event_type: EventType
    side: Side
    price: float
    quantity: float
    order_id: int
    trader_id: int
    timestamp: int
    trade_id: str | None = None

    @property
    def type_code(self) -> int:
        return {"ADD": EVENT_ADD, "CANCEL": EVENT_CANCEL, "EXECUTE": EVENT_EXECUTE}[self.event_type]

    @property
    def side_code(self) -> int:
        return SIDE_BID if self.side == "bid" else SIDE_ASK


@dataclass(frozen=True)
class TradePrint:
    timestamp: int
    price: float
    quantity: float
    aggressor_side: AggressorSide
    trade_id: str


@dataclass(frozen=True)
class DepthRecord:
    timestamp: int
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]
    is_snapshot: bool = False


def opposite_resting_side(aggressor_side: AggressorSide) -> Side:
    return "ask" if aggressor_side == "buy" else "bid"


def expected_aggressor_for_decrease(side: Side) -> AggressorSide:
    return "sell" if side == "bid" else "buy"


def market_add(side: Side, price: float, quantity: float, order_id: int, timestamp: int) -> BookEvent:
    return BookEvent("ADD", side, float(price), float(quantity), int(order_id), MARKET_ID, int(timestamp))


def event_to_dict(event: BookEvent) -> dict[str, Any]:
    return {
        "event_type": event.event_type,
        "side": event.side,
        "price": event.price,
        "quantity": event.quantity,
        "order_id": event.order_id,
        "trader_id": event.trader_id,
        "timestamp": event.timestamp,
        "trade_id": event.trade_id,
    }


def event_from_dict(payload: dict[str, Any]) -> BookEvent:
    return BookEvent(
        event_type=payload["event_type"],
        side=payload["side"],
        price=float(payload["price"]),
        quantity=float(payload["quantity"]),
        order_id=int(payload["order_id"]),
        trader_id=int(payload["trader_id"]),
        timestamp=int(payload["timestamp"]),
        trade_id=payload.get("trade_id"),
    )
