from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lob.constants import TICK_SIZE
from lob.matching import Fill, OrderBook, add_order, best_price, match_marketable_limit

TaskSide = Literal["buy", "sell"]


@dataclass(frozen=True)
class ExecutionTask:
    side: TaskSide
    initial_target: float
    remaining_target: float
    executed_quantity: float
    arrival_mid: float


@dataclass(frozen=True)
class AgentAction:
    offset_ticks: float
    quantity: float


def initial_task(side: TaskSide, target_quantity: float, arrival_mid: float) -> ExecutionTask:
    return ExecutionTask(side, float(target_quantity), float(target_quantity), 0.0, float(arrival_mid))


def price_from_action(book: OrderBook, task: ExecutionTask, action: AgentAction, tick_size: float = TICK_SIZE) -> float:
    if task.side == "buy":
        return round(best_price(book, "ask") + float(action.offset_ticks) * tick_size, 10)
    return round(best_price(book, "bid") - float(action.offset_ticks) * tick_size, 10)


def execute_action(
    book: OrderBook,
    task: ExecutionTask,
    action: AgentAction,
    *,
    trader_id: int,
    timestamp: int,
) -> tuple[ExecutionTask, float, list[Fill]]:
    quantity = max(0.0, min(float(action.quantity), task.remaining_target))
    if quantity <= 0:
        return task, 0.0, []
    limit_price = price_from_action(book, task, action)
    resting_side = "ask" if task.side == "buy" else "bid"
    fills = match_marketable_limit(book, resting_side, limit_price, quantity, incoming_trader_id=trader_id, timestamp=timestamp)
    filled = sum(fill.quantity for fill in fills)
    unfilled = max(0.0, quantity - filled)
    if unfilled > 1e-12:
        own_side = "bid" if task.side == "buy" else "ask"
        add_order(
            book,
            own_side,
            limit_price,
            unfilled,
            order_id=agent_order_id(trader_id, timestamp),
            trader_id=trader_id,
            timestamp=timestamp,
            validation=False,
        )
    reward = reward_from_fills(task, fills)
    updated = ExecutionTask(
        task.side,
        task.initial_target,
        max(0.0, task.remaining_target - filled),
        task.executed_quantity + filled,
        task.arrival_mid,
    )
    return updated, reward, fills


def force_terminal_fill(book: OrderBook, task: ExecutionTask, *, trader_id: int, timestamp: int) -> tuple[ExecutionTask, float, list[Fill]]:
    if task.remaining_target <= 1e-12:
        return task, 0.0, []
    resting_side = "ask" if task.side == "buy" else "bid"
    limit_price = float("inf") if task.side == "buy" else 0.0
    fills = match_marketable_limit(
        book,
        resting_side,
        limit_price,
        task.remaining_target,
        incoming_trader_id=trader_id,
        timestamp=timestamp,
    )
    filled = sum(fill.quantity for fill in fills)
    reward = reward_from_fills(task, fills)
    updated = ExecutionTask(
        task.side,
        task.initial_target,
        max(0.0, task.remaining_target - filled),
        task.executed_quantity + filled,
        task.arrival_mid,
    )
    return updated, reward, fills


def reward_from_fills(task: ExecutionTask, fills: list[Fill]) -> float:
    if task.initial_target <= 0 or task.arrival_mid <= 0:
        raise ValueError("task initial_target and arrival_mid must be positive")
    sign = 1.0 if task.side == "buy" else -1.0
    cost = sum(sign * (fill.price - task.arrival_mid) * fill.quantity for fill in fills)
    return -float(cost) / (task.initial_target * task.arrival_mid)


def agent_order_id(trader_id: int, timestamp: int) -> int:
    return int(trader_id) * 1_000_000_000_000 + int(timestamp)
