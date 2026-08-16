from __future__ import annotations

from env.episode import make_window_from_events, observations, reset, step
from env.execution import AgentAction, ExecutionTask, execute_action, price_from_action, reward_from_fills
from lob.constants import AGENT_ID_START, CAPACITY
from lob.events import BookEvent
from lob.matching import Fill, OrderBook, aggregate_depth, apply_event


def _seeded_book() -> OrderBook:
    book = OrderBook.empty(capacity=8)
    apply_event(book, BookEvent("ADD", "bid", 99.0, 2.0, 1, 0, 1))
    apply_event(book, BookEvent("ADD", "ask", 101.0, 2.0, 2, 0, 1))
    return book


def test_action_offset_convention_positive_is_more_aggressive() -> None:
    book = _seeded_book()
    buy_task = ExecutionTask("buy", 1.0, 1.0, 0.0, 100.0)
    sell_task = ExecutionTask("sell", 1.0, 1.0, 0.0, 100.0)

    assert price_from_action(book, buy_task, AgentAction(offset_ticks=2, quantity=1.0)) == 101.2
    assert price_from_action(book, sell_task, AgentAction(offset_ticks=2, quantity=1.0)) == 98.8


def test_unfilled_limit_remainder_rests_under_agent_id() -> None:
    book = _seeded_book()
    task = ExecutionTask("buy", 1.0, 1.0, 0.0, 100.0)
    updated, reward, fills = execute_action(book, task, AgentAction(offset_ticks=-10, quantity=0.25), trader_id=AGENT_ID_START, timestamp=10)

    assert fills == []
    assert reward == 0.0
    assert updated.remaining_target == 1.0
    assert aggregate_depth(book, "bid") == {99.0: 2.0, 100.0: 0.25}


def test_agent_action_is_processed_before_historical_events() -> None:
    events = [
        BookEvent("ADD", "bid", 99.0, 2.0, 1, 0, 1000),
        BookEvent("ADD", "ask", 101.0, 2.0, 2, 0, 1000),
        BookEvent("CANCEL", "ask", 101.0, 0.5, 0, 0, 1001),
    ]
    state = reset(make_window_from_events(events), target_quantity=0.5)
    result = step(state, [AgentAction(offset_ticks=0, quantity=0.5)], k_events=1)

    assert result.processed_events == 1
    assert state.tasks[0].remaining_target == 0.0
    assert [fill.quantity for fill in result.fills if fill.incoming_trader_id == AGENT_ID_START] == [0.5]
    assert aggregate_depth(state.book, "ask") == {101.0: 1.0}


def test_terminal_shortfall_force_fills_remaining_target() -> None:
    events = [
        BookEvent("ADD", "bid", 99.0, 2.0, 1, 0, 1000),
        BookEvent("ADD", "ask", 101.0, 2.0, 2, 0, 1000),
    ]
    state = reset(make_window_from_events(events), target_quantity=0.25)
    result = step(state, [AgentAction(offset_ticks=-10, quantity=0.0)], k_events=50)

    assert result.done is True
    assert state.tasks[0].remaining_target == 0.0
    assert result.rewards[0] < 0.0
    assert result.fills[-1].price == 101.0


def test_two_agents_share_book_and_can_trade_against_each_other() -> None:
    events = [
        BookEvent("ADD", "bid", 99.0, 2.0, 1, 0, 1000),
        BookEvent("ADD", "ask", 101.0, 2.0, 2, 0, 1000),
        BookEvent("ADD", "bid", 98.0, 1.0, 3, 0, 1002),
    ]
    state = reset(make_window_from_events(events), agent_sides=("sell", "buy"), target_quantity=0.25)
    result = step(
        state,
        [
            AgentAction(offset_ticks=-5, quantity=0.25),
            AgentAction(offset_ticks=0, quantity=0.25),
        ],
        k_events=1,
    )

    agent_fills = [fill for fill in result.fills if fill.resting_trader_id == AGENT_ID_START]
    assert len(agent_fills) == 1
    assert agent_fills[0].incoming_trader_id == AGENT_ID_START + 1
    assert agent_fills[0].price == 99.5


def test_reward_zero_at_arrival_price() -> None:
    task = ExecutionTask("buy", 0.5, 0.5, 0.0, 100.0)
    fills = [Fill(1, 0, AGENT_ID_START, "ask", 100.0, 0.5, 10)]
    assert reward_from_fills(task, fills) == 0.0


def _window_with_matured_execution() -> tuple[list[BookEvent], int]:
    events = [
        BookEvent("ADD", "bid", 100.0, 5.0, 1, 0, 0),
        BookEvent("ADD", "ask", 101.0, 5.0, 2, 0, 0),
    ]
    order_id = 3
    for idx in range(60):
        timestamp = idx + 1
        events.append(BookEvent("EXECUTE", "ask", 101.0, 0.1, 0, 90, timestamp))
        events.append(BookEvent("ADD", "ask", 101.0, 0.1, order_id, 0, timestamp))
        order_id += 1
        events.append(BookEvent("ADD", "bid", 100.0 + idx * 0.02, 0.5, order_id, 0, timestamp))
        order_id += 1
    # The first replayable EXECUTE (window-relative index 0) matures 50 events later.
    return events, 51


def test_window_attaches_flow_signals_and_observation_uses_matured_value() -> None:
    events, mature_position = _window_with_matured_execution()
    window = make_window_from_events(events)
    assert window.flow_signals is not None

    state = reset(window)
    assert observations(state)[0][86] == 0.0
    result = step(state, [AgentAction(offset_ticks=0, quantity=0.0)], k_events=mature_position)
    assert result.processed_events == mature_position
    assert observations(state)[0][86] != 0.0


def test_episode_historical_replay_uses_training_overflow_mode() -> None:
    events = [BookEvent("ADD", "bid", 100.0, 1.0, order_id + 1, 0, 0) for order_id in range(CAPACITY)]
    events.append(BookEvent("ADD", "ask", 101.0, 1.0, 2001, 0, 0))
    events.append(BookEvent("ADD", "bid", 98.0, 1.0, 3001, 0, 1))

    window = make_window_from_events(events)
    state = reset(window)
    result = step(state, [AgentAction(offset_ticks=0, quantity=0.0)], k_events=1)

    assert result.done is True
    assert state.book.bid_overflows == 1
    assert state.book.ask_overflows == 0
