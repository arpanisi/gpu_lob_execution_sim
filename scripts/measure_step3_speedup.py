from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lob.constants import CAPACITY, EVENT_CANCEL, SIDE_BID
from lob.events import BookEvent
from lob.jax_matching import batched_apply_event_arrays, require_jax
from lob.matching import OrderBook, apply_event, stacked_arrays


def main() -> None:
    require_jax()
    import jax.numpy as jnp

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-sizes", default="16,64,256")
    parser.add_argument("--capacity", type=int, default=CAPACITY)
    parser.add_argument("--report-output", default="outputs/reports/step3_speedup.json")
    args = parser.parse_args()

    batch_sizes = [int(item.strip()) for item in args.batch_sizes.split(",") if item.strip()]
    rows = []
    for batch_size in batch_sizes:
        cpu_books = [_seed_book(args.capacity, idx) for idx in range(batch_size)]
        bids_np, asks_np = stacked_arrays(cpu_books)
        events_np = _event_batch(batch_size)

        bids = jnp.asarray(bids_np, dtype=jnp.float32)
        asks = jnp.asarray(asks_np, dtype=jnp.float32)
        events = jnp.asarray(events_np, dtype=jnp.float32)
        bids, asks, _, _, _ = batched_apply_event_arrays(bids, asks, events)
        bids.block_until_ready()

        gpu_start = time.perf_counter()
        for _ in range(args.steps):
            bids, asks, _, _, _ = batched_apply_event_arrays(bids, asks, events)
        bids.block_until_ready()
        gpu_seconds = time.perf_counter() - gpu_start

        cpu_events = [_cancel_event(idx) for idx in range(batch_size)]
        cpu_start = time.perf_counter()
        for _ in range(args.steps):
            for book, event in zip(cpu_books, cpu_events):
                apply_event(book, event, validation=False)
        cpu_seconds = time.perf_counter() - cpu_start

        rows.append(
            {
                "batch_size": batch_size,
                "steps": args.steps,
                "gpu_batched_seconds": gpu_seconds,
                "cpu_sequential_seconds": cpu_seconds,
                "speedup_ratio": cpu_seconds / gpu_seconds if gpu_seconds > 0 else None,
            }
        )

    report = {"measurements": rows, "passed": all(row["speedup_ratio"] is not None for row in rows)}
    output = _project_path(args.report_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _seed_book(capacity: int, offset: int) -> OrderBook:
    book = OrderBook.empty(capacity=capacity)
    for idx in range(4):
        apply_event(book, BookEvent("ADD", "bid", 100.0 - idx, 1_000_000.0, idx + 1 + offset * 100, 0, idx), validation=True)
    return book


def _cancel_event(offset: int) -> BookEvent:
    return BookEvent("CANCEL", "bid", 100.0, 0.001, 0, 0, offset)


def _event_batch(batch_size: int) -> np.ndarray:
    rows = np.zeros((batch_size, 7), dtype=float)
    rows[:, 0] = EVENT_CANCEL
    rows[:, 1] = SIDE_BID
    rows[:, 2] = 100.0
    rows[:, 3] = 0.001
    rows[:, 6] = np.arange(batch_size)
    return rows


def _project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


if __name__ == "__main__":
    main()
