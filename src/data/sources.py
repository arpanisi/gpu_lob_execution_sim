from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


TRADE_BASE_URL = "https://public.bybit.com/trading/BTCUSDT"
ORDERBOOK_BASE_URL = "https://quote-saver.bycsi.com/orderbook/linear/BTCUSDT"


@dataclass(frozen=True)
class ProbeResult:
    url: str
    ok: bool
    status: int | None
    content_length: int | None
    content_type: str | None
    error: str | None = None


def trade_url(date: str) -> str:
    return f"{TRADE_BASE_URL}/BTCUSDT{date}.csv.gz"


def orderbook_url(date: str, depth: int = 500) -> str:
    return f"{ORDERBOOK_BASE_URL}/{date}_BTCUSDT_ob{depth}.data.zip"


def probe(url: str, *, bytes_to_probe: int = 1024) -> ProbeResult:
    headers = {"User-Agent": "gpu-lob-execution-sim/0.1", "Range": f"bytes=0-{bytes_to_probe - 1}"}
    try:
        with urlopen(Request(url, headers=headers), timeout=30) as response:
            return ProbeResult(
                url=url,
                ok=response.status in (200, 206),
                status=response.status,
                content_length=_content_length(response.headers.get("content-length")),
                content_type=response.headers.get("content-type"),
            )
    except HTTPError as exc:
        return ProbeResult(url=url, ok=False, status=exc.code, content_length=None, content_type=None, error=str(exc))
    except Exception as exc:
        return ProbeResult(url=url, ok=False, status=None, content_length=None, content_type=None, error=str(exc))


def download(url: str, destination: str | Path, *, max_bytes: int | None = None) -> Path:
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "gpu-lob-execution-sim/0.1"}
    if max_bytes is not None:
        headers["Range"] = f"bytes=0-{max_bytes - 1}"
    with urlopen(Request(url, headers=headers), timeout=120) as response:
        if response.status not in (200, 206):
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        output.write_bytes(response.read())
    return output


def inspect_trade_gzip(path: str | Path, rows: int = 5) -> list[str]:
    lines: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= rows:
                break
            lines.append(line.rstrip("\n"))
    return lines


def _content_length(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None

