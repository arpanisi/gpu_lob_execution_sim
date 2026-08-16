from __future__ import annotations

import gzip

from data.sources import orderbook_url, trade_url
from data.trades import load_bybit_trade_gzip


def test_locked_bybit_url_patterns() -> None:
    assert trade_url("2024-01-01") == "https://public.bybit.com/trading/BTCUSDT/BTCUSDT2024-01-01.csv.gz"
    assert orderbook_url("2024-01-01") == "https://quote-saver.bycsi.com/orderbook/linear/BTCUSDT/2024-01-01_BTCUSDT_ob500.data.zip"


def test_bybit_trade_parser_converts_archive_seconds_to_internal_milliseconds(tmp_path) -> None:
    sample = tmp_path / "trades.csv.gz"
    sample.write_bytes(
        gzip.compress(
            b"timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,foreignNotional\n"
            b"1704067200.2353,BTCUSDT,Sell,0.002,42324.90,PlusTick,t1,0,0,0\n"
        )
    )
    trades = load_bybit_trade_gzip(sample)
    assert trades[0].timestamp == 1_704_067_200_235
    assert trades[0].aggressor_side == "sell"
    assert trades[0].quantity == 0.002
    assert trades[0].price == 42324.90
