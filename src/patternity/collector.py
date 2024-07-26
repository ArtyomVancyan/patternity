from datetime import datetime, timedelta
from time import mktime
from typing import Literal

import numpy as np
from binance.client import Client
from yfinance import download

Interval = Literal["1m", "5m", "15m", "30m", "1h", "1d", "1w", "1M"]


def to_minutes(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1])
    elif interval.endswith("h"):
        return int(interval[:-1]) * 60
    elif interval.endswith("d"):
        return int(interval[:-1]) * 1440
    elif interval.endswith("w"):
        return int(interval[:-1]) * 10080
    elif interval.endswith("M"):
        return int(interval[:-1]) * 43200
    else:
        raise ValueError("Unsupported interval format")


def get_crypto(symbol: str, interval: Interval = "1h", depth: int = 1000) -> np.array:
    """Collects historical data of a cryptocurrency from Binance."""

    period = 0
    klines = []
    client = Client()
    minutes = to_minutes(interval)

    while len(klines) < depth:
        end = int(mktime((datetime.now() - timedelta(minutes=minutes * period * 1000)).timetuple()) * 1000)
        klines = [
            *(client.get_klines(symbol=symbol, endTime=end, interval=interval, limit=1000)),
            *klines,
        ]
        period += 1

    return np.array([d[4] for d in klines[-depth:]], dtype=np.float64)


def get_stock(symbol: str, interval: Interval = "1h", depth: int = 1000) -> np.array:
    """Collects historical data of a stock from Yahoo Finance."""

    interval = {
        Client.KLINE_INTERVAL_1WEEK: "1wk",
        Client.KLINE_INTERVAL_1MONTH: "1mo",
    }.get(interval, interval)

    end_date = datetime.now()
    start_date = end_date - timedelta(minutes=depth * to_minutes(interval))
    data = download(symbol, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), interval=interval)

    return np.array(data["Close"].values[-depth:], dtype=np.float64)
