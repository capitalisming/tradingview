"""Загрузка реальных OHLCV: Binance spot (crypto, 1h -> 4h/1d) и Yahoo Finance (ETF/FX, 1d)."""
import json, ssl, time, urllib.request, pathlib
import certifi
import pandas as pd

OUT = pathlib.Path(__file__).parent / "data"
TLS = ssl.create_default_context(cafile=certifi.where())
CRYPTO = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "LTCUSDT", "LINKUSDT", "SOLUSDT"]
YF = {"SPY": "SPY", "QQQ": "QQQ", "GLD": "GLD", "TLT": "TLT", "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X"}
START = int(pd.Timestamp("2017-08-01", tz="UTC").timestamp() * 1000)
END = int(pd.Timestamp("2026-10-30", tz="UTC").timestamp() * 1000)


def binance(symbol):
    rows, t = [], START
    while t < END:
        u = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=1h&startTime={t}&limit=1000"
        with urllib.request.urlopen(u, timeout=30, context=TLS) as r:
            batch = json.load(r)
        if not batch:
            break
        rows += batch
        t = batch[-1][0] + 3600_000
        if len(batch) < 1000:
            break
        time.sleep(0.05)
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.set_index("time").astype(float)
    return df[df.index < pd.Timestamp(END, unit="ms", tz="UTC")]


def resample(df, rule):
    return df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


for s in CRYPTO:
    df = binance(s)
    df.to_csv(OUT / f"{s}_1h.csv.gz")
    resample(df, "4h").to_csv(OUT / f"{s}_4h.csv.gz")
    resample(df, "1D").to_csv(OUT / f"{s}_1d.csv.gz")
    print(s, len(df), df.index[0], df.index[-1], flush=True)

import yfinance as yf
for name, tick in YF.items():
    d = yf.download(tick, start="2004-01-01", end="2026-10-30", interval="1d", auto_adjust=True, progress=False)
    d.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in d.columns]
    d = d[["open", "high", "low", "close", "volume"]].dropna()
    d.index = pd.to_datetime(d.index, utc=True).rename("time")
    d.to_csv(OUT / f"{name}_1d.csv.gz")
    print(name, len(d), d.index[0], d.index[-1], flush=True)
