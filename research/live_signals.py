"""Сканер живых сигналов: свежие 1h бары Binance -> 4h -> порт стратегии с параметрами по умолчанию.

Сигнал на последнем ЗАКРЫТОМ 4h-баре исполняется на open следующего бара (то есть сейчас).
Размер позиции: риск 1% от 50 000 USD = 500 USD на стоп.
"""
import json, ssl, sys, time, urllib.request
import certifi, pandas as pd
import kle
from kle import P

CRYPTO = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "LTCUSDT", "LINKUSDT", "SOLUSDT"]
CTX = ssl.create_default_context(cafile=certifi.where())
RISK_USD, BUDGET = 500.0, 50_000.0


def fresh(sym):
    old = kle.load(sym, "1h")
    t = int(old.index[-1].timestamp() * 1000) + 3600_000
    rows = []
    while True:
        u = f"https://data-api.binance.vision/api/v3/klines?symbol={sym}&interval=1h&startTime={t}&limit=1000"
        b = json.load(urllib.request.urlopen(u, timeout=30, context=CTX))
        if not b:
            break
        rows += b
        t = b[-1][0] + 3600_000
        if len(b) < 1000:
            break
    if rows:
        n = pd.DataFrame(rows).iloc[:, :6]
        n.columns = ["time", "open", "high", "low", "close", "volume"]
        n["time"] = pd.to_datetime(n["time"], unit="ms", utc=True)
        n = n.set_index("time").astype(float)
        old = pd.concat([old, n])
    old = old[~old.index.duplicated()]
    d4 = old.resample("4h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    now = pd.Timestamp.now(tz="UTC")
    return d4[d4.index + pd.Timedelta("4h") <= now]            # только закрытые бары


if __name__ == "__main__":
    print("Время запуска (UTC):", pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S"))
    p = P(start="2018-01-01", end="2099-12-31", cost_pct=0.08)
    for sym in CRYPTO:
        d4 = fresh(sym)
        st = {}
        t, _ = kle.run(d4, p, stats=st)
        last = d4.index[-1]
        px = d4.close.iloc[-1]
        line = f"{sym}: последний закрытый 4h бар {last:%m-%d %H:%M}, close {px:,.2f}"
        pend = st.get("pending")
        if pend:
            stop = pend["stop"]
            risk = abs(px - stop)
            qty = RISK_USD / risk
            notional = qty * px
            line += (f"\n   СИГНАЛ {'LONG' if pend['dir'] > 0 else 'SHORT'} ({pend['setup']}, {pend['kind']}, оценка {pend['score']:.0f}): "
                     f"вход ~{px:,.2f}, SL {stop:,.2f}, TP {pend['tp']:,.2f}, риск/ед {risk:,.2f}, "
                     f"объём по правилу {qty:.4f}, номинал {notional:,.0f} USD")
        elif len(t) and t.reason.iloc[-1] == "eod":
            r = t.iloc[-1]
            line += f"\n   открыта: {'LONG' if r.dir > 0 else 'SHORT'} {r.setup}, вход {r.entry_px:,.2f}, SL {r.stop0:,.2f}, TP {r.tp0:,.2f}"
        else:
            line += "\n   сигнала нет"
        print(line)
