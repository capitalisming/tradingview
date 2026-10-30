"""События стратегии и признаки момента сигнала для мета-разметки.

Все признаки считаются по данным до закрытия сигнальной свечи включительно (индекс sig_i).
Вход происходит на open следующего бара, поэтому утечки будущего нет.
"""
from dataclasses import replace
import numpy as np
import pandas as pd
import kle
from kle import P

CRYPTO = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "LTCUSDT", "LINKUSDT", "SOLUSDT"]
TFS = ["4h", "1d"]
EVENT_PARAMS = P(min_touches=2, min_score=0, trend_mode="off", use_vol_flt=False, cost_pct=0.08,
                 start="2018-01-01", end="2099-12-31")

FEATURES = ["setup_bounce", "setup_breakout", "setup_retest", "kind_trend", "touches", "score", "stop_atr",
            "atr_pct", "atr_ratio", "adx", "ema_dist", "ret1", "ret3", "ret10", "ret30", "vol_ratio",
            "range_pos", "hour", "dow", "dir"]


def events_for(symbol, tf):
    df = kle.load(symbol, tf)
    p = EVENT_PARAMS
    d = kle.prepare(df, p)
    tr, _ = kle.run(df, p, d)
    if not len(tr):
        return pd.DataFrame()
    c = df.close.to_numpy()
    h, l = df.high.to_numpy(), df.low.to_numpy()
    atr = np.array(d["atr"])
    atr_sma = np.array(d["atr_sma"])
    adx = np.array(d["adx"])
    ema = np.array(d["ema"])
    vol = df.volume.to_numpy()
    vol_sma = np.array(d["vol_sma"])
    i = tr.sig_i.to_numpy()
    dr = tr.dir.to_numpy()
    a = atr[i]
    out = pd.DataFrame({
        "symbol": symbol, "tf": tf, "sig_i": i, "dir": dr,
        "sig_time": df.index[i], "entry_time": tr.entry_time.to_numpy(), "exit_time": tr.exit_time.to_numpy(),
        "r": tr.r.to_numpy(), "win": (tr.r.to_numpy() > 0).astype(int), "reason": tr.reason.to_numpy(),
        "setup_bounce": (tr.setup == "bounce").astype(int).to_numpy(),
        "setup_breakout": (tr.setup == "breakout").astype(int).to_numpy(),
        "setup_retest": (tr.setup == "retest").astype(int).to_numpy(),
        "kind_trend": (tr.kind == "trend").astype(int).to_numpy(),
        "touches": tr.touches.to_numpy(), "score": tr.score.to_numpy(),
        "stop_atr": np.abs(c[i] - tr.stop0.to_numpy()) / a,
        "atr_pct": a / c[i], "atr_ratio": a / atr_sma[i], "adx": adx[i],
        "ema_dist": dr * (c[i] - ema[i]) / a,
        "vol_ratio": np.where(vol_sma[i] > 0, vol[i] / vol_sma[i], np.nan),
        "hour": df.index[i].hour, "dow": df.index[i].dayofweek,
    })
    for k in (1, 3, 10, 30):
        out[f"ret{k}"] = dr * (c[i] - c[np.maximum(i - k, 0)]) / a
    rng = np.array([(c[j] - l[max(j - 49, 0):j + 1].min()) / max(h[max(j - 49, 0):j + 1].max() - l[max(j - 49, 0):j + 1].min(), 1e-12) for j in i])
    out["range_pos"] = rng
    return out


def build():
    parts = [events_for(s, tf) for s in CRYPTO for tf in TFS]
    ev = pd.concat([p for p in parts if len(p)], ignore_index=True)
    ev["entry_time"] = pd.to_datetime(ev.entry_time, utc=True)
    ev["exit_time"] = pd.to_datetime(ev.exit_time, utc=True)
    ev = ev.dropna(subset=FEATURES + ["r"]).sort_values("entry_time").reset_index(drop=True)
    return ev


if __name__ == "__main__":
    ev = build()
    ev.to_csv("ml_events.csv.gz", index=False)
    cut = pd.Timestamp("2023-01-01", tz="UTC")
    print("всего событий:", len(ev), "| до 2023:", int((ev.entry_time < cut).sum()), "| с 2023:", int((ev.entry_time >= cut).sum()))
    print(ev.groupby("tf").size().to_dict(), "| win rate до 2023: %.3f" % ev[ev.entry_time < cut].win.mean())
    print("пропуски в признаках:", int(ev[FEATURES].isna().sum().sum()))
