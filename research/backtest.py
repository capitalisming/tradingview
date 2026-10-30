"""Прогоны: python3 backtest.py is | sens | oos | null. Правило: OOS запускается один раз, после фиксации параметров."""
import sys, zlib
from dataclasses import replace
from multiprocessing import Pool
import numpy as np, pandas as pd
import kle
from kle import P

CRYPTO = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "LTCUSDT", "LINKUSDT", "SOLUSDT"]
TRAD = {"SPY": .02, "QQQ": .02, "GLD": .02, "TLT": .02, "EURUSD": .01, "GBPUSD": .01}
UNIVERSE = [(s, tf, .08) for s in CRYPTO for tf in ("1h", "4h", "1d")] + [(s, "1d", c) for s, c in TRAD.items()]
IS = ("2018-01-01", "2023-01-01")
OOS = ("2023-01-01", "2026-10-30")
_cache = {}


def data(s, tf):
    if (s, tf) not in _cache:
        _cache[(s, tf)] = kle.load(s, tf)
    return _cache[(s, tf)]


def one(job):
    s, tf, cost, p, null_seed = job
    df = data(s, tf)
    p = replace(p, cost_pct=cost * p.cost_pct)      # p.cost_pct здесь — множитель издержек
    d = kle.prepare(df, p)
    ns = None
    if null_seed is not None:
        base, _ = kle.run(df, p, d)
        win = np.where((df.index >= pd.Timestamp(p.start, tz="UTC")) & (df.index < pd.Timestamp(p.end, tz="UTC")))[0]
        r = np.random.default_rng(null_seed + zlib.crc32(f"{s}{tf}".encode()) % 100000)
        k = max(len(base), 1) * 3                   # с запасом: часть входов отсеют фильтры/позиция
        ns = {int(i): int(x) for i, x in zip(r.choice(win, min(k, len(win)), replace=False), r.choice([-1, 1], k))} if len(win) else {}
    st = {}
    t, eq = kle.run(df, p, d, stats=st, null_sig=ns)
    m = kle.metrics(t, eq, p)
    m.update(symbol=s, tf=tf, stats=st, bh=(df.close[df.index < p.end].iloc[-1] / df.close[df.index >= p.start].iloc[0] - 1) * 100
             if len(df[df.index >= p.start]) else np.nan)
    if len(t):
        a, ax = np.array(d["atr"]), np.array(d["adx"])
        atr_sma = np.array(d["atr_sma"])
        i = t.sig_i.to_numpy()
        t = t.assign(symbol=s, tf=tf, adx=ax[i], atr_ratio=a[i] / atr_sma[i], year=pd.to_datetime(t.entry_time).dt.year)
    return m, t


def batch(p, pool, jobs=None, null_seed=None):
    res = pool.map(one, [(s, tf, c, p, null_seed) for s, tf, c in (jobs or UNIVERSE)], chunksize=1)
    rows = pd.DataFrame([r[0] for r in res])
    tr = pd.concat([r[1] for r in res if len(r[1])], ignore_index=True) if any(len(r[1]) for r in res) else pd.DataFrame()
    return rows, tr


def boot_ci(x, n=4000, seed=7):
    x = np.asarray(x, float)
    if len(x) < 5:
        return (np.nan, np.nan)
    r = np.random.default_rng(seed)
    m = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def block_ci(tr, n=4000, seed=7):
    """Бутстрэп по календарным кварталам: сделки на разных монетах коррелированы, обычный CI слишком узок."""
    q = pd.to_datetime(tr.entry_time).dt.tz_localize(None).dt.to_period("Q").astype(str).to_numpy()
    groups = [tr.r.to_numpy()[q == k] for k in np.unique(q)]
    r = np.random.default_rng(seed)
    out = [np.concatenate([groups[i] for i in r.integers(0, len(groups), len(groups))]).mean() for _ in range(n)]
    return tuple(np.percentile(out, [2.5, 97.5]))


def summ(tr):
    if not len(tr):
        return dict(n=0)
    r = tr.r.to_numpy()
    lo, hi = block_ci(tr)
    return dict(n=len(r), win=round((r > 0).mean() * 100, 1), avgR=round(r.mean(), 3), lo=round(lo, 3), hi=round(hi, 3),
                pfR=round(r[r > 0].sum() / max(-r[r <= 0].sum(), 1e-9), 2))


def split_table(tr, col):
    return pd.DataFrame({k: summ(g) for k, g in tr.groupby(col)}).T


def report(name, rows, tr):
    print(f"\n=== {name} ===\nВсе сделки:", summ(tr))
    print(split_table(tr.assign(side=np.where(tr.dir > 0, "long", "short")), "side").to_string())
    print(split_table(tr, "setup").to_string())
    print(split_table(tr, "kind").to_string())
    print(split_table(tr, "tf").to_string())
    print(rows[["symbol", "tf", "trades", "win", "pf", "ret", "dd", "avg_r", "n_long", "n_short", "long_r", "short_r", "bh"]].round(2).to_string())


if __name__ == "__main__":
    mode = sys.argv[1]
    base = P(cost_pct=1.0)                          # множитель к издержкам инструмента
    IS_P, OOS_P = replace(base, start=IS[0], end=IS[1]), replace(base, start=OOS[0], end=OOS[1])
    with Pool() as pool:
        if mode == "is":
            rows, tr = batch(IS_P, pool)
            report("IN-SAMPLE, значения по умолчанию", rows, tr)
            print(pd.DataFrame(list(rows.stats)).sum().to_string())
        elif mode == "null":
            rows, tr = batch(IS_P, pool)
            print("Стратегия IS:", summ(tr))
            for sd in (1, 2, 3):
                r2, t2 = batch(IS_P, pool, null_seed=sd)
                print(f"Случайные входы, seed {sd}:", summ(t2))
        elif mode == "sens":
            variants = {
                "default": {}, "pivot 3/3": dict(pivot_left=3, pivot_right=3), "pivot 8/8": dict(pivot_left=8, pivot_right=8),
                "touches 2": dict(min_touches=2), "touches 4": dict(min_touches=4),
                "score 0": dict(min_score=0), "score 55": dict(min_score=55),
                "rr 1.5": dict(rr=1.5), "rr 3": dict(rr=3.0),
                "только отскок": dict(enable_break=False), "только пробой": dict(enable_bounce=False),
                "пробой: confirm": dict(break_mode="confirm"), "пробой: retest": dict(break_mode="retest"),
                "только тренд.линии": dict(use_horiz=False), "только уровни": dict(use_trend=False),
                "без фильтра тренда": dict(trend_mode="off"), "без ATR-фильтра": dict(use_vol_flt=False),
                "SL ATR": dict(sl_mode="atr"), "TP уровень": dict(tp_mode="level"),
                "trailing 1R": dict(tp_mode="trail", trail_r=1.0), "BE 1R": dict(be_r=1.0),
                "partial 50% 1R": dict(partial_pct=50), "reverse": dict(allow_reverse=True), "объём": dict(use_volume=True),
            }
            out = {}
            for k, v in variants.items():
                _, tr = batch(replace(IS_P, **v), pool)
                out[k] = summ(tr)
                print(k, out[k], flush=True)
        elif mode == "diag":
            rows, tr = batch(IS_P, pool)
            tr["stop_pct"] = (tr.entry_px - tr.stop0).abs() / tr.entry_px * 100
            print("Медиана стопа, % цены и цена издержек в R:")
            print(tr.groupby("tf").stop_pct.median().round(3).to_string())
            print("\nGross (без издержек) против случайных входов по ТФ:")
            g, tg = batch(replace(IS_P, cost_pct=0.0), pool)
            nul = [batch(replace(IS_P, cost_pct=0.0), pool, null_seed=sd)[1] for sd in (1, 2, 3, 4, 5)]
            n = pd.concat(nul, ignore_index=True)
            for tf in ("1h", "4h", "1d"):
                print(tf, "стратегия", summ(tg[tg.tf == tf]), "\n   случайные", summ(n[n.tf == tf]))
            print("\nGross по сетапам:\n", split_table(tg, "setup").to_string())
            print("\nОценка линии → результат (корзины):")
            tr["sb"] = pd.cut(tr.score, [0, 50, 60, 70, 80, 100])
            print(split_table(tr, "sb").to_string())
            print("корреляция score~R:", round(tr[["score", "r"]].corr().iloc[0, 1], 4))
            print("\nКасания:")
            print(split_table(tr.assign(tb=tr.touches.clip(upper=6)), "tb").to_string())
            tr["reg"] = np.select([tr.adx >= 25, tr.adx < 20], ["тренд ADX>=25", "боковик ADX<20"], "переход")
            tr["vol"] = np.select([tr.atr_ratio >= 1.3, tr.atr_ratio < 0.85], ["высокая вол.", "низкая вол."], "обычная вол.")
            print("\nРежим рынка:\n", split_table(tr, "reg").to_string())
            print("\nВолатильность:\n", split_table(tr, "vol").to_string())
            print("\nСетап × режим (4h+1d):")
            print(split_table(tr[tr.tf != "1h"].assign(c=lambda x: x.setup + " | " + x.reg), "c").to_string())
            print("\nГоды:\n", split_table(tr, "year").to_string())
        elif mode == "oos":
            v2 = dict(enable_bounce=False, adx_gate=True)
            J2 = [j for j in UNIVERSE if j[1] != "1h"]
            r1, t1 = batch(OOS_P, pool)
            report("OOS V1 (умолчания, все ТФ)", r1, t1)
            r2, t2 = batch(replace(OOS_P, **v2), pool, jobs=J2)
            report("OOS V2 (4h+1d, ADX-режимы)", r2, t2)
            ri, ti = batch(replace(IS_P, **v2), pool, jobs=J2)
            print("\nV2 на in-sample:", summ(ti))
            n1 = pd.concat([batch(OOS_P, pool, null_seed=s)[1] for s in (1, 2, 3, 4, 5)], ignore_index=True)
            print("OOS случайные, все ТФ:", summ(n1))
            print("OOS случайные, 4h+1d:", summ(n1[n1.tf != "1h"]))
            print("\nV1 OOS годы:\n", split_table(t1, "year").to_string())
            print("\nV2 OOS годы:\n", split_table(t2, "year").to_string())
            print("\nЧувствительность V2 OOS к издержкам:")
            for m in (0.0, 1.0, 2.0):
                print(m, summ(batch(replace(OOS_P, cost_pct=m, **v2), pool, jobs=J2)[1]))
            print("\nV2 OOS только крипто:", summ(t2[t2.symbol.str.endswith("USDT")]), "\nV2 OOS только ETF/FX:", summ(t2[~t2.symbol.str.endswith("USDT")]))
        elif mode == "extra":
            r1, t = batch(OOS_P, pool)
            t["reg"] = np.select([t.adx >= 25, t.adx < 20], ["тренд ADX>=25", "боковик ADX<20"], "переход")
            t["vol"] = np.select([t.atr_ratio >= 1.3, t.atr_ratio < 0.85], ["высокая вол.", "низкая вол."], "обычная вол.")
            print("OOS V1 режим:\n", split_table(t, "reg").to_string(), "\n", split_table(t, "vol").to_string())
            print("\nДубли сигналов (symbol, tf, sig_i):", t.duplicated(["symbol", "tf", "sig_i"]).sum())
            s = pd.DataFrame(list(r1.stats)).sum()
            print("Пробоев линий:", s.breaks, "из них ложных (возврат до входа):", s.false_breaks, f"= {s.false_breaks / s.breaks:.1%}")
            print("Причины выхода:", t.reason.value_counts().to_dict())
            sl = t[t.reason == "sl"]
            print("Стопы хуже -1.5R (гэпы):", int((sl.r < -1.5).sum()), "из", len(sl), "; худший R", round(t.r.min(), 2))
            print("Сделок в год на инструмент/ТФ:", (t.groupby(["symbol", "tf"]).size().groupby("tf").mean() / 3.83).round(1).to_dict())
        elif mode == "paired":
            # Парный блочный бутстрэп разницы «стратегия минус случайные входы»: кварталы ресемплируются
            # одновременно для обеих выборок, поэтому общая рыночная компонента сокращается.
            def qgroups(tr):
                q = pd.to_datetime(tr.entry_time).dt.tz_localize(None).dt.to_period("Q").astype(str).to_numpy()
                return {k: tr.r.to_numpy()[q == k] for k in np.unique(q)}

            def paired(a, b, n=4000, seed=11):
                ga, gb = qgroups(a), qgroups(b)
                ks = sorted(set(ga) | set(gb))
                r = np.random.default_rng(seed)
                ds = []
                for _ in range(n):
                    pick = [ks[i] for i in r.integers(0, len(ks), len(ks))]
                    xa = np.concatenate([ga.get(k, np.empty(0)) for k in pick])
                    xb = np.concatenate([gb.get(k, np.empty(0)) for k in pick])
                    if len(xa) and len(xb):
                        ds.append(xa.mean() - xb.mean())
                return round(a.r.mean() - b.r.mean(), 3), tuple(np.round(np.percentile(ds, [2.5, 97.5]), 3))

            for name, P_, seeds in (("IS", IS_P, (1, 2, 3, 4, 5)), ("OOS", OOS_P, (1, 2, 3, 4, 5))):
                _, t = batch(P_, pool)
                nul = pd.concat([batch(P_, pool, null_seed=sd)[1] for sd in seeds], ignore_index=True)
                for lab, f in (("все ТФ", lambda x: x), ("1h", lambda x: x[x.tf == "1h"]), ("4h", lambda x: x[x.tf == "4h"]),
                               ("1d", lambda x: x[x.tf == "1d"]), ("4h+1d", lambda x: x[x.tf != "1h"])):
                    a, b = f(t), f(nul)
                    print(name, lab, "стратегия", round(a.r.mean(), 3), "(n=%d)" % len(a), "случайные", round(b.r.mean(), 3),
                          "(n=%d)" % len(b), "разность и 95% CI:", paired(a, b))
            # V2 против случайных входов (OOS, 4h+1d)
            v2 = dict(enable_bounce=False, adx_gate=True)
            J2 = [j for j in UNIVERSE if j[1] != "1h"]
            _, t2 = batch(replace(OOS_P, **v2), pool, jobs=J2)
            nul = pd.concat([batch(OOS_P, pool, jobs=J2, null_seed=sd)[1] for sd in (1, 2, 3, 4, 5)], ignore_index=True)
            print("OOS V2 против случайных 4h+1d: разность и 95% CI:", paired(t2, nul))
        elif mode == "ref":
            df = data("BTCUSDT", "4h")
            t, _ = kle.run(df, replace(OOS_P, cost_pct=0.08))
            cols = ["entry_time", "dir", "setup", "kind", "entry_px", "stop0", "tp0", "exit_time", "exit_px", "reason", "r"]
            t[cols].to_csv("ref_trades_BTCUSDT_4h.csv", index=False)
            print("Эталон BTCUSDT 4h, окно OOS:", len(t), "сделок; причины выхода:", t.reason.value_counts().to_dict())
        elif mode == "hist":
            df = data("BTCUSDT", "4h")
            p0 = replace(OOS_P, cost_pct=0.08)
            base = set(kle.run(df, p0)[0].entry_time)
            print("BTC 4h OOS, сделок на полной истории:", len(base), "(сравнение порта с самим собой на обрезанной истории, не с Pine)")
            for cut in ("2022-12-01", "2022-10-01", "2022-06-01", "2021-01-01"):
                x = set(kle.run(df[df.index >= cut], p0)[0].entry_time)
                print(f"история с {cut}: сделок {len(x)}, совпало {len(x & base)}")
