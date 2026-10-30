"""Побарный порт key_lines_strategy.pine + эмулятор исполнения, повторяющий модель
брокера TradingView (market на open следующего бара, стопы/лимиты по пути O-H-L-C,
гэп через стоп исполняется по open). Порядок шагов совпадает с Pine-скриптом.

Это исследовательский инструмент, а не замена Strategy Tester: см. README.
"""
from dataclasses import dataclass, replace
import math
import numpy as np
import pandas as pd

ST_ACTIVE, ST_BROKEN = 0, 1
K_TREND, K_HORIZ = 1, 2
SIG_BOUNCE, SIG_BREAK, SIG_RETEST = 1, 2, 3
SETUP = {SIG_BOUNCE: "bounce", SIG_BREAK: "breakout", SIG_RETEST: "retest"}
PIVOT_KEEP, POOL_MAX = 24, 40


@dataclass(frozen=True)
class P:
    allow_long: bool = True
    allow_short: bool = True
    start: str = "2018-01-01"
    end: str = "2099-12-31"
    atr_len: int = 14
    pivot_left: int = 5
    pivot_right: int = 5
    min_touches: int = 3
    touch_tol: float = 0.25
    merge_atr: float = 0.5
    min_gap: int = 10
    max_span: int = 300
    max_age: int = 150
    max_slope: float = 0.5
    use_trend: bool = True
    use_horiz: bool = True
    min_score: float = 40
    w_touch: float = 1
    w_react: float = 1
    w_fresh: float = 1
    w_span: float = 1
    enable_bounce: bool = True
    enable_break: bool = True
    break_mode: str = "both"       # confirm | retest | both
    break_dist: float = 0.25
    min_body: float = 0.5
    max_pierce: float = 0.5
    reject_frac: float = 0.6
    retest_bars: int = 12
    use_volume: bool = False
    vol_mult: float = 1.2
    cooldown: int = 10
    adx_gate: bool = False
    break_adx_max: float = 20
    retest_adx_min: float = 25
    trend_mode: str = "ema"        # off | ema
    ema_len: int = 200
    use_vol_flt: bool = True
    atr_min: float = 0.6
    atr_max: float = 2.5
    risk_pct: float = 1.0
    max_exposure: float = 100
    max_trades_day: int = 3
    allow_reverse: bool = False
    sl_mode: str = "swing"         # swing | atr
    sl_lookback: int = 5
    sl_buf: float = 0.25
    sl_atr: float = 1.5
    min_stop: float = 0.5
    max_stop: float = 3.0
    tp_mode: str = "rr"            # rr | level | trail
    rr: float = 2.0
    min_rr: float = 1.5
    partial_pct: float = 0
    partial_r: float = 1.0
    be_r: float = 0
    trail_r: float = 0
    trail_atr: float = 2.0
    capital: float = 10000
    cost_pct: float = 0.08         # на каждое исполнение


class Line:
    __slots__ = ("kind", "side", "x1", "y1", "slope", "touches", "first", "last", "react",
                 "state", "break_bar", "false_breaks", "flips", "last_sig")

    def __init__(self, kind, side, x1, y1, slope, touches, first, last, react):
        self.kind, self.side, self.x1, self.y1, self.slope = kind, side, x1, y1, slope
        self.touches, self.first, self.last, self.react = touches, first, last, react
        self.state, self.break_bar, self.false_breaks, self.flips = ST_ACTIVE, 0, 0, 0
        self.last_sig = -1_000_000

    def at(self, x):
        return self.y1 + self.slope * (x - self.x1)


_NULL_LINE = Line(K_HORIZ, 1, 0, 0.0, 0.0, 0, 0, 0, 0.0)


def line_score(l, now, p):
    t = min((l.touches - 1) / 4.0, 1.0)
    r = min(l.react / max(l.touches, 1) / 3.0, 1.0)
    f = max(1.0 - (now - l.last) / p.max_age, 0.0)
    d = min((l.last - l.first) / (p.max_span * 0.5), 1.0)
    w = max(p.w_touch + p.w_react + p.w_fresh + p.w_span, 1e-4)
    return 100.0 * (p.w_touch * t + p.w_react * r + p.w_fresh * f + p.w_span * d) / w


def rma(x, n):
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = x[:n].mean()
    a = 1.0 / n
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def sma(x, n):
    return pd.Series(x).rolling(n).mean().to_numpy()


def adx(h, l, c, n=14):
    """Эквивалент третьего значения ta.dmi(n, n)."""
    up, dn = np.r_[0, h[1:] - h[:-1]], np.r_[0, l[:-1] - l[1:]]
    pdm, ndm = np.where((up > dn) & (up > 0), up, 0.), np.where((dn > up) & (dn > 0), dn, 0.)
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    a = rma(tr, n)
    pdi, ndi = 100 * rma(pdm, n) / a, 100 * rma(ndm, n) / a
    dx = 100 * np.abs(pdi - ndi) / np.maximum(pdi + ndi, 1e-9)
    out = np.full(len(c), np.nan)
    if len(c) > n:
        out[n - 1:] = rma(dx[n - 1:], n)
    return out


def prepare(df, p):
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    tr[0] = h[0] - l[0]
    a = rma(tr, p.atr_len)
    L, R = p.pivot_left, p.pivot_right
    w = L + R + 1
    n = len(c)
    ph, pl = np.full(n, np.nan), np.full(n, np.nan)
    # Pivot проверяется на баре t для кандидата t-R: информация справа уже существует.
    for t in range(w - 1, n):
        k = t - R
        hw, lw = h[t - w + 1:t + 1], l[t - w + 1:t + 1]
        if h[k] == hw.max() and (hw == h[k]).sum() == 1:
            ph[t] = h[k]
        if l[k] == lw.min() and (lw == l[k]).sum() == 1:
            pl[t] = l[k]
    s = pd.Series
    return dict(o=o.tolist(), h=h.tolist(), l=l.tolist(), c=c.tolist(), v=v.tolist(),
                atr=a.tolist(), atr_sma=sma(a, 100).tolist(), ema=ema(c, p.ema_len).tolist(),
                ph=ph.tolist(), pl=pl.tolist(), adx=adx(h, l, c).tolist(),
                right_hi=s(h).rolling(R).max().tolist(), right_lo=s(l).rolling(R).min().tolist(),
                swing_hi=s(h).rolling(p.sl_lookback).max().tolist(),
                swing_lo=s(l).rolling(p.sl_lookback).min().tolist(),
                vol_sma=sma(v, 20).tolist(), n=n)


def register_pivot(i, pb, pp, react, store, side, pool, a, c, p):
    tol = p.touch_tol * a
    touched = False
    for l in pool:
        if l.kind == K_TREND and l.state == ST_ACTIVE and l.side == side and abs(pp - l.at(pb)) <= tol:
            l.touches += 1
            l.last = pb
            l.react += react
            touched = True
    if p.use_horiz:
        best, best_d = None, p.merge_atr * a
        for l in pool:
            if l.kind == K_HORIZ:
                d = abs(pp - l.y1)
                if d <= best_d:
                    best_d, best = d, l
        if best is None:
            pool.append(Line(K_HORIZ, 1 if c[i] >= pp else -1, pb, pp, 0.0, 1, pb, pb, react))
        else:
            best.y1 = (best.y1 * best.touches + pp) / (best.touches + 1)
            best.touches += 1
            best.last = max(best.last, pb)
            best.react += react
    if p.use_trend and not touched and store:
        best_t, best_s, best_q, best_slope, best_r = 0, 0, None, 0.0, 0.0
        for q in reversed(store):
            span = pb - q[0]
            if span > p.max_span:
                break
            if span >= p.min_gap and side * (pp - q[1]) > 0:
                slope = (pp - q[1]) / span
                if abs(slope) <= p.max_slope * a:
                    valid = True
                    back = i - q[0]
                    for k in range(back + 1):
                        lv = q[1] + slope * (back - k)
                        if side * (c[i - k] - lv) < -tol:
                            valid = False
                            break
                    if valid:
                        t, rs = 2, react + q[2]
                        for r in store:
                            if q[0] < r[0] < pb and abs(r[1] - (q[1] + slope * (r[0] - q[0]))) <= tol:
                                t += 1
                                rs += r[2]
                        if t > best_t or (t == best_t and span > best_s):
                            best_t, best_s, best_q, best_slope, best_r = t, span, q, slope, rs
        if best_q is not None:
            pool.append(Line(K_TREND, side, best_q[0], best_q[1], best_slope, best_t, best_q[0], pb, best_r))
    store.append((pb, pp, react))
    if len(store) > PIVOT_KEEP:
        store.pop(0)
    if len(pool) > POOL_MAX:
        worst = min(range(len(pool)), key=lambda j: line_score(pool[j], i, p) + (100.0 if pool[j].touches >= p.min_touches else 0.0))
        pool.pop(worst)


def finish(pool, idx):
    l = pool[idx]
    if l.kind == K_TREND:
        pool.pop(idx)
    else:
        l.side, l.state = -l.side, ST_ACTIVE
        l.flips += 1
        if l.flips >= 2:
            pool.pop(idx)


def run(df, p=P(), d=None, stats=None, null_sig=None):
    """Возвращает (trades, equity). stats собирает счётчики отбраковки.
    null_sig: {бар: направление} — случайные входы вместо сигналов (нулевая гипотеза)
    с теми же стопами, целями, фильтрами и сайзингом."""
    d = d or prepare(df, p)
    O, H, L, C, V = d["o"], d["h"], d["l"], d["c"], d["v"]
    ATR, ASMA, EMA = d["atr"], d["atr_sma"], d["ema"]
    PH, PL = d["ph"], d["pl"]
    idx = df.index
    days = idx.floor("D")
    ts_start, ts_end = pd.Timestamp(p.start, tz="UTC"), pd.Timestamp(p.end, tz="UTC")
    in_win = (idx >= ts_start) & (idx < ts_end)
    n = d["n"]
    pool, piv_h, piv_l = [], [], []
    trades, equity = [], np.full(n, np.nan)
    cash, pos, avg = p.capital, 0.0, 0.0
    t_dir, t_sig, t_entry, t_stop, t_tp, t_tp1, t_risk, t_ext, t_filled = 0, 0, 0.0, 0.0, math.nan, 0.0, 0.0, 0.0, False
    tp1_done = False
    pend = None                     # отложенный market-вход
    close_next = False              # strategy.close_all в конце окна: исполнение на open следующего бара
    cur = None                      # запись текущей сделки
    last_exit, day_count = -1_000_000, 0
    cost = p.cost_pct / 100
    st = stats if stats is not None else {}
    for key in ("sig_long", "sig_short", "conflict", "skip_trend", "skip_vol", "skip_day", "skip_cooldown",
                "skip_inpos", "skip_stop_wide", "skip_tp", "entered", "breaks", "false_breaks"):
        st.setdefault(key, 0)

    def fill_close(i, q, px, reason):
        nonlocal cash, pos, cur
        cash += (px - avg) * q * (1 if pos > 0 else -1) - q * px * cost
        pos -= q if pos > 0 else -q
        cur["pnl"] += (px - avg) * q * (1 if cur["dir"] > 0 else -1) - q * px * cost
        cur["exit_i"], cur["exit_px"], cur["reason"] = i, px, reason
        if abs(pos) < 1e-12:
            pos = 0.0
            cur["exit_time"] = idx[i]
            cur["r"] = cur["pnl"] / cur["risk_cash"]
            trades.append(cur)
            cur = None

    for i in range(n):
        o, h, l, c = O[i], H[i], L[i], C[i]
        # ── брокер: исполнение на этом баре ──
        if close_next and pos != 0:
            fill_close(i, abs(pos), o, "period_end")
        close_next = False
        if pend is not None:
            if pos != 0:  # разворот
                fill_close(i, abs(pos), o, "reverse")
            q = pend["qty"]
            cash -= q * o * cost
            pos, avg = q * pend["dir"], o
            cur = {"dir": pend["dir"], "entry_i": i, "entry_time": idx[i], "entry_px": o, "qty": q, "setup": pend["setup"],
                   "kind": pend["kind"], "score": pend["score"], "touches": pend["touches"], "risk_cash": max(pend["risk"] * q, 1e-9),
                   "pnl": -q * o * cost, "sig_i": pend["sig_i"], "stop0": pend["stop"], "tp0": pend["tp"]}
            pend = None
        if pos != 0 and t_dir != 0:
            dr = 1 if pos > 0 else -1
            path = (o, h, l, c) if abs(h - o) < abs(l - o) else (o, l, h, c)
            # гэп за стоп/цель на открытии
            if dr * (o - t_stop) <= 0:
                fill_close(i, abs(pos), o, "sl")
            elif not math.isnan(t_tp) and dr * (o - t_tp) >= 0:
                fill_close(i, abs(pos), o, "tp")
            else:
                for a, b in zip(path[:-1], path[1:]):
                    if pos == 0:
                        break
                    if dr * (b - a) > 0:  # движение в сторону прибыли
                        if p.partial_pct > 0 and not tp1_done and dr * (b - t_tp1) >= 0:
                            tp1_done = True
                            q = abs(pos) * p.partial_pct / 100
                            cur["partial"] = True
                            fill_close(i, q, t_tp1, "tp1")
                        if pos != 0 and not math.isnan(t_tp) and dr * (b - t_tp) >= 0:
                            fill_close(i, abs(pos), t_tp, "tp")
                    elif dr * (b - t_stop) <= 0:
                        fill_close(i, abs(pos), t_stop, "sl")

        # ── скрипт: расчёт на закрытии бара ──
        if idx[i] >= ts_end and pos != 0:
            close_next = True
        if i > 0 and days[i] != days[i - 1]:
            day_count = 0
        if t_dir != 0 and pos == 0 and i > t_sig:
            t_dir, t_filled, tp1_done, last_exit = 0, False, False, i
        a = ATR[i]
        if t_dir != 0 and pos * t_dir > 0:
            if not t_filled:
                t_filled = True
                t_entry = avg
                t_risk = abs(t_entry - t_stop)
            t_ext = max(t_ext, h) if t_dir > 0 else min(t_ext, l)
            gain = t_dir * (t_ext - t_entry)
            if p.be_r > 0 and gain >= p.be_r * t_risk:
                t_stop = max(t_stop, t_entry) if t_dir > 0 else min(t_stop, t_entry)
            if p.trail_r > 0 and gain >= p.trail_r * t_risk:
                tr = t_ext - t_dir * p.trail_atr * a
                t_stop = max(t_stop, tr) if t_dir > 0 else min(t_stop, tr)

        as_ = ASMA[i]
        if not (a == a and as_ == as_):
            equity[i] = cash + pos * (c - avg)
            continue
        tol, brk, pierce = p.touch_tol * a, p.break_dist * a, p.max_pierce * a
        rng = max(h - l, 1e-12)
        body = abs(c - o) / rng
        cp = C[i - 1] if i else c
        vol_break = (not p.use_volume) or not (V[i] > 0) or V[i] > d["vol_sma"][i] * p.vol_mult
        l_type = s_type = 0
        l_score = s_score = -1.0
        l_line = s_line = None

        for j in range(len(pool) - 1, -1, -1):
            ln = pool[j]
            if i - ln.last > p.max_age:
                pool.pop(j)
                continue
            s = ln.side
            v, vp = ln.at(i), ln.at(i - 1)
            near, far = (l, h) if s == 1 else (h, l)
            c_pos = (c - l) / rng if s == 1 else (h - c) / rng
            sc = line_score(ln, i, p)
            ok = ln.touches >= p.min_touches and sc >= p.min_score and i - ln.last_sig >= p.cooldown
            sig_t = sig_d = 0
            if ln.state == ST_ACTIVE:
                if s * (c - v) < -brk:
                    first = s * (cp - vp) >= -brk
                    strong = s * (c - o) < 0 and body >= p.min_body and vol_break
                    if ln.touches >= p.min_touches and strong and first:
                        ln.state, ln.break_bar = ST_BROKEN, i
                        st["breaks"] += 1
                    else:
                        finish(pool, j)
                elif p.enable_bounce:
                    touched = -pierce <= s * (near - v) <= tol
                    if touched and s * (cp - vp) > 0 and s * (c - v) > 0 and s * (c - o) > 0 and c_pos >= p.reject_frac:
                        sig_t, sig_d = SIG_BOUNCE, s
            else:
                age = i - ln.break_bar
                if s * (c - v) > 0:
                    ln.state = ST_ACTIVE
                    ln.false_breaks += 1
                    st["false_breaks"] += 1
                    if ln.false_breaks >= 2:
                        finish(pool, j)
                else:
                    done = False
                    if age == 1 and p.break_mode != "retest":
                        if p.enable_break and s * (c - v) < -brk and s * (c - o) < 0:
                            sig_t, sig_d = SIG_BREAK, -s
                        if p.break_mode == "confirm":
                            done = True
                    if sig_t == 0 and p.break_mode != "confirm":
                        if p.enable_break and s * (far - v) >= -tol and s * (c - o) < 0 and 1.0 - c_pos >= p.reject_frac:
                            sig_t, sig_d, done = SIG_RETEST, -s, True
                        elif age >= p.retest_bars:
                            done = True
                    if done:
                        finish(pool, j)
            if p.adx_gate and ((sig_t == SIG_BREAK and not d["adx"][i] < p.break_adx_max) or (sig_t == SIG_RETEST and not d["adx"][i] >= p.retest_adx_min)):
                sig_t = 0
            if sig_t and ok:
                if sig_d > 0 and sc > l_score:
                    l_type, l_score, l_line = sig_t, sc, ln
                if sig_d < 0 and sc > s_score:
                    s_type, s_score, s_line = sig_t, sc, ln

        R = p.pivot_right
        if PL[i] == PL[i]:
            ar = ATR[i - R]
            register_pivot(i, i - R, PL[i], (d["right_hi"][i] - PL[i]) / ar if ar == ar else 0.0, piv_l, 1, pool, a, C, p)
        if PH[i] == PH[i]:
            ar = ATR[i - R]
            register_pivot(i, i - R, PH[i], (PH[i] - d["right_lo"][i]) / ar if ar == ar else 0.0, piv_h, -1, pool, a, C, p)

        eq = cash + pos * (c - avg)
        equity[i] = eq
        st["sig_long"] += l_type > 0
        st["sig_short"] += s_type > 0
        if l_type and s_type:
            st["conflict"] += 1
        dir_sig = 0 if (l_type and s_type) else 1 if l_type else -1 if s_type else 0
        if null_sig is not None:
            dir_sig = null_sig.get(i, 0)
            l_type = s_type = SIG_BOUNCE
            l_line = s_line = _NULL_LINE
        can = False
        if dir_sig and in_win[i]:
            trend_ok = p.trend_mode == "off" or (c > EMA[i] if dir_sig > 0 else c < EMA[i])
            ratio = a / as_
            if not (p.allow_long if dir_sig > 0 else p.allow_short) or not trend_ok:
                st["skip_trend"] += 1
            elif p.use_vol_flt and not (p.atr_min <= ratio <= p.atr_max):
                st["skip_vol"] += 1
            elif day_count >= p.max_trades_day:
                st["skip_day"] += 1
            elif i - last_exit < p.cooldown:
                st["skip_cooldown"] += 1
            elif not (pos == 0 or (p.allow_reverse and pos * dir_sig < 0)):
                st["skip_inpos"] += 1
            else:
                can = True
        if can:
            src = l_line if dir_sig > 0 else s_line
            typ = l_type if dir_sig > 0 else s_type
            score = l_score if dir_sig > 0 else s_score
            stop = ((d["swing_lo"][i] if dir_sig > 0 else d["swing_hi"][i]) - dir_sig * p.sl_buf * a) if p.sl_mode == "swing" else c - dir_sig * p.sl_atr * a
            if dir_sig * (c - stop) < p.min_stop * a:
                stop = c - dir_sig * p.min_stop * a
            risk = abs(c - stop)
            tp = math.nan if p.tp_mode == "trail" else c + dir_sig * p.rr * risk
            tp_ok = True
            if p.tp_mode == "level":
                nearest = None
                for ln in pool:
                    if ln.side == -dir_sig and ln.state == ST_ACTIVE and ln.touches >= p.min_touches:
                        lv = ln.at(i) - dir_sig * tol
                        if dir_sig * (lv - c) > 0 and (nearest is None or dir_sig * (nearest - lv) > 0):
                            nearest = lv
                if nearest is not None:
                    tp = nearest
                    tp_ok = dir_sig * (tp - c) >= p.min_rr * risk
            qty = min(eq * p.risk_pct / 100 / risk, eq * p.max_exposure / 100 / c)
            if risk > p.max_stop * a:
                st["skip_stop_wide"] += 1
            elif not tp_ok:
                st["skip_tp"] += 1
            elif qty > 0:
                st["entered"] += 1
                pend = dict(dir=dir_sig, qty=qty, setup=SETUP[typ], stop=stop, tp=tp, risk=risk, sig_i=i, score=score,
                            kind="trend" if src.kind == K_TREND else "level", touches=src.touches)
                t_dir, t_sig, t_entry, t_stop, t_tp, t_risk, t_ext, t_filled, tp1_done = dir_sig, i, c, stop, tp, risk, c, False, False
                t_tp1 = c + dir_sig * p.partial_r * risk
                day_count += 1
                src.last_sig = i

    st["pending"] = pend            # сигнал на последнем баре, который исполнится на следующем open
    st["lines"] = [(l.kind, l.side, l.at(n - 1), l.touches, line_score(l, n - 1, p), l.state, n - 1 - l.last) for l in pool]
    if cur is not None:             # незакрытая сделка в конце данных
        fill_close(n - 1, abs(pos), C[-1], "eod")
    eq = pd.Series(equity, index=idx).ffill()
    return pd.DataFrame(trades), eq


def metrics(tr, eq, p=P(), bars_per_year=None):
    w = eq[(eq.index >= pd.Timestamp(p.start, tz="UTC")) & (eq.index < pd.Timestamp(p.end, tz="UTC"))].dropna()
    out = dict(trades=len(tr), win=np.nan, pf=np.nan, ret=(w.iloc[-1] / p.capital - 1) * 100 if len(w) else 0.0,
               dd=((w / w.cummax() - 1).min() * 100) if len(w) else 0.0, avg_r=np.nan, avg_win_r=np.nan, avg_loss_r=np.nan,
               long_r=np.nan, short_r=np.nan, n_long=0, n_short=0)
    if len(tr):
        g, ls = tr.pnl[tr.pnl > 0].sum(), -tr.pnl[tr.pnl <= 0].sum()
        out.update(win=(tr.pnl > 0).mean() * 100, pf=g / ls if ls > 0 else np.inf, avg_r=tr.r.mean(),
                   avg_win_r=tr.r[tr.r > 0].mean(), avg_loss_r=tr.r[tr.r <= 0].mean(),
                   n_long=int((tr.dir > 0).sum()), n_short=int((tr.dir < 0).sum()),
                   long_r=tr.r[tr.dir > 0].mean(), short_r=tr.r[tr.dir < 0].mean())
    return out


def load(name, tf):
    return pd.read_csv(f"{__file__.rsplit('/', 1)[0]}/data/{name}_{tf}.csv.gz", index_col=0, parse_dates=True)
