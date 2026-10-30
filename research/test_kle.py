"""Проверки логики порта: причинность (нет заглядывания), зеркальность Long/Short, сценарии."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import kle
from kle import P, run

BASE = P(trend_mode="off", use_vol_flt=False, min_touches=2, min_score=0, cooldown=0, start="2000-01-01")


def frame(o, h, l, c, v=None, freq="4h"):
    n = len(c)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v if v is not None else np.ones(n)},
                        index=pd.date_range("2020-01-01", periods=n, freq=freq, tz="UTC"))


def walk(n=6000, seed=1, vol=0.008):
    r = np.random.default_rng(seed)
    # режимы: тренд/боковик + толстые хвосты, чтобы появлялись и линии, и пробои
    drift = np.repeat(r.choice([-1, 0, 0, 1], size=n // 300 + 1), 300)[:n] * vol * 0.08
    ret = drift + r.standard_t(4, n) * vol * 0.6
    c = 100 * np.exp(np.cumsum(ret))
    o = np.r_[c[0], c[:-1]] * (1 + r.normal(0, vol * 0.05, n))
    h = np.maximum(o, c) * (1 + np.abs(r.normal(0, vol * 0.5, n)))
    l = np.minimum(o, c) * (1 - np.abs(r.normal(0, vol * 0.5, n)))
    return frame(o, h, l, c, r.lognormal(0, .4, n))


def key(t):
    return list(zip(t.sig_i, t.dir, t.setup)) if len(t) else []


def test_prefix_invariance():
    df = walk()
    full, _ = run(df, BASE)
    assert len(full) > 30, len(full)
    for k in (1500, 2500, 3500, 4500, 5500):
        pre, _ = run(df.iloc[:k], BASE)
        a = [x for x in key(full) if x[0] < k - 1]
        b = [x for x in key(pre) if x[0] < k - 1]
        assert a == b, (k, len(a), len(b))


def test_future_perturbation():
    df = walk(seed=2)
    k = 3000
    full, _ = run(df, BASE)
    alt = df.copy()
    alt.iloc[k:, :4] = alt.iloc[k:, :4].to_numpy()[::-1] * 1.37
    other, _ = run(alt, BASE)
    a = [x for x in key(full) if x[0] < k - 1]
    b = [x for x in key(other) if x[0] < k - 1]
    assert a == b and len(a) > 5


def test_mirror_symmetry():
    df = walk(seed=3)
    K = df[["high"]].to_numpy().max() * 1.1
    m = pd.DataFrame({"open": K - df.open, "high": K - df.low, "low": K - df.high, "close": K - df.close, "volume": df.volume}, index=df.index)
    p = replace_cost(BASE)
    a, _ = run(df, p)
    b, _ = run(m, p)
    ka = [(i, d, s) for i, d, s in key(a)]
    kb = [(i, -d, s) for i, d, s in key(b)]
    match = len(set(ka) & set(kb)) / max(len(ka), len(kb))
    assert match > 0.9, match


def replace_cost(p):
    from dataclasses import replace
    return replace(p, cost_pct=0.0)


def test_no_overlap_and_sane_levels():
    df = walk(seed=4)
    t, eq = run(df, BASE)
    assert (t.entry_i.to_numpy()[1:] > t.exit_i.to_numpy()[:-1] - 1).all()      # одна позиция за раз
    assert (t.entry_i > t.sig_i).all()                                         # вход строго после сигнала
    assert (t.dir * (t.entry_px - t.stop0) > -1e-9).sum() >= 0.98 * len(t)     # стоп с нужной стороны (кроме гэпов)
    sl = t[t.reason == "sl"]
    assert (sl.r < 0).all()
    assert abs(eq.iloc[-1] - (BASE.capital + t.pnl.sum())) < 1e-6              # учёт капитала сходится


def flat_bars(n, px=100.0, spread=1.0):
    return [px] * n, [px + spread] * n, [px - spread] * n, [px] * n


def bounce_scene(tail):
    """Боковик 100-110 с тремя подтверждёнными минимумами, затем сценарий tail."""
    c = []
    for _ in range(3):
        c += list(np.linspace(108, 101, 12)) + list(np.linspace(101, 108, 13)[1:-1])
    c = list(np.r_[np.full(130, 108.0), c, tail])
    c = np.array(c)
    o = np.r_[c[0], c[:-1]]
    r = np.random.default_rng(7)      # без точных равенств соседних экстремумов
    h = np.maximum(o, c) + 0.3 + r.uniform(0, .02, len(c))
    l = np.minimum(o, c) - 0.3 - r.uniform(0, .02, len(c))
    return frame(o, h, l, c)


def test_bounce_long_on_tested_support():
    tail = list(np.linspace(108, 101.0, 12)) + [103.5, 105, 107, 109, 111, 113, 115, 117, 119, 121]
    t, _ = run(bounce_scene(tail), BASE)
    assert len(t) and (t.dir > 0).any() and "bounce" in set(t.setup[t.dir > 0]), t


def test_false_break_does_not_enter_short():
    # закрытие под поддержкой и немедленный возврат: брейкаут-входа в Short быть не должно
    p = replace_cost(BASE)
    tail = list(np.linspace(108, 101.5, 12)) + [97.5, 103.5, 106, 108, 108, 108]
    t, _ = run(bounce_scene(tail), p)
    n0 = len(bounce_scene([108.0]))
    late = t[t.sig_i >= n0 - 1] if len(t) else t
    assert not len(late) or not ((late.dir < 0) & (late.setup == "breakout")).any(), late


def test_gap_through_stop_fills_at_open():
    df = bounce_scene(list(np.linspace(108, 101.0, 12)) + [103.5, 105, 106])
    t0, _ = run(df, replace_cost(BASE))
    assert len(t0)
    tr = t0[(t0.dir > 0)].iloc[-1]
    i = int(tr.entry_i) + 1
    d2 = df.iloc[:i].copy()
    gap_px = tr.stop0 * 0.9
    extra = frame([gap_px] * 5, [gap_px + .1] * 5, [gap_px - .1] * 5, [gap_px] * 5)
    extra.index = pd.date_range(d2.index[-1] + pd.Timedelta("4h"), periods=5, freq="4h", tz="UTC")
    t1, _ = run(pd.concat([d2, extra]), replace_cost(BASE))
    x = t1[t1.entry_i == tr.entry_i].iloc[0]
    assert x.reason == "sl" and abs(x.exit_px - gap_px) < 1e-9 and x.r < -1.5, x


def test_cooldown_blocks_reentry():
    from dataclasses import replace
    df = walk(seed=5)
    t, _ = run(df, replace(BASE, cooldown=25))
    gaps = t.sig_i.to_numpy()[1:] - t.exit_i.to_numpy()[:-1]
    assert (gaps >= 25).all()


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name)
            except Exception as e:
                fails += 1; print("FAIL", name, type(e).__name__, str(e)[:400])
    sys.exit(fails)
