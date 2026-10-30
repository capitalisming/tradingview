"""Мета-разметка: walk-forward на 2018–2022 (фаза 1) и одиночный прогон 2023–2026 (фаза 2).

python ml_wf.py is   — фаза 1, не читает события с выходом после 2022-12-31
python ml_wf.py oos  — фаза 2, выбранная модель обучена на данных до 2023, запускать один раз
"""
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
FEATURES = ["setup_bounce", "setup_breakout", "setup_retest", "kind_trend", "touches", "score", "stop_atr",
            "atr_pct", "atr_ratio", "adx", "ema_dist", "ret1", "ret3", "ret10", "ret30", "vol_ratio",
            "range_pos", "hour", "dow", "dir"]
CUT = pd.Timestamp("2023-01-01", tz="UTC")
SHARE = 0.30


def make(name):
    if name == "M1":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=0)


def load(phase):
    ev = pd.read_csv("ml_events.csv.gz", parse_dates=["entry_time", "exit_time"])
    ev["entry_time"] = pd.to_datetime(ev.entry_time, utc=True)
    ev["exit_time"] = pd.to_datetime(ev.exit_time, utc=True)
    if phase == "is":
        ev = ev[ev.exit_time < CUT]          # фаза 1 не видит ничего, что заканчивается в 2023+
    return ev.reset_index(drop=True)


def walk(ev, model_name, test_years):
    """Для каждого года: обучение на событиях с выходом до начала года, отбор top-SHARE по p(win)."""
    rows = []
    for y in test_years:
        start = pd.Timestamp(f"{y}-01-01", tz="UTC")
        end = pd.Timestamp(f"{y + 1}-01-01", tz="UTC")
        tr = ev[ev.exit_time < start]
        te = ev[(ev.entry_time >= start) & (ev.entry_time < end)].copy()
        if len(tr) < 150 or len(te) < 20 or tr.win.nunique() < 2:
            continue
        m = make(model_name).fit(tr[FEATURES], tr.win)
        te["p"] = m.predict_proba(te[FEATURES])[:, 1]
        k = int(np.ceil(SHARE * len(te)))
        te["sel"] = False
        te.loc[te.p.sort_values(ascending=False).index[:k], "sel"] = True
        te["year"] = y
        te["ntrain"] = len(tr)
        rows.append(te)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def quarter_ci(res, n=4000, seed=5):
    q = res.entry_time.dt.tz_localize(None).dt.to_period("Q").astype(str).to_numpy()
    keys = np.unique(q)
    gs = {k: (res.r.to_numpy()[(q == k) & res.sel.to_numpy()], res.r.to_numpy()[q == k]) for k in keys}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        a = np.concatenate([gs[keys[i]][0] for i in pick])
        b = np.concatenate([gs[keys[i]][1] for i in pick])
        if len(a):
            diffs.append(a.mean() - b.mean())
    return np.percentile(diffs, [2.5, 97.5])


def perm_p(res, n=5000, seed=3):
    """Доля случайных подвыборок той же доли внутри каждого года, у которых средний R не ниже наблюдаемого."""
    obs = res[res.sel].r.mean()
    rng = np.random.default_rng(seed)
    groups = [g.r.to_numpy() for _, g in res.groupby("year")]
    ks = [int(res[(res.year == y) & res.sel].shape[0]) for y, _ in res.groupby("year")]
    cnt = 0
    for _ in range(n):
        pooled = np.concatenate([rng.choice(g, size=k, replace=False) for g, k in zip(groups, ks)])
        cnt += pooled.mean() >= obs
    return cnt / n


def report(res, label):
    sel = res[res.sel]
    lo, hi = quarter_ci(res)
    auc = roc_auc_score(res.win, res.p) if res.win.nunique() > 1 else np.nan
    print(f"{label}: всего {len(res)}, отобрано {len(sel)}, "
          f"R отобранных {sel.r.mean():+.3f}, R всех {res.r.mean():+.3f}, разность {sel.r.mean() - res.r.mean():+.3f} "
          f"CI [{lo:+.3f}; {hi:+.3f}], перестановочный p={perm_p(res):.3f}, AUC={auc:.3f}, win отобранных {sel.win.mean():.3f}")
    by = res.groupby("year").apply(lambda g: pd.Series({"n": len(g), "all": g.r.mean(), "sel": g[g.sel].r.mean()}), include_groups=False)
    print(by.round(3).to_string())
    return sel.r.mean() - res.r.mean()


if __name__ == "__main__":
    phase = sys.argv[1]
    if phase == "is":
        ev = load("is")
        print(f"События фазы 1: {len(ev)}, win {ev.win.mean():.3f}, средний R {ev.r.mean():+.3f}")
        out = {}
        for name in ("M1", "M2"):
            res = walk(ev, name, [2020, 2021, 2022])
            out[name] = report(res, name)
            res.to_csv(f"ml_phase1_{name}.csv", index=False)
        pick = "M1" if abs(out["M1"] - out["M2"]) <= 0.02 or out["M1"] > out["M2"] else "M2"
        print("Для фазы 2 выбрана модель по правилу предрегистрации:", pick)
        open("ml_choice.txt", "w").write(pick)
    elif phase == "oos":
        pick = open("ml_choice.txt").read().strip()
        ev = load("oos")
        res = walk(ev, pick, [2023, 2024, 2025, 2026])
        # обучение один раз на данных до 2023: поэтому повторяем обучение только на exit_time < 2023-01-01
        tr = ev[ev.exit_time < CUT]
        te = ev[ev.entry_time >= CUT].copy()
        m = make(pick).fit(tr[FEATURES], tr.win)
        te["p"] = m.predict_proba(te[FEATURES])[:, 1]
        te["year"] = te.entry_time.dt.year
        te["sel"] = False
        for y, g in te.groupby("year"):
            k = int(np.ceil(SHARE * len(g)))
            te.loc[g.p.sort_values(ascending=False).index[:k], "sel"] = True
        sel = te[te.sel]
        lo, hi = quarter_ci(te)
        pv = perm_p(te)
        print(f"ФАЗА 2, модель {pick}, обучена на {len(tr)} событиях до 2023")
        print(f"Всего событий {len(te)}, отобрано {len(sel)}, R отобранных {sel.r.mean():+.3f}, R всех {te.r.mean():+.3f}, "
              f"разность {sel.r.mean() - te.r.mean():+.3f}, CI разности [{lo:+.3f}; {hi:+.3f}], перестановочный p={pv:.3f}")
        c1, c2, c3, c4 = sel.r.mean() > 0, lo > 0, pv < 0.05, len(sel) >= 150
        print("Условия: R>0:", c1, "| нижняя граница CI>0:", c2, "| p<0.05:", c3, "| n>=150:", c4)
        print("ИТОГ:", "ПРЕИМУЩЕСТВО ПОДТВЕРЖДЕНО" if all([c1, c2, c3, c4]) else "преимущество НЕ подтверждено")
        te.to_csv("ml_phase2_result.csv", index=False)
