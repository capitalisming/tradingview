"""Контроль метода на данных 2018–2022: с перемешанными метками обучения «преимущества» быть не должно."""
import numpy as np, pandas as pd
import ml_wf as W
ev = W.load("is")
real = W.walk(ev, "M2", [2020, 2021, 2022]); obs = real[real.sel].r.mean() - real.r.mean()
rng = np.random.default_rng(11); diffs = []
for s in range(25):
    e = ev.copy()
    # перемешиваем метки только внутри обучающей части: метки событий переставляются среди ранних событий каждого года
    e["win_true"] = e.win
    perm = rng.permutation(len(e))
    e["win"] = e.win.to_numpy()[perm]
    rows = []
    for y in [2020, 2021, 2022]:
        start = pd.Timestamp(f"{y}-01-01", tz="UTC"); end = pd.Timestamp(f"{y+1}-01-01", tz="UTC")
        tr = e[e.exit_time < start]; te = e[(e.entry_time >= start) & (e.entry_time < end)].copy()
        m = W.make("M2").fit(tr[W.FEATURES], tr.win)
        te["p"] = m.predict_proba(te[W.FEATURES])[:, 1]
        k = int(np.ceil(W.SHARE * len(te))); te["sel"] = False
        te.loc[te.p.sort_values(ascending=False).index[:k], "sel"] = True; rows.append(te)
    r = pd.concat(rows); diffs.append(r[r.sel].r.mean() - r.r.mean())
d = np.array(diffs)
print(f"Реальные метки: разность {obs:+.3f}")
print(f"Перемешанные метки, 25 повторов: среднее {d.mean():+.3f}, станд. откл. {d.std():.3f}, максимум {d.max():+.3f}, доля >= реальной: {(d >= obs).mean():.2f}")
