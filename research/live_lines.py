"""Подтверждённые линии стратегии на последнем закрытом 4h-баре: ближайшие к цене."""
import pandas as pd
import kle
from kle import P
from live_signals import fresh, CRYPTO

p = P(start="2018-01-01", end="2099-12-31", cost_pct=0.08)
print("Время (UTC):", pd.Timestamp.now(tz="UTC").strftime("%H:%M"))
for sym in CRYPTO:
    d4 = fresh(sym)
    st = {}
    kle.run(d4, p, stats=st)
    px = d4.close.iloc[-1]
    a = kle.prepare(d4, p)["atr"][-1]
    rows = []
    for kind, side, v, touches, score, state, age in st["lines"]:
        if touches >= 3 and score >= 40 and state == 0:
            rows.append(("поддержка" if side > 0 else "сопротивление", "трендовая" if kind == 1 else "уровень", v, (v - px) / px * 100, touches, score, age))
    rows.sort(key=lambda r: abs(r[3]))
    print(f"\n{sym}: цена {px:,.4g}, ATR(4h) {a:,.4g} ({a / px * 100:.2f}%)")
    for r in rows[:4]:
        print(f"   {r[0]:13s} {r[1]:9s} {r[2]:>12,.4g}  {r[3]:+6.2f}%  касаний {r[4]}  оценка {r[5]:.0f}  без касания {r[6]} баров")
