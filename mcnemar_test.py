import json, numpy as np
from pathlib import Path
from statsmodels.stats.contingency_tables import mcnemar

summary = json.load(open('sprint5_summary.json'))
bonferroni = 0.05 / sum(1 for sym in summary
                        if Path(f'{sym}_ensemble_result.json').exists())

print(f"{'Symbol':12} {'b':>5} {'c':>5} {'p-value':>10} {'Sig (Bonf)':>12}")
print("-" * 48)

for sym in summary:
    p = Path(f'{sym}_ensemble_result.json')
    if not p.exists():
        continue
    er     = json.load(open(p))
    y_true = np.array(er['y_cls_test'])
    y_pred = (np.array(er['cls_pred_test']) > 0.5).astype(int)
    naive  = np.ones_like(y_true) * int(y_true.mean() >= 0.5)

    b = int(np.sum((y_pred == y_true) & (naive != y_true)))
    c = int(np.sum((y_pred != y_true) & (naive == y_true)))

    if b + c == 0:
        print(f"{sym:12} {'—':>5} {'—':>5} {'—':>10} {'—':>12}")
        continue

    result = mcnemar([[0, b], [c, 0]], exact=True)
    sig    = "YES *" if result.pvalue < bonferroni else "—"
    print(f"{sym:12} {b:>5} {c:>5} {result.pvalue:>10.4f} {sig:>12}")

print(f"\nBonferroni threshold: p < {bonferroni:.4f}")