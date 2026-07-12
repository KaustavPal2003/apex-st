import json, csv, numpy as np
from pathlib import Path

summary = json.load(open('sprint5_summary.json'))

# ── model_performance.csv ─────────────────────────────────────────────
with open('logs/model_performance.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Symbol','Dir Acc %','RMSE','Corr','Calibration Status','Empirical Coverage'])
    for sym, d in summary.items():
        conf_path = Path(f'{sym}_conformal.json')
        if not conf_path.exists():
            continue
        conf = json.load(open(conf_path))
        w.writerow([
            sym,
            d.get('test_dir_acc',''),
            d.get('test_rmse',''),
            d.get('test_corr',''),
            'well-calibrated' if conf.get('status','').startswith('✅') else 'check calibration',
            round(conf.get('empirical_coverage',0)*100, 1)
        ])

# ── calibration_summary.csv ───────────────────────────────────────────
with open('logs/calibration_summary.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Symbol','Nominal Coverage','Empirical Coverage','Half Width','Calibrated'])
    for sym in summary:
        conf_path = Path(f'{sym}_conformal.json')
        if not conf_path.exists():
            continue
        conf = json.load(open(conf_path))
        w.writerow([
            sym,
            f"{conf.get('nominal_coverage',0)*100:.0f}%",
            f"{conf.get('empirical_coverage',0)*100:.1f}%",
            round(conf.get('half_width',0), 4),
            'Yes' if conf.get('status','').startswith('✅') else 'No'
        ])

# ── cusum_summary.csv ─────────────────────────────────────────────────
with open('logs/cusum_summary.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Symbol','Baseline Mean','Baseline Std','Threshold h','In-sample Flags','Max S+','Max S-'])
    for sym in summary:
        cs_path = Path(f'{sym}_cusum_state.json')
        if not cs_path.exists():
            continue
        cs = json.load(open(cs_path))
        w.writerow([
            sym,
            round(cs.get('mu0',0), 6),
            round(cs.get('sigma0',0), 6),
            cs.get('h',''),
            cs.get('insample_drift_flags',''),
            round(cs.get('max_s_pos_insample',0), 3),
            round(cs.get('max_s_neg_insample',0), 3)
        ])

# ── drift_analysis (H1 vs H2 test window) ────────────────────────────
print(f"\n{'Symbol':12} {'Gap H1':>8} {'Gap H2':>8} {'Drift?':>8}")
print("-" * 44)

for sym in summary:
    p = Path(f'{sym}_ensemble_result.json')
    if not p.exists():
        continue
    er     = json.load(open(p))
    y_cls  = np.array(er['y_cls_test'])
    y_pred = (np.array(er['cls_pred_test']) > 0.5).astype(int)

    n, mid = len(y_cls), len(y_cls) // 2

    y1, p1 = y_cls[:mid], y_pred[:mid]
    naive1 = max(y1.mean(), 1 - y1.mean()) * 100
    gap1   = (p1 == y1).mean() * 100 - naive1

    y2, p2 = y_cls[mid:], y_pred[mid:]
    naive2 = max(y2.mean(), 1 - y2.mean()) * 100
    gap2   = (p2 == y2).mean() * 100 - naive2

    drift = "YES ⚠" if gap2 < gap1 - 3 else "—"
    print(f"{sym:12} {gap1:>+7.1f}% {gap2:>+7.1f}%  {drift}")

print("\nGenerated: logs/model_performance.csv")
print("Generated: logs/calibration_summary.csv")
print("Generated: logs/cusum_summary.csv")