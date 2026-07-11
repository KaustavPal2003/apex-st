# save as generate_docs.py and run it
import json, csv, pathlib

# Sprint 5 summary table
summary = json.load(open('sprint5_summary.json'))
with open('logs/model_performance.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Symbol','Dir Acc %','RMSE','Corr','Calibration Status','Empirical Coverage'])
    for sym, d in summary.items():
        conf = json.load(open(f'{sym}_conformal.json'))
        w.writerow([
            sym,
            d.get('test_dir_acc',''),
            d.get('test_rmse',''),
            d.get('test_corr',''),
            conf.get('status','').replace('✅','').replace('⚠','').strip(),
            round(conf.get('empirical_coverage',0)*100, 1)
        ])

# Conformal calibration table
with open('logs/calibration_summary.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Symbol','Nominal Coverage','Empirical Coverage','Half Width','Calibrated'])
    for sym in summary:
        conf = json.load(open(f'{sym}_conformal.json'))
        w.writerow([
            sym,
            f"{conf.get('nominal_coverage',0)*100:.0f}%",
            f"{conf.get('empirical_coverage',0)*100:.1f}%",
            round(conf.get('half_width',0), 4),
            'Yes' if conf.get('status','').startswith('✅') else 'No'
        ])

# CUSUM drift summary
with open('logs/cusum_summary.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Symbol','Baseline Mean','Baseline Std','Threshold h','In-sample Flags','Max S+','Max S-'])
    for sym in summary:
        cs = json.load(open(f'{sym}_cusum_state.json'))
        w.writerow([
            sym,
            round(cs.get('mu0',0), 6),
            round(cs.get('sigma0',0), 6),
            cs.get('h',''),
            cs.get('insample_drift_flags',''),
            round(cs.get('max_s_pos_insample',0), 3),
            round(cs.get('max_s_neg_insample',0), 3)
        ])

print("Generated: logs/model_performance.csv")
print("Generated: logs/calibration_summary.csv")
print("Generated: logs/cusum_summary.csv")