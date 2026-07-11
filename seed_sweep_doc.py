# save as seed_sweep_doc.py
import subprocess, json, re

results = {}
for seed in [1, 2, 3]:
    for sym in ['BAJFINANCE', 'SUNPHARMA', 'NESTLEIND', 'ADANIENT']:
        out = subprocess.run(
            ['python', 'apex_synth_runner_v2.py', '--epochs', '18',
             '--seed', str(seed), '--sprint2-only', sym],
            capture_output=True, text=True
        ).stdout
        match = re.search(r'Dir.*?(\d+\.\d+)%', out)
        if match:
            results.setdefault(sym, {})[f'seed_{seed}'] = float(match.group(1))

import csv
with open('logs/seed_stability.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Symbol','Seed 1','Seed 2','Seed 3','Spread'])
    for sym, vals in results.items():
        vals_list = list(vals.values())
        spread = round(max(vals_list) - min(vals_list), 2)
        w.writerow([sym] + vals_list + [spread])
print("Generated: logs/seed_stability.csv")