import json, numpy as np
from pathlib import Path

summary = json.load(open('sprint5_summary.json'))
for sym in summary:
    result_path = Path(f'{sym}_ensemble_result.json')
    if not result_path.exists():
        print(f"{sym:12} — skipped (no ensemble_result.json)")
        continue
    er = json.load(open(result_path))
    y_cls = np.array(er['y_cls_test'])
    naive_baseline = max(y_cls.mean(), 1 - y_cls.mean()) * 100
    model_acc = summary[sym].get('test_dir_acc', 0)
    gap = model_acc - naive_baseline
    print(f"{sym:12} naive={naive_baseline:.1f}%  model={model_acc:.1f}%  gap={gap:+.1f}%")