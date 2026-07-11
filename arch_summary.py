# save as arch_summary.py
import torch, json
from pathlib import Path

with open('logs/architecture_summary.txt', 'w') as f:
    f.write("APEX-ST Architecture Summary\n")
    f.write("="*50 + "\n\n")
    
    # Model sizes
    symbols = json.load(open('watchlist.json'))['watchlist']
    f.write("Sprint 2 — GRU Model Parameters\n")
    f.write("-"*30 + "\n")
    for sym in symbols[:3]:  # sample 3
        ckpt = Path(f'{sym}_apex_st_best.pt')
        if ckpt.exists():
            state = torch.load(ckpt, map_location='cpu')
            params = sum(p.numel() for p in state['model_state'].values())
            f.write(f"  {sym}: {params:,} parameters\n")
    
    f.write("\nSprint 5 — XGBoost Ensemble\n")
    f.write("-"*30 + "\n")
    import xgboost as xgb
    for sym in symbols[:3]:
        reg = xgb.Booster()
        reg.load_model(f'{sym}_ensemble_xgb.json')
        f.write(f"  {sym} reg: {reg.num_boosted_rounds()} trees\n")
    
    f.write("\nData Summary\n")
    f.write("-"*30 + "\n")
    import numpy as np
    for sym in symbols[:3]:
        X = np.load(f'{sym}_apex_X_test.npy')
        f.write(f"  {sym} test set: {X.shape[0]} samples × {X.shape[1]} timesteps × {X.shape[2]} features\n")

print("Generated: logs/architecture_summary.txt")