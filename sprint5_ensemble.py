"""
sprint5_ensemble.py — Ensemble, Calibration & Drift Monitoring
═══════════════════════════════════════════════════════════════════════════
Combines the 21 Sprint 4 fusion checkpoints (7 symbols × 3 variants) into
a single production-ready prediction system.

Three components, run independently or together:

  1. ENSEMBLE STACKING
     XGBoost meta-learner trained on the 3 fusion variants' predictions
     (+ engineered meta-features) → one calibrated final prediction.

  2. CONFORMAL PREDICTION
     Distribution-free prediction intervals with guaranteed coverage
     (e.g. "90% of the time, the true 30d return falls in this band").
     No assumptions about the error distribution — works regardless of
     whether residuals are Gaussian, skewed, or heavy-tailed.

  3. CUSUM DRIFT DETECTION
     Cumulative-sum control chart on rolling prediction error.
     Flags when the model's error distribution has shifted — signals
     it's time to retrain rather than trust the current weights.

Usage
─────
  python sprint5_ensemble.py                       # all 7 watchlist symbols
  python sprint5_ensemble.py RELIANCE TCS           # specific symbols
  python sprint5_ensemble.py --skip-drift           # ensemble + conformal only
  python sprint5_ensemble.py --alpha 0.05           # 95% conformal coverage
  python sprint5_ensemble.py --check                # verify saved artifacts

Output files (per symbol)
──────────────────────────
  {SYM}_ensemble_xgb.json          XGBoost meta-learner (regression)
  {SYM}_ensemble_xgb_cls.json      XGBoost meta-learner (classification)
  {SYM}_conformal.json             Calibrated interval half-widths
  {SYM}_cusum_state.json           CUSUM baseline + control limits
  sprint5_summary.json             Cross-symbol comparison table
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG (mirrors sprint4_fusion.py constants)
# ─────────────────────────────────────────────────────────────────────────────
SYMBOLS_DEFAULT = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
VARIANTS        = ['FusionA_Gated', 'FusionB_MLP', 'FusionC_Transformer']
PRICE_DIM       = None  # resolved at runtime from model.encode() output
                        # V2 (active): 64   V1 baseline: 256
SENT_DIM        = 64
GAT_DIM         = 64
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_ALPHA   = 0.10     # 1 - alpha = 90% nominal coverage
CUSUM_K         = 0.5      # CUSUM slack parameter (in std-dev units)
CUSUM_H         = 5.0      # CUSUM decision threshold (in std-dev units)

BANNER = "═" * 68


def get_symbols(args_symbols):
    if args_symbols:
        return [s.upper() for s in args_symbols]
    wl = Path('watchlist.json')
    if wl.exists():
        data = json.loads(wl.read_text()).get('watchlist', [])
        if data:
            return [s.upper() for s in data]
    return SYMBOLS_DEFAULT


# ─────────────────────────────────────────────────────────────────────────────
# LOAD FROZEN EMBEDDINGS + FUSION MODELS  (same logic as sprint4_fusion.py)
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_encode(model) -> None:
    """
    Attach a consistent encode() method to whichever Sprint 2 model was loaded.

    APEXSTModelV2 (apex_synth_runner_v2) already defines encode() natively via
    its st_branch / tf_branch / fusion attributes.

    APEXSTModel (sprint2_model) does not — this shim adds one using the same
    real attribute names (st_branch / tf_branch / fusion).  Note: branch1 and
    branch2 have never existed in either class; the old code referencing them
    was the source of silent random-noise substitution.
    """
    if hasattr(model, 'encode') and callable(model.encode):
        return  # V2: already present

    def _v1_encode(x):
        with torch.no_grad():
            st    = model.st_branch(x)
            tf    = model.tf_branch(x)
            fused = model.fusion(torch.cat([st, tf], dim=-1))
        return fused

    model.encode = _v1_encode


def load_price_encoder(symbol: str):
    """
    Load Sprint 2 checkpoint.  Returns (model, price_embed_dim) where
    price_embed_dim is the actual dimension of model.encode() output, probed
    via a dummy forward pass — NOT the raw feature dim stored in the checkpoint.

      V2 (apex_synth_runner_v2): fusion_dim // 2 = 64
      V1 (sprint2_model):        fusion_dim // 2 = 256
    """
    ckpt_path = Path(f'{symbol}_apex_st_best.pt')
    if not ckpt_path.exists():
        return None, None
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        feat_dim      = ckpt['feat_dim']
        model_version = ckpt.get('model_version', 'v1_baseline')

        sys.path.insert(0, str(Path('.').resolve()))
        if 'v2' in model_version:
            from apex_synth_runner_v2 import APEXSTModelV2
            model = APEXSTModelV2(in_features=feat_dim).to(DEVICE)
        else:
            from sprint2_model import APEXSTModel
            model = APEXSTModel(in_features=feat_dim).to(DEVICE)

        model.load_state_dict(ckpt['model_state'])
        model.eval()
        _ensure_encode(model)

        with torch.no_grad():
            dummy    = torch.zeros(1, 60, feat_dim, device=DEVICE)
            emb_dim  = model.encode(dummy).shape[-1]

        return model, emb_dim
    except Exception as e:
        print(f"  ⚠  Could not load Sprint 2 model: {e}")
        return None, None


def extract_price_embeddings(model, X: np.ndarray) -> np.ndarray:
    """
    Run frozen Sprint 2 encoder via encode() and return (N, embed_dim) array.

    No padding or truncation is applied — the embedding dimension flows
    naturally from the model and must match what the fusion models were
    trained with (guaranteed because sprint4 now saves price_embed_dim and
    sprint5 reads it back when loading fusion checkpoints).
    """
    embs = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            batch = torch.tensor(X[i:i + 256], dtype=torch.float32).to(DEVICE)
            embs.append(model.encode(batch).cpu().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)


def load_split_data(symbol: str, split: str):
    """Loads price/sent/gat arrays + targets for one split. Returns dict or None."""
    files = {
        'X':     f'{symbol}_apex_X_{split}.npy',
        'y_reg': f'{symbol}_apex_y_reg_{split}.npy',
        'y_cls': f'{symbol}_apex_y_cls_{split}.npy',
        'sent':  f'{symbol}_apex_sent_{split}.npy',
        'gat':   f'{symbol}_apex_gat_{split}.npy',
    }
    missing = [v for v in files.values() if not Path(v).exists()]
    if missing:
        return None
    return {k: np.load(v) for k, v in files.items()}


def load_fusion_model(symbol: str, variant: str):
    """Load a trained Sprint 4 fusion checkpoint and rebuild the model."""
    ckpt_path = Path(f'{symbol}_fusion_{variant}_best.pt')
    if not ckpt_path.exists():
        return None
    try:
        sys.path.insert(0, str(Path('.').resolve()))
        from sprint4_fusion import GatedFusion, ConcatFusion, CrossModalTransformer
        cls_map = {
            'FusionA_Gated':       GatedFusion,
            'FusionB_MLP':         ConcatFusion,
            'FusionC_Transformer': CrossModalTransformer,
        }
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        # price_embed_dim was saved by the fixed sprint4; fall back to 64 (V2
        # default) for any checkpoint written before this fix was applied.
        price_dim = ckpt.get('price_embed_dim', 64)
        model = cls_map[variant](price_dim=price_dim).to(DEVICE)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        return model
    except Exception as e:
        print(f"  ⚠  Could not load {variant} for {symbol}: {e}")
        return None


def get_fusion_predictions(model, price_emb, sent, gat):
    """Run a frozen fusion model, return (reg_pred, cls_prob)."""
    # Align all three modalities to the shortest array before stacking.
    # GAT embeddings may have a slightly different row count than Sprint 1/2
    # arrays due to walk-forward split boundary arithmetic differences.
    n = min(len(price_emb), len(sent), len(gat))
    if n < max(len(price_emb), len(sent), len(gat)):
        print(f"    ⚠  Aligning modalities: price={len(price_emb)} "
              f"sent={len(sent)} gat={len(gat)} → trimmed to {n}")
    price_emb, sent, gat = price_emb[:n], sent[:n], gat[:n]

    with torch.no_grad():
        p = torch.tensor(price_emb, dtype=torch.float32).to(DEVICE)
        s = torch.tensor(sent,      dtype=torch.float32).to(DEVICE)
        g = torch.tensor(gat,       dtype=torch.float32).to(DEVICE)
        reg, cls, conf, gates = model(p, s, g)
        return (reg.cpu().numpy(), torch.sigmoid(cls).cpu().numpy(),
                conf.cpu().numpy() if hasattr(conf, 'cpu') else conf)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD META-FEATURES  (3 variants × {reg, cls_prob, conf} + engineered)
# ─────────────────────────────────────────────────────────────────────────────
def build_meta_features(symbol: str, split: str, price_model, available_variants):
    """
    Returns (meta_X, y_reg, y_cls) where meta_X has columns:
      [reg_A, cls_A, conf_A, reg_B, cls_B, conf_B, reg_C, cls_C, conf_C,
       reg_mean, reg_std, cls_mean, agreement]
    """
    data = load_split_data(symbol, split)
    if data is None:
        return None, None, None

    price_emb = extract_price_embeddings(price_model, data['X'])
    sent, gat = data['sent'], data['gat']
    y_reg, y_cls = data['y_reg'].flatten(), data['y_cls'].flatten()

    n = len(y_reg)
    sent  = sent[:n]
    gat   = gat[:n]
    price_emb = price_emb[:n]

    cols = []
    reg_preds, cls_preds = [], []

    for variant in VARIANTS:
        model = available_variants.get(variant)
        if model is None:
            # Fill with neutral values if a variant is missing
            cols += [np.zeros(n), np.full(n, 0.5), np.zeros(n)]
            reg_preds.append(np.zeros(n))
            cls_preds.append(np.full(n, 0.5))
            continue
        reg, cls_p, conf = get_fusion_predictions(model, price_emb, sent, gat)
        reg, cls_p = reg.flatten()[:n], cls_p.flatten()[:n]
        conf = np.asarray(conf).flatten()[:n] if hasattr(conf, '__len__') else np.full(n, float(conf))
        cols += [reg, cls_p, conf]
        reg_preds.append(reg)
        cls_preds.append(cls_p)

    reg_arr = np.stack(reg_preds, axis=1)   # (n, 3)
    cls_arr = np.stack(cls_preds, axis=1)   # (n, 3)

    reg_mean = reg_arr.mean(axis=1)
    reg_std  = reg_arr.std(axis=1)
    cls_mean = cls_arr.mean(axis=1)
    # Agreement: fraction of variants agreeing with majority direction
    cls_votes = (cls_arr > 0.5).astype(int)
    majority  = (cls_votes.mean(axis=1) > 0.5).astype(int)
    agreement = (cls_votes == majority[:, None]).mean(axis=1)

    cols += [reg_mean, reg_std, cls_mean, agreement]
    meta_X = np.stack(cols, axis=1).astype(np.float32)

    # Trim y_reg and y_cls to match meta_X row count.
    # get_fusion_predictions() may have trimmed to a shorter length
    # (min of price/sent/gat) than the original y_reg/y_cls length.
    n_out = len(meta_X)
    y_reg = y_reg[:n_out]
    y_cls = y_cls[:n_out]

    return meta_X, y_reg, y_cls


META_COLUMNS = (
    [f'{v.split("_")[0]}_{m}' for v in VARIANTS for m in ('reg', 'cls', 'conf')] +
    ['reg_mean', 'reg_std', 'cls_mean', 'agreement']
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ENSEMBLE STACKING  (XGBoost meta-learner)
# ─────────────────────────────────────────────────────────────────────────────
def train_ensemble(symbol: str, alpha: float) -> dict:
    print(f"\n{BANNER}")
    print(f"  SPRINT 5 — Ensemble Stacking — {symbol}")
    print(BANNER)

    price_model, price_dim = load_price_encoder(symbol)
    if price_model is None:
        print(f"  ❌ No Sprint 2 checkpoint for {symbol} — skipping")
        return None

    available_variants = {}
    for v in VARIANTS:
        m = load_fusion_model(symbol, v)
        if m is not None:
            available_variants[v] = m
    if not available_variants:
        print(f"  ❌ No Sprint 4 fusion checkpoints found for {symbol}")
        return None
    print(f"  ✓ Loaded {len(available_variants)}/3 fusion variants: "
          f"{list(available_variants.keys())}")

    print(f"\n[1/4] Building meta-features (train/val/test)…")
    splits = {}
    for split in ('train', 'val', 'test'):
        mx, yr, yc = build_meta_features(symbol, split, price_model, available_variants)
        if mx is None:
            print(f"  ❌ Missing data for split '{split}'")
            return None
        splits[split] = (mx, yr, yc)
        print(f"  ✓ {split:<5} meta_X={mx.shape}  y_reg={yr.shape}  y_cls={yc.shape}")

    X_tr, yr_tr, yc_tr = splits['train']
    X_va, yr_va, yc_va = splits['val']
    X_te, yr_te, yc_te = splits['test']

    print(f"\n[2/4] Training XGBoost regression meta-learner…")
    reg_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=42, n_jobs=-1,
        early_stopping_rounds=20, eval_metric='rmse',
    )
    reg_model.fit(X_tr, yr_tr, eval_set=[(X_va, yr_va)], verbose=False)
    reg_pred_te = reg_model.predict(X_te)
    reg_rmse = float(np.sqrt(np.mean((reg_pred_te - yr_te) ** 2)))
    reg_corr = float(np.corrcoef(reg_pred_te, yr_te)[0, 1]) if reg_pred_te.std() > 1e-8 else 0.0
    print(f"  Test RMSE: {reg_rmse:.4f}   Test Corr: {reg_corr:.3f}")

    print(f"\n[3/4] Training XGBoost classification meta-learner…")
    cls_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=42, n_jobs=-1,
        early_stopping_rounds=20, eval_metric='logloss',
    )
    cls_model.fit(X_tr, yc_tr, eval_set=[(X_va, yc_va)], verbose=False)
    cls_pred_te = cls_model.predict_proba(X_te)[:, 1]
    dir_acc = float(((cls_pred_te > 0.5).astype(int) == yc_te.astype(int)).mean() * 100)
    print(f"  Test Directional Accuracy: {dir_acc:.2f}%")

    # Feature importance
    fi = reg_model.feature_importances_
    top_idx = np.argsort(fi)[::-1][:5]
    print(f"\n  Top-5 meta-features (regression):")
    for idx in top_idx:
        print(f"    {META_COLUMNS[idx]:<14} importance={fi[idx]:.3f}")

    print(f"\n[4/4] Saving ensemble models…")
    reg_model.save_model(f'{symbol}_ensemble_xgb.json')
    cls_model.save_model(f'{symbol}_ensemble_xgb_cls.json')

    # Compute residuals on val set for conformal calibration downstream
    reg_pred_va = reg_model.predict(X_va)
    residuals_va = np.abs(reg_pred_va - yr_va)

    result = {
        'symbol': symbol,
        'available_variants': list(available_variants.keys()),
        'test_rmse': round(reg_rmse, 4),
        'test_corr': round(reg_corr, 3),
        'test_dir_acc': round(dir_acc, 2),
        'meta_columns': META_COLUMNS,
        'val_residuals': residuals_va.tolist(),   # used by conformal step
        'reg_pred_test': reg_pred_te.tolist(),
        'y_reg_test': yr_te.tolist(),
        'cls_pred_test': cls_pred_te.tolist(),
        'y_cls_test': yc_te.tolist(),
    }
    # Persist the full prediction arrays so apex_inference.py can serve
    # real last-test-row predictions without re-running the fusion models.
    with open(f'{symbol}_ensemble_result.json', 'w') as f:
        json.dump(result, f)
    print(f"  ✓ Saved {symbol}_ensemble_xgb.json + _cls.json + _ensemble_result.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFORMAL PREDICTION  (split conformal, distribution-free)
# ─────────────────────────────────────────────────────────────────────────────
def conformal_calibrate(symbol: str, ensemble_result: dict, alpha: float) -> dict:
    """
    Split conformal prediction: uses held-out validation residuals to compute
    a single half-width q such that P(|y - yhat| <= q) >= 1 - alpha,
    with finite-sample guarantee regardless of the residual distribution.
    """
    print(f"\n{BANNER}")
    print(f"  SPRINT 5 — Conformal Calibration — {symbol}  (alpha={alpha})")
    print(BANNER)

    residuals = np.array(ensemble_result['val_residuals'])
    n = len(residuals)
    if n == 0:
        print("  ❌ No validation residuals available")
        return None

    # Conformal quantile: ceil((n+1)(1-alpha)) / n -th order statistic
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    half_width = float(np.quantile(residuals, q_level))

    print(f"  Validation residuals : n={n}")
    print(f"  Conformal quantile   : {q_level:.4f}")
    print(f"  Interval half-width  : ±{half_width:.4f}  (z-score units)")

    # Verify empirical coverage on test set
    reg_pred_te = np.array(ensemble_result['reg_pred_test'])
    y_reg_te    = np.array(ensemble_result['y_reg_test'])
    covered = np.abs(reg_pred_te - y_reg_te) <= half_width
    empirical_coverage = float(covered.mean())

    print(f"  Nominal coverage     : {(1-alpha)*100:.1f}%")
    print(f"  Empirical coverage   : {empirical_coverage*100:.1f}%  (test set)")
    status = "✅ well-calibrated" if abs(empirical_coverage - (1-alpha)) < 0.08 else "⚠ check calibration"
    print(f"  Status               : {status}")

    result = {
        'symbol': symbol,
        'alpha': alpha,
        'nominal_coverage': 1 - alpha,
        'half_width': half_width,
        'empirical_coverage': empirical_coverage,
        'n_calibration': n,
    }

    with open(f'{symbol}_conformal.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  ✓ Saved {symbol}_conformal.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. CUSUM DRIFT DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def cusum_baseline(symbol: str, ensemble_result: dict,
                   k: float = CUSUM_K, h: float = CUSUM_H) -> dict:
    """
    Establishes the CUSUM baseline (mean error, std error) from test-set
    residuals, and control limits for future drift detection.

    CUSUM formula (two-sided):
      S+_t = max(0, S+_{t-1} + (e_t - mu_0)/sigma_0 - k)
      S-_t = max(0, S-_{t-1} - (e_t - mu_0)/sigma_0 - k)
      Flag drift if S+_t > h or S-_t > h
    """
    print(f"\n{BANNER}")
    print(f"  SPRINT 5 — CUSUM Drift Baseline — {symbol}")
    print(BANNER)

    reg_pred = np.array(ensemble_result['reg_pred_test'])
    y_reg    = np.array(ensemble_result['y_reg_test'])
    errors   = reg_pred - y_reg

    mu0    = float(errors.mean())
    sigma0 = float(errors.std()) + 1e-8

    print(f"  Baseline error mean  : {mu0:+.4f}")
    print(f"  Baseline error std   : {sigma0:.4f}")
    print(f"  CUSUM k (slack)      : {k}")
    print(f"  CUSUM h (threshold)  : {h}")

    # Run CUSUM over the test set itself as a sanity check (should not drift,
    # since this is in-distribution data the meta-learner was tuned on)
    s_pos, s_neg = 0.0, 0.0
    max_s_pos, max_s_neg = 0.0, 0.0
    drift_points = []
    for i, e in enumerate(errors):
        z = (e - mu0) / sigma0
        s_pos = max(0.0, s_pos + z - k)
        s_neg = max(0.0, s_neg - z - k)
        max_s_pos = max(max_s_pos, s_pos)
        max_s_neg = max(max_s_neg, s_neg)
        if s_pos > h or s_neg > h:
            drift_points.append(i)

    print(f"  Max S+ on test set   : {max_s_pos:.2f}  (threshold {h})")
    print(f"  Max S- on test set   : {max_s_neg:.2f}  (threshold {h})")
    print(f"  In-sample drift flags: {len(drift_points)}/{len(errors)}  "
          f"{'✓ stable' if len(drift_points) == 0 else '⚠ check baseline'}")

    result = {
        'symbol': symbol,
        'mu0': mu0,
        'sigma0': sigma0,
        'k': k,
        'h': h,
        'max_s_pos_insample': max_s_pos,
        'max_s_neg_insample': max_s_neg,
        'insample_drift_flags': len(drift_points),
        'n_insample': len(errors),
    }

    with open(f'{symbol}_cusum_state.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  ✓ Saved {symbol}_cusum_state.json")
    print(f"\n  Usage in production: feed new daily errors through cusum_update()")
    print(f"  in apex_inference.py (Sprint 6) — flags retraining need automatically")
    return result


def cusum_update(new_error: float, state: dict) -> dict:
    """
    Call this in production with each new observed error to update the
    running CUSUM statistics. Returns updated state + drift flag.
    Kept here as a reference implementation for Sprint 6 to import.
    """
    mu0, sigma0 = state['mu0'], state['sigma0']
    k, h = state['k'], state['h']
    z = (new_error - mu0) / sigma0
    s_pos = max(0.0, state.get('s_pos', 0.0) + z - k)
    s_neg = max(0.0, state.get('s_neg', 0.0) - z - k)
    drift = s_pos > h or s_neg > h
    state['s_pos'], state['s_neg'] = s_pos, s_neg
    state['drift_detected'] = drift
    return state


# ─────────────────────────────────────────────────────────────────────────────
# FILE CHECK
# ─────────────────────────────────────────────────────────────────────────────
def check_files(symbols):
    print(f"\n{BANNER}\n  SPRINT 5 FILE CHECK\n{BANNER}")
    all_ok = True
    for sym in symbols:
        for suffix, label in [
            (f'{sym}_ensemble_xgb.json',     'XGBoost regressor'),
            (f'{sym}_ensemble_xgb_cls.json', 'XGBoost classifier'),
            (f'{sym}_conformal.json',        'Conformal calibration'),
            (f'{sym}_cusum_state.json',      'CUSUM baseline'),
        ]:
            f = Path(suffix)
            if f.exists():
                print(f"  ✅  {f.name:<32} {label}")
            else:
                print(f"  ❌  {f.name:<32} MISSING"); all_ok = False
    print(f"\n  {'✅ All present' if all_ok else '❌ Some missing'}")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Sprint 5 — Ensemble + Calibration + Drift')
    p.add_argument('symbols', nargs='*')
    p.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                   help=f'Conformal miscoverage rate (default: {DEFAULT_ALPHA} = 90% coverage)')
    p.add_argument('--cusum-k', type=float, default=CUSUM_K)
    p.add_argument('--cusum-h', type=float, default=CUSUM_H)
    p.add_argument('--skip-conformal', action='store_true')
    p.add_argument('--skip-drift', action='store_true')
    p.add_argument('--check', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    symbols = get_symbols(args.symbols)

    print("\n" + "█" * 68)
    print("  SPRINT 5 — ENSEMBLE, CALIBRATION & DRIFT MONITORING")
    print(f"  Symbols : {', '.join(symbols)}")
    print(f"  Alpha   : {args.alpha}  (nominal coverage: {(1-args.alpha)*100:.0f}%)")
    print(f"  Device  : {DEVICE}")
    print("█" * 68)

    if args.check:
        check_files(symbols)
        return

    summary = {}
    for sym in symbols:
        ens_result = train_ensemble(sym, args.alpha)
        if ens_result is None:
            summary[sym] = {'status': '❌ ensemble failed'}
            continue

        conf_result = None
        if not args.skip_conformal:
            conf_result = conformal_calibrate(sym, ens_result, args.alpha)

        cusum_result = None
        if not args.skip_drift:
            cusum_result = cusum_baseline(sym, ens_result, args.cusum_k, args.cusum_h)

        summary[sym] = {
            'status': '✅',
            'test_rmse': ens_result['test_rmse'],
            'test_corr': ens_result['test_corr'],
            'test_dir_acc': ens_result['test_dir_acc'],
            'conformal_half_width': conf_result['half_width'] if conf_result else None,
            'empirical_coverage': conf_result['empirical_coverage'] if conf_result else None,
            'cusum_h': cusum_result['h'] if cusum_result else None,
        }

    print(f"\n{BANNER}\n  SPRINT 5 SUMMARY\n{BANNER}")
    print(f"  {'Symbol':<12} {'Status':<8} {'RMSE':>7} {'Corr':>7} {'DirAcc':>8} "
          f"{'±width':>8} {'Coverage':>9}")
    print(f"  {'-'*12} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*9}")
    for sym, s in summary.items():
        if s['status'] == '✅':
            print(f"  {sym:<12} {s['status']:<8} {s['test_rmse']:>7.4f} "
                  f"{s['test_corr']:>7.3f} {s['test_dir_acc']:>7.2f}% "
                  f"{s['conformal_half_width']:>8.4f} "
                  f"{s['empirical_coverage']*100:>8.1f}%")
        else:
            print(f"  {sym:<12} {s['status']}")

    with open('sprint5_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✓ Summary saved → sprint5_summary.json")

    n_ok = sum(1 for s in summary.values() if s['status'] == '✅')
    print(f"\n{BANNER}")
    if n_ok == len(symbols):
        print(f"""
  ✅ Sprint 5 complete — {n_ok}/{len(symbols)} symbols.

  What you now have, per symbol:
    {{SYM}}_ensemble_xgb.json      → meta-learner combining 3 fusion variants
    {{SYM}}_ensemble_xgb_cls.json  → directional meta-learner
    {{SYM}}_conformal.json         → calibrated ±interval with coverage guarantee
    {{SYM}}_cusum_state.json       → drift-detection baseline for production

  Next steps
  ──────────
    Sprint 6  →  FastAPI inference endpoint + TensorRT export
                 (loads ensemble + conformal + cusum, serves /predict)
    Verify    →  python sprint5_ensemble.py --check
        """)
    else:
        print(f"  ⚠ {len(symbols) - n_ok}/{len(symbols)} symbols failed — check Sprint 2/4 checkpoints exist")
    print(BANNER)


if __name__ == '__main__':
    main()
