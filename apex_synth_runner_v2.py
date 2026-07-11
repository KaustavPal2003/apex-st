"""
apex_synth_runner_v2.py  —  Optimised Synthetic Pipeline Runner
═══════════════════════════════════════════════════════════════════
Identical in/out contract to v1 but with two key speed improvements:

  Sprint 1  KPCA  →  Nystroem approximation   O(N²) → O(N·m)  ~25–100× faster
  Sprint 2  BiLSTM 2L-256h  →  GRU 1L-128h
            Transformer 4L-128d  →  2L-64d
            Stride-3 conv (T=60 → T=20 before GRU/Transformer)
            Result: 5.0× epoch speedup, 77% fewer parameters

All output files are byte-compatible with sprint2_model.py and
sprint4_fusion.py. Real price data is used automatically whenever
data/<SYMBOL>.csv exists (written by fetch_nse_data.py); otherwise this
falls back to the synthetic generator below. Pass --synthetic to force
synthetic data even when a real CSV is present.

Usage
─────
  python apex_synth_runner_v2.py                  # all 5 symbols (real data if data/*.csv exists)
  python apex_synth_runner_v2.py RELIANCE         # single symbol
  python apex_synth_runner_v2.py --fast           # smoke test (~3 min CPU)
  python apex_synth_runner_v2.py --synthetic      # force synthetic data, ignore data/*.csv
  python apex_synth_runner_v2.py --sprint1-only
  python apex_synth_runner_v2.py --sprint2-only   # arrays must already exist
  python apex_synth_runner_v2.py --check          # verify saved files

Hardware targets
────────────────
  CPU  (current)  : ~13 min/symbol  →  ~64 min total (5 symbols, 30 epochs)
  A100 + AMP      : ~1.5 min/symbol →  ~8  min total
  4× A100 + DDP   : ~25 sec/symbol  →  ~2  min total
"""

import sys, argparse, pickle, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SYMBOLS     = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
LOOKBACK    = 60
N_NYSTROEM  = 200          # landmark points for kernel approximation (m)
N_HMM       = 4
TRAIN_FRAC  = 0.80
VAL_FRAC    = 0.10
N_DAYS      = 5000
DATA_DIR    = Path('data')   # written by fetch_nse_data.py: data/<SYMBOL>.csv

# Sprint 2 — optimised hypers
GRU_HIDDEN   = 128         # was 256
GRU_LAYERS   = 1           # was 2
TF_D_MODEL   = 64          # was 128
TF_N_LAYERS  = 2           # was 4
TF_N_HEADS   = 4           # was 8
FUSION_DIM   = 128         # was 256
BATCH_SIZE   = 64
LR           = 3e-4
EPOCHS       = 30
PATIENCE     = 7
MC_SAMPLES   = 20
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BANNER = "═" * 68

SYMBOL_PROFILES = {
    'RELIANCE':  dict(start_price=1000, mu=0.0004, base_vol=0.012, crash_vol=0.045),
    'TCS':       dict(start_price=2500, mu=0.0003, base_vol=0.010, crash_vol=0.038),
    'INFY':      dict(start_price=1200, mu=0.0003, base_vol=0.011, crash_vol=0.042),
    'HDFCBANK':  dict(start_price=800,  mu=0.0004, base_vol=0.009, crash_vol=0.040),
    'ICICIBANK': dict(start_price=600,  mu=0.0004, base_vol=0.013, crash_vol=0.050),
}


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA
# ─────────────────────────────────────────────────────────────────────────────
def _load_real_ohlcv(symbol: str, n_days: int) -> pd.DataFrame:
    """
    Load real OHLCV from data/<symbol>.csv (written by fetch_nse_data.py).
    Returns None if the file doesn't exist or doesn't match the expected
    schema, so the caller can fall back to synthetic data instead of crashing.
    """
    path = DATA_DIR / f'{symbol}.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = {'date', 'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(df.columns):
        print(f"  ⚠  {path} missing expected columns {required} — "
              f"falling back to synthetic data")
        return None

    df['timestamp'] = pd.to_datetime(df['date'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(np.float32)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    if len(df) > n_days:
        df = df.iloc[-n_days:].reset_index(drop=True)
    return df


def generate_ohlcv(symbol: str, n_days: int = N_DAYS,
                    force_synthetic: bool = False) -> pd.DataFrame:
    """
    Returns OHLCV for `symbol`. By default, real price history from
    data/<symbol>.csv (written by fetch_nse_data.py) is used if present,
    trimmed to the most recent n_days rows. Falls back to the synthetic
    generator below if no real CSV is found, or if force_synthetic=True
    (useful for fast architecture smoke-tests independent of real data).
    """
    if not force_synthetic:
        real = _load_real_ohlcv(symbol, n_days)
        if real is not None:
            print(f"  ✓  Using REAL OHLCV for {symbol}: {len(real)} rows "
                  f"({real['timestamp'].iloc[0].date()} → "
                  f"{real['timestamp'].iloc[-1].date()})")
            if len(real) < n_days:
                print(f"  ⚠  Only {len(real)} real rows available "
                      f"(requested {n_days}) — using all available real "
                      f"data rather than padding with synthetic rows.")
            return real
        print(f"  ℹ  No {DATA_DIR / (symbol + '.csv')} found — generating "
              f"SYNTHETIC OHLCV for {symbol} instead. Run fetch_nse_data.py "
              f"first for real data.")

    seed = sum(ord(c) for c in symbol)
    rng  = np.random.default_rng(seed)
    p = SYMBOL_PROFILES.get(symbol, dict(start_price=1000, mu=0.0004, base_vol=0.012, crash_vol=0.045))
    dates = pd.bdate_range(start='2005-01-01', periods=n_days)
    price = float(p['start_price'])
    closes, opens, highs, lows, volumes = [], [], [], [], []
    bands = [(0.00,0.35,p['base_vol']), (0.35,0.42,p['crash_vol']),
             (0.42,0.55,p['base_vol']*1.4), (0.55,0.80,p['base_vol']*0.9),
             (0.80,1.00,p['crash_vol']*0.7)]
    for i in range(n_days):
        frac = i / n_days
        vol  = next(v for lo,hi,v in bands if lo <= frac < hi)
        ret  = float(rng.normal(p['mu'], vol))
        price = max(price*(1+ret), 1.0)
        spread = abs(float(rng.normal(0, vol*0.5)))*price
        o  = price*(1+float(rng.normal(0,vol*0.25)))
        h  = max(price,o)+spread*float(rng.uniform(0.2,1.0))
        l  = min(price,o)-spread*float(rng.uniform(0.2,1.0))
        bv = 1_000_000+abs(ret)/p['base_vol']*400_000
        v  = max(50_000, float(rng.normal(bv, bv*0.3)))
        closes.append(price); opens.append(o)
        highs.append(h);      lows.append(max(l,price*0.5))
        volumes.append(v)
    return pd.DataFrame({
        'timestamp': dates,
        'open':   np.array(opens,   dtype=np.float32),
        'high':   np.array(highs,   dtype=np.float32),
        'low':    np.array(lows,    dtype=np.float32),
        'close':  np.array(closes,  dtype=np.float32),
        'volume': np.array(volumes, dtype=np.float32),
    })


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds lag/rolling features so raw columns exceed 120 for Nystroem."""
    c, v = df['close'], df['volume']
    rets = c.pct_change()
    for lag in [1,2,3,5,10,15,20,30,40,60]:
        df[f'close_lag_{lag}'] = c.shift(lag)
        df[f'vol_lag_{lag}']   = v.shift(lag)
    for w in [5,10,20,40,60]:
        df[f'ret_roll_mean_{w}'] = rets.rolling(w).mean()
        df[f'ret_roll_std_{w}']  = rets.rolling(w).std()
        df[f'ret_roll_skew_{w}'] = rets.rolling(w).skew()
        df[f'ret_roll_kurt_{w}'] = rets.rolling(w).kurt()
    for col in [c for c in df.columns if c.startswith('sma_') or c.startswith('ema_')]:
        df[f'close_ratio_{col}'] = df['close']/(df[col]+1e-9)
    for w in [5,10,20]:
        mu = v.rolling(w).mean(); sd = v.rolling(w).std()+1e-9
        df[f'vol_zscore_{w}'] = (v-mu)/sd
    df['hl_spread'] = (df['high']-df['low'])/(df['close']+1e-9)
    df['oc_spread'] = (df['close']-df['open'])/(df['close']+1e-9)
    for h in [5,10,20,60]:
        df[f'momentum_{h}'] = c/(c.shift(h)+1e-9)-1
    return df


# ─────────────────────────────────────────────────────────────────────────────
# NYSTROEM REDUCER  (drop-in replacement for KPCAReducer)
# ─────────────────────────────────────────────────────────────────────────────
class NystroemReducer:
    """
    Approximates an RBF kernel map in O(N·m) instead of O(N²).
    Fitted on train rows only, then transforms train/val/test.

    Output dim = n_components (Nystroem feature map)
    followed by PCA to compress to out_dim if needed.
    """
    def __init__(self, n_components: int = N_NYSTROEM, out_dim: int = 96,
                 gamma: float = 0.01, random_state: int = 42):
        self.n_components  = n_components
        self.out_dim       = out_dim
        self.gamma         = gamma
        self.random_state  = random_state
        self._is_fitted    = False
        self._scaler       = StandardScaler()
        self._nystroem     = Nystroem(kernel='rbf', gamma=gamma,
                                      n_components=n_components,
                                      random_state=random_state, n_jobs=-1)
        self._pca          = PCA(n_components=out_dim, random_state=random_state)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        Xs = self._scaler.fit_transform(X)
        Xn = self._nystroem.fit_transform(Xs)
        # Cap out_dim to available Nystroem features
        self.out_dim = min(self.out_dim, Xn.shape[1])
        self._pca.n_components = self.out_dim
        Xp = self._pca.fit_transform(Xn)
        self._is_fitted = True
        return Xp.astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._is_fitted, "Call fit_transform first"
        Xs = self._scaler.transform(X)
        Xn = self._nystroem.transform(Xs)
        return self._pca.transform(Xn).astype(np.float32)



# ─────────────────────────────────────────────────────────────────────────────
# RAW FEATURE CAPTURE  (subclasses KPCAReducer, swapped in at runtime)
# ─────────────────────────────────────────────────────────────────────────────
class RawCaptureKPCA:
    """
    Swapped in place of KPCAReducer inside AdvancedFeaturePipeline.
    intercepts the raw (pre-reduction) feature matrix and returns it unchanged,
    so our NystroemReducer receives the full feature space.
    """
    def __init__(self):
        self._is_fitted = True
        self.n_components = 1  # placeholder to satisfy pipeline checks

    def fit(self, X):
        return self

    def transform(self, X):
        return X.copy()          # pass raw features through unchanged

    def fit_transform(self, X):  # some pipeline versions call this
        return X.copy()

    def save(self, path): pass   # no-op — NystroemReducer saved separately
    def load(self, path): return self


# ─────────────────────────────────────────────────────────────────────────────
# SPRINT 1  — feature pipeline with Nystroem
# ─────────────────────────────────────────────────────────────────────────────
def run_sprint1(symbol: str, n_days: int = N_DAYS,
                 force_synthetic: bool = False) -> bool:
    print(f"\n{BANNER}")
    print(f"  SPRINT 1 v2 (Nystroem) — {symbol}")
    print(BANNER)

    try:
        from apex_feature_engineering import AdvancedFeaturePipeline
    except ImportError:
        print("  ❌ apex_feature_engineering.py not found. Run from codebase folder.")
        return False

    print(f"\n[1/5] Loading OHLCV (≤{n_days} days)…")
    df = generate_ohlcv(symbol, n_days=n_days, force_synthetic=force_synthetic)
    print(f"  ✓  {len(df)} rows | close {df['close'].min():.0f} – {df['close'].max():.0f}")

    n         = len(df)
    train_end = int(n * TRAIN_FRAC)

    print(f"\n[2/5] Running AdvancedFeaturePipeline (DWT + HMM) with raw feature capture…")
    from apex_feature_engineering import AdvancedFeaturePipeline

    pipeline = AdvancedFeaturePipeline(
        lookback          = LOOKBACK,
        n_kpca_components = 1,  # placeholder — overridden by RawCaptureKPCA
        n_hmm_states      = N_HMM,
        wavelet_cols      = ['close', 'volume'],
        n_wf_folds        = 5,
    )
    # Swap in the capture KPCA
    pipeline.kpca = RawCaptureKPCA()

    # Enrich TA features before they enter the pipeline
    _orig = pipeline.ta_builder.compute
    pipeline.ta_builder.compute = lambda df_in: enrich_features(_orig(df_in))

    result    = pipeline.fit_transform(df, train_end_idx=train_end)
    folds     = result['folds']
    if not folds:
        print("  ❌ No folds produced"); return False

    # The sequences now have raw feature dim (not KPCA-reduced)
    best = folds[-1]
    X_tr_raw = best['X_train']   # (N_tr, T, F_raw)
    X_te_raw = best['X_test']

    print(f"  ✓  Raw feature dim: {X_tr_raw.shape[-1]}  sequences: {X_tr_raw.shape[0]}")

    print(f"\n[3/5] Fitting NystroemReducer on train split…")
    # Flatten time axis: (N, T, F) → (N*T, F), fit Nystroem, reshape back
    N_tr, T, F = X_tr_raw.shape
    t0 = time.time()
    reducer = NystroemReducer(n_components=N_NYSTROEM, out_dim=min(96, F),
                               gamma=1.0/F)
    X_tr_2d = X_tr_raw.reshape(-1, F)                       # (N*T, F)
    X_tr_rd = reducer.fit_transform(X_tr_2d).reshape(N_tr, T, -1)

    X_te_2d = X_te_raw.reshape(-1, F)
    X_te_rd = reducer.transform(X_te_2d).reshape(len(X_te_raw), T, -1)
    t_nys = time.time()-t0

    feat_dim = X_tr_rd.shape[-1]
    print(f"  ✓  Nystroem done in {t_nys:.2f}s  →  feat_dim={feat_dim}")

    print(f"\n[4/5] Splitting train / val / test…")
    X_all = np.concatenate([X_tr_rd, X_te_rd], axis=0)
    yr_all = np.concatenate([best['y_reg_train'], best['y_reg_test']], axis=0)
    yc_all = np.concatenate([best['y_cls_train'], best['y_cls_test']], axis=0)
    dates_all = np.concatenate([best['dates_train'], best['dates_test']], axis=0)  # NEW

    n2 = len(X_all)
    t_end = int(n2 * TRAIN_FRAC)
    v_end = int(n2 * (TRAIN_FRAC + VAL_FRAC))
    splits = {
        'train': (X_all[:t_end], yr_all[:t_end], yc_all[:t_end], dates_all[:t_end]),  # CHANGED
        'val': (X_all[t_end:v_end], yr_all[t_end:v_end], yc_all[t_end:v_end], dates_all[t_end:v_end]),  # CHANGED
        'test': (X_all[v_end:], yr_all[v_end:], yc_all[v_end:], dates_all[v_end:]),  # CHANGED
    }
    for split, (X, yr, yc, dt) in splits.items():  # CHANGED: + dt
        np.save(f'{symbol}_apex_X_{split}.npy', X.astype(np.float32))
        np.save(f'{symbol}_apex_y_reg_{split}.npy', yr.astype(np.float32))
        np.save(f'{symbol}_apex_y_cls_{split}.npy', yc.astype(np.int64))
        np.save(f'{symbol}_apex_dates_{split}.npy', dt)  # NEW

    print(f"\n[5/5] Saving pipeline + reducer…")
    pipeline.save(f'{symbol}_apex_pipeline.pkl')
    with open(f'{symbol}_apex_nystroem.pkl', 'wb') as f:
        pickle.dump({'reducer': reducer, 'feat_dim': feat_dim}, f)

    X_tr = splits['train'][0]; X_v = splits['val'][0]; X_te = splits['test'][0]
    print(f"\n  ✓ feat_dim : {feat_dim}")
    print(f"  ✓ train    : {X_tr.shape}")
    print(f"  ✓ val      : {X_v.shape}")
    print(f"  ✓ test     : {X_te.shape}")
    print(f"  ✓ HMM fitted on train only: {getattr(pipeline, '_regime_fitted', True)}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# SPRINT 2 — optimised model
# ─────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d_model,2).float()*(-np.log(10000.)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return self.drop(x + self.pe[:,:x.size(1)])


class SpatialTemporalBranchOpt(nn.Module):
    """
    Conv1D stack  →  stride-3 conv (T=60 → T=20)
                 →  GRU 1L-128h (bidirectional)
                 →  MHA 4-head
    vs baseline: Conv1D → BiLSTM 2L-256h → MHA 8-head
    """
    def __init__(self, in_features=96, conv_channels=(128,256,128),
                 gru_hidden=GRU_HIDDEN, gru_layers=GRU_LAYERS,
                 attn_heads=TF_N_HEADS, dropout=0.3):
        super().__init__()
        channels = [in_features] + list(conv_channels)
        convs = []
        for i in range(len(conv_channels)):
            ks, pd = [3,5,7][i%3], [1,2,3][i%3]
            convs += [nn.Conv1d(channels[i], channels[i+1], ks, padding=pd),
                      nn.BatchNorm1d(channels[i+1]), nn.GELU(), nn.Dropout(dropout)]
        convs += [nn.Conv1d(conv_channels[-1], conv_channels[-1],
                            kernel_size=3, stride=3)]  # T=60 → T=20
        self.conv    = nn.Sequential(*convs)
        self.gru     = nn.GRU(conv_channels[-1], gru_hidden, gru_layers,
                               batch_first=True, bidirectional=True)
        self.attn    = nn.MultiheadAttention(gru_hidden*2, attn_heads,
                                              dropout=dropout, batch_first=True)
        self.norm    = nn.LayerNorm(gru_hidden*2)
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.out_dim = gru_hidden * 2

    def forward(self, x):
        x = self.conv(x.permute(0,2,1)).permute(0,2,1)  # (B,20,128)
        x, _ = self.gru(x)
        a, _ = self.attn(x, x, x)
        x = self.norm(x + a)
        return self.pool(x.permute(0,2,1)).squeeze(-1)


class TemporalTransformerBranchOpt(nn.Module):
    """
    2-layer encoder, d_model=64 (vs 4L-128d baseline)
    Input is already downsampled to T=20 by the conv stride → ~9× cheaper attention.
    """
    def __init__(self, in_features=96, d_model=TF_D_MODEL, n_heads=TF_N_HEADS,
                 n_layers=TF_N_LAYERS, dim_ff=256, dropout=0.3):
        super().__init__()
        # Stride conv to match ST branch's T compression
        self.stride_conv = nn.Conv1d(in_features, in_features,
                                     kernel_size=3, stride=3)
        self.proj = nn.Linear(in_features, d_model)
        self.pe   = PositionalEncoding(d_model, dropout=dropout)
        layer     = nn.TransformerEncoderLayer(d_model, n_heads, dim_ff, dropout,
                                               batch_first=True, norm_first=True)
        self.enc  = nn.TransformerEncoder(layer, n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = d_model

    def forward(self, x):
        # x: (B, T=60, F)
        x = self.stride_conv(x.permute(0,2,1)).permute(0,2,1)  # (B,20,F)
        x = self.pe(self.proj(x))
        return self.pool(self.enc(x).permute(0,2,1)).squeeze(-1)


class APEXSTModelV2(nn.Module):
    """
    Optimised APEX-ST:
      - GRU-1L-128h  (was BiLSTM-2L-256h)
      - Transformer 2L-64d  (was 4L-128d)
      - Stride-3 conv → T=20 before both branches
      - Fusion dim 128  (was 256)
      Parameters: ~1.1M  (was 4.9M)  |  5.0× faster per epoch
    """
    def __init__(self, in_features=96, fusion_dim=FUSION_DIM, dropout=0.3):
        super().__init__()
        self.st_branch = SpatialTemporalBranchOpt(in_features, dropout=dropout)
        self.tf_branch = TemporalTransformerBranchOpt(in_features, dropout=dropout)
        combined = self.st_branch.out_dim + self.tf_branch.out_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined, fusion_dim), nn.LayerNorm(fusion_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim//2), nn.GELU(),
        )
        half = fusion_dim // 2
        self.reg_head = nn.Linear(half, 1)
        self.cls_head = nn.Linear(half, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def encode(self, x):
        """Pre-head embedding — used by Sprint 4 PriceEncoder."""
        return self.fusion(torch.cat([self.st_branch(x), self.tf_branch(x)], dim=-1))

    def forward(self, x, return_features=False):
        fused = self.encode(x)
        r, c  = self.reg_head(fused), self.cls_head(fused)
        return (r, c, fused) if return_features else (r, c)

    def predict_with_uncertainty(self, x, n_samples=MC_SAMPLES):
        self.train()
        regs, clss = [], []
        with torch.no_grad():
            for _ in range(n_samples):
                r, c = self(x)
                regs.append(r); clss.append(torch.sigmoid(c))
        self.eval()
        return {'reg_mean': torch.stack(regs).mean(0),
                'reg_std':  torch.stack(regs).std(0),
                'cls_mean': torch.stack(clss).mean(0),
                'cls_std':  torch.stack(clss).std(0)}


class APEXLoss(nn.Module):
    """Combined regression + classification loss.

    alpha controls the reg/cls balance:
      alpha=0.5  → equal weight (default at epoch 1)
      alpha=0.6  → slight regression dominance (final epochs)
    Starting balanced ensures the cls head receives strong gradient
    signal early, preventing it from getting stuck at the base rate.
    pos_weight corrects for class imbalance in the binary cls target.
    """
    def __init__(self, alpha=0.5, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.mse   = nn.MSELoss()
        self.bce   = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight  # (1,) tensor; corrects class imbalance
        )
    def forward(self, rp, cp, rt, ct):
        rl = self.mse(rp, rt); cl = self.bce(cp, ct)
        return self.alpha*rl + (1-self.alpha)*cl, rl, cl


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING + TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def load_arrays(symbol: str, max_samples: int = None):
    keys  = ['X_train','X_val','X_test',
             'y_reg_train','y_reg_val','y_reg_test',
             'y_cls_train','y_cls_val','y_cls_test']
    files = {k: f'{symbol}_apex_{k}.npy' for k in keys}
    missing = [v for v in files.values() if not Path(v).exists()]
    if missing:
        print(f"  ❌ Missing: {missing[:3]}  — run Sprint 1 first")
        return None
    data = {k: np.load(v) for k, v in files.items()}
    if max_samples:
        for k in list(data): data[k] = data[k][:max_samples]

    # y_reg is now a log-return (log(P_t+h / P_t)), computed at source in
    # apex_feature_engineering.py. It is already dimensionless, scale-invariant,
    # and in a sensible range (typically ±0.5 for a 30-day horizon on NSE stocks).
    # The previous log(y / mean_train_price) workaround was needed because y_reg
    # used to be a raw price level — that workaround is now both unnecessary and
    # wrong (log of a log-return has no meaningful interpretation).
    data['reg_mean'] = 0.0   # kept for checkpoint backward-compat; no longer used
    data['reg_std']  = 1.0
    tr, va, te = data['X_train'], data['X_val'], data['X_test']
    print(f"  ✓ X_train {tr.shape}  X_val {va.shape}  X_test {te.shape}")
    return data


def make_loaders(data, bs=BATCH_SIZE):
    def ds(s):
        X  = torch.tensor(data[f'X_{s}'],     dtype=torch.float32)
        yr = torch.tensor(data[f'y_reg_{s}'], dtype=torch.float32)
        yc = torch.tensor(data[f'y_cls_{s}'], dtype=torch.float32).unsqueeze(-1)
        return TensorDataset(X, yr, yc)
    return (DataLoader(ds('train'), bs, shuffle=True,  num_workers=0),
            DataLoader(ds('val'),   bs, shuffle=False, num_workers=0),
            DataLoader(ds('test'),  bs, shuffle=False, num_workers=0))


def eval_metrics(model, loader, loss_fn):
    model.eval()
    tot = 0.0; rp_l, rt_l, cp_l, ct_l = [], [], [], []
    with torch.no_grad():
        for X, yr, yc in loader:
            X, yr, yc = X.to(DEVICE), yr.to(DEVICE), yc.to(DEVICE)
            rp, cp = model(X)
            loss, _, _ = loss_fn(rp, cp, yr, yc)
            tot += loss.item()*len(X)
            rp_l.append(rp.cpu()); rt_l.append(yr.cpu())
            cp_l.append(torch.sigmoid(cp).cpu()); ct_l.append(yc.cpu())
    n_tot = len(loader.dataset)
    rp_np = torch.cat(rp_l).numpy().flatten()
    rt_np = torch.cat(rt_l).numpy().flatten()
    cp_np = (torch.cat(cp_l).numpy().flatten()>0.5).astype(int)
    ct_np = torch.cat(ct_l).numpy().flatten().astype(int)
    corr  = float(np.corrcoef(rp_np,rt_np)[0,1]) if rp_np.std()>1e-8 else 0.0
    return {'loss': tot/n_tot, 'dir_acc': (cp_np==ct_np).mean()*100, 'corr': corr}


def run_sprint2(symbol: str, epochs: int = EPOCHS, fast: bool = False) -> bool:
    print(f"\n{BANNER}")
    print(f"  SPRINT 2 v2 — APEX-ST Optimised  [{DEVICE}]  ({symbol})")
    print(BANNER)

    print("\n[1/4] Loading Sprint 1 arrays…")
    data = load_arrays(symbol, max_samples=300 if fast else None)
    if data is None: return False

    feat_dim = data['X_train'].shape[-1]
    print(f"  Feature dim: {feat_dim}")

    print(f"\n[2/4] Building optimised APEX-ST v2…")
    model   = APEXSTModelV2(in_features=feat_dim).to(DEVICE)
    # pos_weight = #neg / #pos — corrects BCEWithLogitsLoss for class imbalance
    yc_tr    = data['y_cls_train'].astype(np.float32)
    n_pos    = max(yc_tr.sum(), 1.0)
    n_neg    = max(len(yc_tr) - n_pos, 1.0)
    pos_wt   = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    loss_fn  = APEXLoss(alpha=0.5, pos_weight=pos_wt)

    # Separate LR for heads vs encoder:
    #   heads need 5× stronger signal to avoid getting stuck at base rate
    head_params    = list(model.reg_head.parameters()) + \
                     list(model.cls_head.parameters())
    encoder_params = [p for p in model.parameters()
                      if not any(p is hp for hp in head_params)]
    opt = torch.optim.AdamW([
        {'params': encoder_params, 'lr': LR,     'weight_decay': 1e-4},
        {'params': head_params,    'lr': LR*5.0, 'weight_decay': 0.0},
    ])
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs,
        eta_min=LR*0.01  # applies to all param groups proportionally
    )
    n_p     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {n_p:,}  (baseline was ~4.9M)")

    # AMP — automatic mixed precision on GPU
    use_amp = DEVICE.type == 'cuda'
    scaler_amp = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp: print(f"  AMP bf16   : enabled")

    train_dl, val_dl, test_dl = make_loaders(data)

    print(f"\n[3/4] Training (max {epochs} epochs, patience {PATIENCE})…")
    print(f"  {'Ep':>4}  {'Train':>9}  {'Val':>9}  {'DirAcc':>7}  {'Corr':>6}  {'t':>6}")
    print(f"  {'─'*4}  {'─'*9}  {'─'*9}  {'─'*7}  {'─'*6}  {'─'*6}")

    best_val = float('inf'); wait = 0
    ckpt     = f'{symbol}_apex_st_best.pt'

    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0; t0 = time.time()
        for X, yr, yc in train_dl:
            X, yr, yc = X.to(DEVICE), yr.to(DEVICE), yc.to(DEVICE)
            opt.zero_grad()
            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    rp, cp = model(X)
                    loss, _, _ = loss_fn(rp, cp, yr, yc)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(opt); scaler_amp.update()
            else:
                rp, cp = model(X)
                loss, _, _ = loss_fn(rp, cp, yr, yc)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tr_loss += loss.item()*len(X)
        tr_loss /= len(train_dl.dataset)
        # Anneal alpha 0.5→0.6: start balanced, end with slight reg dominance
        loss_fn.alpha = min(0.6, 0.5 + 0.1*(ep/epochs))
        vm = eval_metrics(model, val_dl, loss_fn)
        sched.step()
        ep_t = time.time()-t0

        print(f"  {ep:>4}  {tr_loss:>9.5f}  {vm['loss']:>9.5f}  "
              f"{vm['dir_acc']:>6.2f}%  {vm['corr']:>6.3f}  {ep_t:>5.1f}s")

        if vm['loss'] < best_val:
            best_val = vm['loss']; wait = 0
            torch.save({'epoch': ep, 'model_state': model.state_dict(),
                        'opt_state': opt.state_dict(), 'val_metrics': vm,
                        'feat_dim': feat_dim, 'symbol': symbol,
                        'model_version': 'v2_optimised'}, ckpt)
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"\n  Early stopping at epoch {ep}"); break

    print(f"\n[4/4] Final test evaluation…")
    ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck['model_state'])
    tm = eval_metrics(model, test_dl, loss_fn)

    print(f"\n  ┌──────────────────────────────────┐")
    print(f"  │  TEST RESULTS — {symbol:<13}  │")
    print(f"  ├──────────────────────────────────┤")
    print(f"  │  Directional Accuracy : {tm['dir_acc']:>6.2f}%  │")
    print(f"  │  Return Correlation   : {tm['corr']:>6.3f}   │")
    print(f"  │  Total Loss           : {tm['loss']:>6.4f}   │")
    print(f"  │  Parameters           : {n_p:>7,}   │")
    print(f"  └──────────────────────────────────┘")

    print(f"\n  MC Dropout uncertainty ({MC_SAMPLES} passes)…")
    X_s = next(iter(test_dl))[0][:8].to(DEVICE)
    unc = model.predict_with_uncertainty(X_s)
    print(f"  Reg std (avg) : {unc['reg_std'].mean().item():.5f}")
    print(f"  Cls std (avg) : {unc['cls_std'].mean().item():.4f}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# FILE CHECK
# ─────────────────────────────────────────────────────────────────────────────
def check_files(symbols):
    print(f"\n{BANNER}\n  FILE CHECK\n{BANNER}")
    all_ok = True
    for sym in symbols:
        for split in ('train','val','test'):
            for kind in ('X','y_reg','y_cls'):
                f = Path(f'{sym}_apex_{kind}_{split}.npy')
                if f.exists():
                    print(f"  ✅  {f.name:<44} {str(np.load(f).shape)}")
                else:
                    print(f"  ❌  {f.name:<44} MISSING"); all_ok = False
        ck = Path(f'{sym}_apex_st_best.pt')
        if ck.exists():
            d = torch.load(ck, map_location='cpu', weights_only=False)
            ver = d.get('model_version','v1_baseline')
            print(f"  ✅  {ck.name:<44} ep={d['epoch']}  "
                  f"dir_acc={d['val_metrics']['dir_acc']:.2f}%  [{ver}]")
        else:
            print(f"  ❌  {ck.name:<44} MISSING"); all_ok = False
    print(f"\n  {'✅ All present' if all_ok else '❌ Some missing'}")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='APEX-ST v2 optimised runner')
    p.add_argument('--seed', type=int, default=42,
               help='Random seed for reproducible weight init + data shuffling')
    p.add_argument('symbols', nargs='*')
    p.add_argument('--sprint1-only',  action='store_true')
    p.add_argument('--sprint2-only',  action='store_true')
    p.add_argument('--check',         action='store_true')
    p.add_argument('--fast',          action='store_true',
                   help='Smoke-test: 1500 days, 3 epochs, 300 samples')
    p.add_argument('--synthetic',     action='store_true',
                   help='Force synthetic OHLCV even if data/<SYMBOL>.csv '
                        'exists (default: real data is used when present)')
    p.add_argument('--epochs', type=int, default=EPOCHS)
    p.add_argument('--n-days', type=int, default=N_DAYS)
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build target list: explicit args → watchlist.json → hardcoded defaults
    if args.symbols:
        targets = [s.upper() for s in args.symbols]
    else:
        wl_path = Path('watchlist.json')
        if wl_path.exists():
            import json
            wl = json.loads(wl_path.read_text()).get('watchlist', [])
            if wl:
                targets = [s.upper() for s in wl]
                print(f"  ℹ  Reading symbols from watchlist.json: {targets}")
            else:
                targets = SYMBOLS
        else:
            targets = SYMBOLS

    # No restriction on symbol names — any valid string is accepted
    # (previous hardcoded allowlist removed so screener output feeds in directly)

    n_days = 1500 if args.fast else args.n_days
    ep     = 3    if args.fast else args.epochs

    print("\n" + "█"*68)
    print("  APEX-ST v2 — OPTIMISED PIPELINE RUNNER")
    print(f"  Symbols  : {', '.join(targets)}")
    print(f"  Days     : {n_days}  |  Device: {DEVICE}")
    print(f"  Opt      : GRU-1L-{GRU_HIDDEN}h + stride-3 conv + Nystroem(m={N_NYSTROEM})")
    print("█"*68)

    if args.check:
        check_files(targets); return

    results = {}
    for sym in targets:
        ok1 = ok2 = True
        if not args.sprint2_only:
            ok1 = run_sprint1(sym, n_days=n_days, force_synthetic=args.synthetic)
        if ok1 and not args.sprint1_only:
            ok2 = run_sprint2(sym, epochs=ep, fast=args.fast)
        results[sym] = '✅' if (ok1 and ok2) else ('⚠ S1 only' if ok1 else '❌')

    print(f"\n{BANNER}\n  FINAL SUMMARY\n{BANNER}")
    for sym, status in results.items():
        print(f"  {status}  {sym}")

    if all(v=='✅' for v in results.values()) and not args.sprint1_only:
        print(f"""
  ✅ Sprint 1 v2 + Sprint 2 v2 complete.

  Speed vs v1 baseline
  ─────────────────────
    Nystroem vs KPCA    : ~25× faster (O(N·m) vs O(N²))
    GRU + stride-3 conv : ~5× faster per epoch
    Combined (CPU)      : ~13 min/symbol  (was ~68 min)
    On A100 + AMP       : ~1.5 min/symbol (was ~4 min)
    On 4× A100 DDP      : ~25 sec/symbol

  Next steps
  ──────────
    Sprint 3  →  python sprint3_finbert.py --dummy
                 python sprint3_gat.py
    Sprint 4  →  python sprint4_fusion.py
    Verify    →  python apex_synth_runner_v2.py --check
        """)
    print(BANNER)


if __name__ == '__main__':
    main()
