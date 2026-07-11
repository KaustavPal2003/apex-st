"""
sprint2_model.py — Sprint 2: Price Branches (APEX-ST) — BASELINE REFERENCE
═══════════════════════════════════════════════════════════════════════════
⚠  NOT part of the active pipeline. apex_synth_runner_v2.py contains the
   optimised model (GRU + stride-conv) actually used by Sprint 3/4/5.
   This file is kept as the original full-size architecture for reference
   and writes to a SEPARATE checkpoint name ({SYMBOL}_apex_st_baseline_best.pt)
   so it can never collide with or overwrite the v2 pipeline's checkpoints.

Branch 1 — Spatial-Temporal : Conv1D stack → BiLSTM → Multi-head Attention
Branch 2 — Temporal Transformer : 12-layer encoder with positional encoding
Fusion   — Concat → FC (gated cross-modal attention comes in Sprint 4)
Output   — Dual head: regression (next-day return) + classification (up/down)

Usage
─────
  python sprint2_model.py                    # train all 5 stocks
  python sprint2_model.py RELIANCE           # single stock
  python sprint2_model.py RELIANCE --epochs 50

Requirements
────────────
  pip install torch torchinfo
"""

import sys
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SYMBOLS      = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
SEQ_LEN      = 60       # lookback window
FEAT_DIM     = 120      # KPCA output from Sprint 1
BATCH_SIZE   = 64
LR           = 3e-4
EPOCHS       = 30
PATIENCE     = 7        # early stopping
MC_SAMPLES   = 20       # Monte Carlo dropout samples at inference
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BANNER       = "═" * 68


# ─────────────────────────────────────────────────────────────────────────────
# BRANCH 1 — SPATIAL-TEMPORAL (Conv1D → BiLSTM → Multi-head Attention)
# ─────────────────────────────────────────────────────────────────────────────
class SpatialTemporalBranch(nn.Module):
    """
    Conv1D stack extracts local temporal patterns,
    BiLSTM captures long-range dependencies bidirectionally,
    Multi-head attention re-weights the most predictive timesteps.
    """
    def __init__(self,
                 in_features:   int = FEAT_DIM,
                 conv_channels: list = [128, 256, 128],
                 lstm_hidden:   int = 256,
                 lstm_layers:   int = 2,
                 attn_heads:    int = 8,
                 dropout:       float = 0.3):
        super().__init__()

        # ── Conv1D stack ────────────────────────────────────────────────────
        conv_layers = []
        in_ch = in_features
        for out_ch in conv_channels:
            conv_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)

        # ── BiLSTM ──────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size  = conv_channels[-1],
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            bidirectional = True,
            dropout     = dropout if lstm_layers > 1 else 0,
        )
        lstm_out_dim = lstm_hidden * 2      # bidirectional

        # ── Multi-head Self-Attention ────────────────────────────────────────
        self.attn = nn.MultiheadAttention(
            embed_dim   = lstm_out_dim,
            num_heads   = attn_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.attn_norm = nn.LayerNorm(lstm_out_dim)
        self.dropout   = nn.Dropout(dropout)

        self.out_dim = lstm_out_dim

    def forward(self, x):
        # x: (B, T, F)
        # Conv1D expects (B, F, T)
        c = self.conv(x.permute(0, 2, 1))          # (B, C, T)
        c = c.permute(0, 2, 1)                      # (B, T, C)

        # BiLSTM
        lstm_out, _ = self.lstm(c)                  # (B, T, 2*H)

        # Self-attention with residual
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        attn_out     = self.attn_norm(lstm_out + self.dropout(attn_out))

        # Pool: mean + last timestep concat
        pooled = torch.cat([
            attn_out.mean(dim=1),
            attn_out[:, -1, :],
        ], dim=-1)                                  # (B, 4*H)

        return pooled                               # (B, out_dim*2)


# ─────────────────────────────────────────────────────────────────────────────
# BRANCH 2 — TEMPORAL TRANSFORMER (12-layer encoder)
# ─────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TemporalTransformerBranch(nn.Module):
    """
    12-layer Transformer encoder with:
    - learned input projection
    - sinusoidal positional encoding
    - pre-norm architecture (more stable training)
    - CLS token for sequence-level representation
    """
    def __init__(self,
                 in_features: int = FEAT_DIM,
                 d_model:     int = 256,
                 n_heads:     int = 8,
                 n_layers:    int = 12,
                 ff_mult:     int = 4,
                 dropout:     float = 0.1):
        super().__init__()

        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # Transformer layers (pre-norm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * ff_mult,
            dropout         = dropout,
            activation      = 'gelu',
            batch_first     = True,
            norm_first      = True,    # pre-norm
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = n_layers,
            norm       = nn.LayerNorm(d_model),
        )

        self.out_dim = d_model

    def forward(self, x):
        # x: (B, T, F)
        B = x.size(0)

        # Project input
        x = self.input_proj(x)                        # (B, T, D)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)        # (B, 1, D)
        x   = torch.cat([cls, x], dim=1)              # (B, T+1, D)

        # Positional encoding
        x = self.pos_enc(x)

        # Transformer
        out = self.transformer(x)                      # (B, T+1, D)

        # CLS output as sequence representation
        cls_out  = out[:, 0, :]                        # (B, D)
        mean_out = out[:, 1:, :].mean(dim=1)           # (B, D)

        return torch.cat([cls_out, mean_out], dim=-1)  # (B, 2*D)


# ─────────────────────────────────────────────────────────────────────────────
# APEX-ST TWO-BRANCH MODEL
# ─────────────────────────────────────────────────────────────────────────────
class APEXSTModel(nn.Module):
    """
    APEX-ST: two-branch price model.
    Sprint 4 will add gated cross-modal attention fusion.
    Sprint 3 will add Branch 3 (FinBERT) and Branch 4 (GAT).

    Dual output:
      reg_out  — next-day log return  (regression)
      cls_out  — direction up/down    (binary classification logit)
    """
    def __init__(self,
                 in_features:  int   = FEAT_DIM,
                 st_lstm_hid:  int   = 256,
                 tf_d_model:   int   = 256,
                 tf_n_layers:  int   = 12,
                 fusion_dim:   int   = 512,
                 dropout:      float = 0.3):
        super().__init__()

        # Branch 1
        self.st_branch = SpatialTemporalBranch(
            in_features   = in_features,
            conv_channels = [128, 256, 128],
            lstm_hidden   = st_lstm_hid,
            lstm_layers   = 2,
            attn_heads    = 8,
            dropout       = dropout,
        )
        st_out = self.st_branch.out_dim * 2   # mean + last concat

        # Branch 2
        self.tf_branch = TemporalTransformerBranch(
            in_features = in_features,
            d_model     = tf_d_model,
            n_heads     = 8,
            n_layers    = tf_n_layers,
            dropout     = dropout * 0.33,
        )
        tf_out = self.tf_branch.out_dim * 2   # cls + mean concat

        # Fusion (Sprint 4: gated cross-modal attention)
        fused_dim = st_out + tf_out
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Dual output head
        head_in = fusion_dim // 2
        self.reg_head = nn.Linear(head_in, 1)   # regression: log return
        self.cls_head = nn.Linear(head_in, 1)   # classification: up/down logit

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, return_features=False):
        """
        x: (B, T, F)
        Returns: (reg_out, cls_out) each (B, 1)
        """
        st  = self.st_branch(x)                        # (B, st_out)
        tf  = self.tf_branch(x)                        # (B, tf_out)
        fused = self.fusion(torch.cat([st, tf], dim=-1))  # (B, fusion//2)

        reg_out = self.reg_head(fused)                 # (B, 1)
        cls_out = self.cls_head(fused)                 # (B, 1) — logit

        if return_features:
            return reg_out, cls_out, fused
        return reg_out, cls_out

    def predict_with_uncertainty(self, x, n_samples=MC_SAMPLES):
        """
        MC Dropout inference: run forward pass n_samples times
        with dropout enabled. Returns mean + std of predictions.
        """
        self.train()   # keep dropout active
        regs, clss = [], []
        with torch.no_grad():
            for _ in range(n_samples):
                r, c = self(x)
                regs.append(r)
                clss.append(torch.sigmoid(c))
        self.eval()

        reg_stack = torch.stack(regs, dim=0)           # (S, B, 1)
        cls_stack = torch.stack(clss, dim=0)

        return {
            'reg_mean':  reg_stack.mean(0),
            'reg_std':   reg_stack.std(0),
            'cls_mean':  cls_stack.mean(0),
            'cls_std':   cls_stack.std(0),
        }


# ─────────────────────────────────────────────────────────────────────────────
# LOSS — combined regression + classification
# ─────────────────────────────────────────────────────────────────────────────
class APEXLoss(nn.Module):
    """
    L = alpha * MSE(reg) + (1-alpha) * BCE(cls)
    alpha is annealed from 0.8 → 0.5 over training.
    """
    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.alpha    = alpha
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, reg_pred, cls_pred, reg_target, cls_target):
        reg_loss = self.mse_loss(reg_pred, reg_target)
        cls_loss = self.bce_loss(cls_pred, cls_target)
        return self.alpha * reg_loss + (1 - self.alpha) * cls_loss, reg_loss, cls_loss


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_apex_data(symbol: str):
    """Load Sprint 1 output arrays. Returns None if files missing."""
    prefix = symbol
    required = [
        f'{prefix}_apex_X_train.npy',
        f'{prefix}_apex_X_val.npy',
        f'{prefix}_apex_X_test.npy',
        f'{prefix}_apex_y_reg_train.npy',
        f'{prefix}_apex_y_reg_val.npy',
        f'{prefix}_apex_y_reg_test.npy',
        f'{prefix}_apex_y_cls_train.npy',
        f'{prefix}_apex_y_cls_val.npy',
        f'{prefix}_apex_y_cls_test.npy',
    ]
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        print(f"  ❌ Missing files: {missing}")
        print(f"     Run: python apex_synth_runner_v2.py {symbol} --sprint1-only")
        return None

    data = {f.replace(f'{prefix}_apex_', '').replace('.npy', ''): np.load(f)
            for f in required}

    # Normalise regression targets to have unit std
    reg_std = data['y_reg_train'].std() + 1e-8
    for split in ['train', 'val', 'test']:
        data[f'y_reg_{split}'] = data[f'y_reg_{split}'] / reg_std
    data['reg_std'] = reg_std

    print(f"  ✓ X_train {data['X_train'].shape}  "
          f"X_val {data['X_val'].shape}  "
          f"X_test {data['X_test'].shape}")
    return data


def make_loaders(data: dict, batch_size: int = BATCH_SIZE):
    def to_tensors(split):
        X  = torch.tensor(data[f'X_{split}'],      dtype=torch.float32)
        yr = torch.tensor(data[f'y_reg_{split}'],  dtype=torch.float32)
        yc = torch.tensor(data[f'y_cls_{split}'],  dtype=torch.float32).unsqueeze(-1)
        return TensorDataset(X, yr, yc)

    train_dl = DataLoader(to_tensors('train'), batch_size=batch_size,
                          shuffle=True,  num_workers=0, pin_memory=False)
    val_dl   = DataLoader(to_tensors('val'),   batch_size=batch_size,
                          shuffle=False, num_workers=0, pin_memory=False)
    test_dl  = DataLoader(to_tensors('test'),  batch_size=batch_size,
                          shuffle=False, num_workers=0, pin_memory=False)
    return train_dl, val_dl, test_dl


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(model, loader, loss_fn, device):
    model.eval()
    total_loss = reg_loss_sum = cls_loss_sum = 0
    all_reg_pred, all_reg_true = [], []
    all_cls_pred, all_cls_true = [], []

    with torch.no_grad():
        for X, yr, yc in loader:
            X, yr, yc = X.to(device), yr.to(device), yc.to(device)
            rp, cp    = model(X)
            loss, rl, cl = loss_fn(rp, cp, yr, yc)

            total_loss    += loss.item() * len(X)
            reg_loss_sum  += rl.item()   * len(X)
            cls_loss_sum  += cl.item()   * len(X)

            all_reg_pred.append(rp.cpu())
            all_reg_true.append(yr.cpu())
            all_cls_pred.append(torch.sigmoid(cp).cpu())
            all_cls_true.append(yc.cpu())

    n = len(loader.dataset)
    reg_pred = torch.cat(all_reg_pred).numpy().flatten()
    reg_true = torch.cat(all_reg_true).numpy().flatten()
    cls_pred = (torch.cat(all_cls_pred).numpy().flatten() > 0.5).astype(int)
    cls_true = torch.cat(all_cls_true).numpy().flatten().astype(int)

    # Directional accuracy (most useful metric for trading)
    dir_acc = (cls_pred == cls_true).mean() * 100

    # Correlation between predicted and actual returns
    corr = float(np.corrcoef(reg_pred, reg_true)[0, 1]) if reg_pred.std() > 1e-8 else 0.0

    return {
        'loss':    total_loss / n,
        'reg_loss': reg_loss_sum / n,
        'cls_loss': cls_loss_sum / n,
        'dir_acc': dir_acc,
        'corr':    corr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train_symbol(symbol: str, epochs: int = EPOCHS) -> bool:
    print(f"\n{BANNER}")
    print(f"  SPRINT 2 — Training APEX-ST on {symbol}")
    print(BANNER)

    # Load data
    print("\n[1/4] Loading Sprint 1 datasets...")
    data = load_apex_data(symbol)
    if data is None:
        return False

    # Detect actual feature dim from data (may be < 120 for small datasets)
    feat_dim = data['X_train'].shape[-1]
    print(f"  Feature dim: {feat_dim}")

    # Build model
    print(f"\n[2/4] Building APEX-ST model on {DEVICE}...")
    model    = APEXSTModel(in_features=feat_dim).to(DEVICE)
    loss_fn  = APEXLoss(alpha=0.7)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=LR * 0.01
    )

    # Count params
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # Loaders
    train_dl, val_dl, test_dl = make_loaders(data)

    # Training loop
    print(f"\n[3/4] Training for up to {epochs} epochs (patience={PATIENCE})...")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>9}  "
          f"{'Dir Acc':>8}  {'Corr':>6}  {'LR':>8}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*9}  {'─'*8}  {'─'*6}  {'─'*8}")

    best_val_loss  = float('inf')
    patience_count = 0
    best_ckpt      = f'{symbol}_apex_st_baseline_best.pt'

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for X, yr, yc in train_dl:
            X, yr, yc = X.to(DEVICE), yr.to(DEVICE), yc.to(DEVICE)
            optimiser.zero_grad()
            rp, cp = model(X)
            loss, _, _ = loss_fn(rp, cp, yr, yc)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item() * len(X)

        train_loss /= len(train_dl.dataset)

        # Anneal alpha: 0.7 → 0.5 over training
        loss_fn.alpha = max(0.5, 0.7 - 0.2 * (epoch / epochs))

        # ── validate ──
        val_m = compute_metrics(model, val_dl, loss_fn, DEVICE)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"  {epoch:>5}  {train_loss:>10.5f}  {val_m['loss']:>9.5f}  "
              f"{val_m['dir_acc']:>7.2f}%  {val_m['corr']:>6.3f}  {current_lr:.2e}")

        # ── early stopping ──
        if val_m['loss'] < best_val_loss:
            best_val_loss  = val_m['loss']
            patience_count = 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'opt_state':   optimiser.state_dict(),
                'val_metrics': val_m,
                'feat_dim':    feat_dim,
                'symbol':      symbol,
                'model_version': 'v1_baseline',
            }, best_ckpt)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    # ── final evaluation ──
    print(f"\n[4/4] Final evaluation on test set...")
    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])

    test_m = compute_metrics(model, test_dl, loss_fn, DEVICE)
    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │  TEST RESULTS — {symbol:<12}    │")
    print(f"  ├─────────────────────────────────┤")
    print(f"  │  Directional Accuracy : {test_m['dir_acc']:6.2f}%  │")
    print(f"  │  Return Correlation   : {test_m['corr']:6.3f}   │")
    print(f"  │  Total Loss           : {test_m['loss']:6.4f}   │")
    print(f"  │  Best epoch checkpoint: {Path(best_ckpt).name:<9} │")
    print(f"  └─────────────────────────────────┘")

    # MC Dropout uncertainty on first test batch
    print(f"\n  MC Dropout uncertainty ({MC_SAMPLES} samples)...")
    X_sample = next(iter(test_dl))[0][:8].to(DEVICE)
    unc = model.predict_with_uncertainty(X_sample)
    print(f"  Reg std (avg): {unc['reg_std'].mean().item():.5f}")
    print(f"  Cls std (avg): {unc['cls_std'].mean().item():.4f}")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# MODEL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def print_model_summary(feat_dim: int = FEAT_DIM):
    model = APEXSTModel(in_features=feat_dim)
    total   = sum(p.numel() for p in model.parameters())
    branch1 = sum(p.numel() for p in model.st_branch.parameters())
    branch2 = sum(p.numel() for p in model.tf_branch.parameters())
    fusion  = sum(p.numel() for p in model.fusion.parameters())
    heads   = (sum(p.numel() for p in model.reg_head.parameters()) +
               sum(p.numel() for p in model.cls_head.parameters()))

    print(f"\n  APEX-ST Model Architecture")
    print(f"  {'─'*45}")
    print(f"  Input shape       : (batch, {SEQ_LEN}, {feat_dim})")
    print(f"  Branch 1 (ST)     : {branch1:>10,} params  Conv1D→BiLSTM→MHA")
    print(f"  Branch 2 (TF)     : {branch2:>10,} params  12-layer Transformer")
    print(f"  Fusion            : {fusion:>10,} params  FC concat (→ gated in S4)")
    print(f"  Output heads      : {heads:>10,} params  reg + cls")
    print(f"  {'─'*45}")
    print(f"  Total             : {total:>10,} params")
    print(f"  Device            : {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('symbols', nargs='*', default=SYMBOLS,
                   help='Symbols to train (default: all 5)')
    p.add_argument('--epochs', type=int, default=EPOCHS)
    p.add_argument('--lr',     type=float, default=LR)
    p.add_argument('--batch',  type=int, default=BATCH_SIZE)
    p.add_argument('--summary', action='store_true',
                   help='Print model summary and exit')
    return p.parse_args()


def main():
    args    = parse_args()
    # Strip shell comment tokens: anything starting with '#' and everything after it.
    # This handles cases like: python sprint2_model.py RELIANCE # some comment
    raw = args.symbols
    filtered = []
    for tok in raw:
        if tok.startswith('#'):
            break          # '#' and all subsequent tokens are a shell comment
        filtered.append(tok)
    symbols = [s.upper() for s in filtered] if filtered else SYMBOLS

    print("\n" + "█" * 68)
    print("  SPRINT 2 — APEX-ST TWO-BRANCH MODEL")
    print(f"  Branch 1 : Conv1D → BiLSTM → Multi-head Attention")
    print(f"  Branch 2 : 12-layer Temporal Transformer")
    print(f"  Output   : regression (return) + classification (direction)")
    print(f"  Device   : {DEVICE}")
    print("█" * 68)

    print_model_summary()

    if args.summary:
        return

    # Check PyTorch is available
    print(f"\n  PyTorch {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  ⚠  No GPU detected — training on CPU (slower but works fine)")

    results = {}
    for sym in symbols:
        ok = train_symbol(sym, epochs=args.epochs)
        results[sym] = '✅' if ok else '❌'

    # Summary table
    print(f"\n{BANNER}")
    print("  SPRINT 2 SUMMARY")
    print(BANNER)
    for sym, status in results.items():
        ckpt = Path(f'{sym}_apex_st_baseline_best.pt')
        note = ""
        if ckpt.exists():
            c    = torch.load(ckpt, map_location='cpu', weights_only=False)
            m    = c.get('val_metrics', {})
            note = (f"dir_acc={m.get('dir_acc', 0):.1f}%  "
                    f"corr={m.get('corr', 0):.3f}  "
                    f"epoch={c.get('epoch', '?')}")
        print(f"  {status}  {sym:<12}  {note}")

    passed = sum(1 for v in results.values() if v == '✅')
    print(f"\n  {passed}/{len(symbols)} stocks trained")
    if passed:
        print(f"""
  ✅ Sprint 2 complete!

  Checkpoints saved: {{SYMBOL}}_apex_st_baseline_best.pt
  For inference:
    ckpt  = torch.load('RELIANCE_apex_st_baseline_best.pt')
    model = APEXSTModel(in_features=ckpt['feat_dim'])
    model.load_state_dict(ckpt['model_state'])
    pred  = model(X_tensor)   # (reg_out, cls_out)

  Next → Sprint 3: FinBERT sentiment + GAT correlation graph
        """)
    print(BANNER)


if __name__ == '__main__':
    main()
