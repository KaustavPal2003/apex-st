"""
sprint4_fusion.py — Sprint 4: Gated Cross-Modal Fusion (APEX-ST Full Model)
═══════════════════════════════════════════════════════════════════════════════
Combines three modality branches with three fusion strategies and runs
a full ablation comparison across all 5 stocks.

Branches (all frozen)
──────────────────────
  Branch 1+2 : APEX-ST price model  (Sprint 2 checkpoint)  → 256-d
  Branch 3   : FinBERT sentiment    (Sprint 3 .npy)         →  64-d
  Branch 4   : GAT graph embedding  (Sprint 3 .npy)         →  64-d
  Total input to fusion             : 384-d per timestep

Fusion variants (trained, frozen branches)
──────────────────────────────────────────
  FusionA : Gated Attention      — per-modality softmax gates
  FusionB : Concat MLP           — baseline flatten → FC layers
  FusionC : Cross-Modal Transformer — multi-head attention across modalities

Outputs (all variants)
───────────────────────
  reg_out     : predicted next-day return  (regression)
  cls_out     : direction logit            (binary classification)
  confidence  : trade confidence 0–100     (calibrated from cls prob + reg mag)

Usage
─────
  python sprint4_fusion.py                    # train & compare all 5 stocks
  python sprint4_fusion.py RELIANCE           # single stock
  python sprint4_fusion.py --summary          # architecture overview
  python sprint4_fusion.py --ablation-only    # print last saved ablation table

Requirements
────────────
  pip install -r requirements.txt   (torch, numpy, scikit-learn, xgboost, …)
  Sprint 2 checkpoint  : {SYMBOL}_apex_st_best.pt
  Sprint 2 price seqs  : {SYMBOL}_apex_X_{train|val|test}.npy
  Sprint 3 sentiment   : {SYMBOL}_apex_sent_{train|val|test}.npy
  Sprint 3 GAT         : {SYMBOL}_apex_gat_{train|val|test}.npy
  Sprint 1/2 labels    : {SYMBOL}_apex_y_reg_{split}.npy
                         {SYMBOL}_apex_y_cls_{split}.npy
"""

import sys
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
# Read symbols from watchlist.json if present, else fall back to defaults
import json as _json
_DEFAULT_SYMBOLS = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
try:
    _wl = _json.loads(open('watchlist.json').read()).get('watchlist', [])
    SYMBOLS = _wl if _wl else _DEFAULT_SYMBOLS
except FileNotFoundError:
    SYMBOLS = _DEFAULT_SYMBOLS
PRICE_DIM    = None   # NOT used as a constructor default (fusion classes default to 64).
                      # Resolved at runtime via model.encode() probe in load_price_encoder().
                      # V2 (active):  fusion_dim//2 = 64
                      # V1 baseline:  fusion_dim//2 = 256
SENT_DIM     = 64     # FinBERT projector output
GAT_DIM      = 64     # GAT output
# FUSION_IN is computed at runtime once PRICE_DIM is known (see process_symbol)
FUSION_DIM   = 128    # hidden dim inside fusion layers
N_HEADS      = 4      # cross-modal transformer heads
N_CM_LAYERS  = 2      # cross-modal transformer layers
EPOCHS       = 40
PATIENCE     = 8
LR           = 3e-4
BATCH        = 32
DROPOUT      = 0.3
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BANNER       = "═" * 68
VARIANTS     = ['FusionA_Gated', 'FusionB_MLP', 'FusionC_Transformer']


# ─────────────────────────────────────────────────────────────────────────────
# APEX-ST PRICE ENCODER  (loads Sprint 2 checkpoint, extracts penultimate layer)
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_encode(model) -> None:
    """
    Both APEXSTModelV2 (apex_synth_runner_v2) and APEXSTModel (sprint2_model)
    expose an encode() method that returns the pre-head embedding via their
    native .st_branch / .tf_branch / .fusion attributes.

    V2 already defines encode() natively.  V1 (APEXSTModel) does not — this
    shim adds a compatible one so the rest of this file can call model.encode()
    uniformly without touching branch1/branch2 (which have never existed on
    either class).
    """
    if hasattr(model, 'encode') and callable(model.encode):
        return  # V2: already present

    # V1 baseline: st_branch + tf_branch → fusion → pre-head embedding
    def _v1_encode(x):
        with torch.no_grad():
            st    = model.st_branch(x)
            tf    = model.tf_branch(x)
            fused = model.fusion(torch.cat([st, tf], dim=-1))
        return fused

    model.encode = _v1_encode


def load_price_encoder(symbol: str):
    """
    Load Sprint 2 checkpoint and return (model, price_embed_dim).

    Handles both model versions:
      v2_optimised  (apex_synth_runner_v2.APEXSTModelV2)  → encode() dim = fusion_dim//2 = 64
      v1_baseline   (sprint2_model.APEXSTModel)            → encode() dim = fusion_dim//2 = 256
    """
    ckpt_path = f'{symbol}_apex_st_best.pt'
    if not Path(ckpt_path).exists():
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
        _ensure_encode(model)   # no-op for V2, adds shim for V1

        # Probe actual embedding dimension with a single dummy forward pass
        with torch.no_grad():
            dummy   = torch.zeros(1, 60, feat_dim, device=DEVICE)
            emb_dim = model.encode(dummy).shape[-1]

        print(f"  ✓ Loaded Sprint 2 ({model_version}) — feat_dim={feat_dim}  "
              f"price_embed_dim={emb_dim}")
        return model, emb_dim   # return actual embed dim, not feat_dim
    except Exception as e:
        print(f"  ⚠  Could not load Sprint 2 model: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# FROZEN MODALITY EMBEDDINGS  (pre-compute once, then use as fixed inputs)
# ─────────────────────────────────────────────────────────────────────────────
def extract_price_embeddings(apex_model, X: np.ndarray) -> np.ndarray:
    """
    Run Sprint 2 model in eval mode via encode(), extract pre-head embeddings.

    encode() is the only correct path:
      • APEXSTModelV2  — native encode() uses .st_branch / .tf_branch / .fusion
      • APEXSTModel    — _ensure_encode() shim uses the same real attributes
    Both models' branch attributes are st_branch / tf_branch (never branch1 /
    branch2, which have never existed in this codebase).

    Returns: (N, embed_dim)  where embed_dim is determined by the loaded model.
    """
    apex_model.eval()
    embs = []

    with torch.no_grad():
        for i in range(0, len(X), 64):
            batch = torch.tensor(X[i:i + 64], dtype=torch.float32).to(DEVICE)
            embs.append(apex_model.encode(batch).cpu().numpy())

    return np.vstack(embs).astype(np.float32)


def load_modality_data(symbol: str, split: str):
    """
    Load all modality embeddings for a given symbol and split.
    Returns: (price_emb, sent_emb, gat_emb, y_reg, y_cls) or None on failure.
    """
    sent_path = Path(f'{symbol}_apex_sent_{split}.npy')
    gat_path  = Path(f'{symbol}_apex_gat_{split}.npy')
    yr_path   = Path(f'{symbol}_apex_y_reg_{split}.npy')
    yc_path   = Path(f'{symbol}_apex_y_cls_{split}.npy')
    x_path    = Path(f'{symbol}_apex_X_{split}.npy')

    missing = [str(p) for p in [sent_path, gat_path, yr_path, yc_path, x_path]
               if not p.exists()]
    if missing:
        return None, missing

    sent  = np.load(sent_path).astype(np.float32)
    gat   = np.load(gat_path).astype(np.float32)
    y_reg = np.load(yr_path).squeeze().astype(np.float32)
    y_cls = np.load(yc_path).squeeze().astype(np.float32)
    X     = np.load(x_path).astype(np.float32)

    # Truncate all to same length
    n = min(len(sent), len(gat), len(y_reg), len(y_cls), len(X))
    return (X[:n], sent[:n], gat[:n], y_reg[:n], y_cls[:n]), []


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE SCORE
# ─────────────────────────────────────────────────────────────────────────────
def compute_confidence(cls_prob: torch.Tensor,
                       reg_out:  torch.Tensor,
                       reg_scale: float = 1.0) -> torch.Tensor:
    """
    Trade confidence score 0–100.

    Formula: confidence = cls_certainty * magnitude_bonus * 100
      cls_certainty  = |cls_prob - 0.5| * 2        (0=max uncertainty, 1=max certainty)
      magnitude_bonus = sigmoid(|reg_out| / reg_scale)  (bigger move = more confident)

    Both factors must agree for high confidence.
    """
    cls_certainty   = (cls_prob - 0.5).abs() * 2          # (B,)
    magnitude_bonus = torch.sigmoid(reg_out.abs() / (reg_scale + 1e-8))
    confidence      = cls_certainty * magnitude_bonus * 100
    return confidence.clamp(0, 100)


# ─────────────────────────────────────────────────────────────────────────────
# FUSION VARIANT A — GATED ATTENTION
# ─────────────────────────────────────────────────────────────────────────────
class GatedFusion(nn.Module):
    """
    Learns a soft gate (softmax weights) over the three modalities:
    price (256-d), sentiment (64-d), graph (64-d).

    Each modality is first projected to FUSION_DIM, then a gate network
    produces 3 attention weights that are used to compute a weighted sum.
    The gated representation is passed through a final MLP.
    """
    def __init__(self, price_dim=64, sent_dim=SENT_DIM,
                 gat_dim=GAT_DIM, hidden=FUSION_DIM, dropout=DROPOUT):
        super().__init__()
        # Per-modality projection to common dim
        self.proj_price = nn.Sequential(
            nn.Linear(price_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.proj_sent  = nn.Sequential(
            nn.Linear(sent_dim,  hidden), nn.LayerNorm(hidden), nn.GELU())
        self.proj_gat   = nn.Sequential(
            nn.Linear(gat_dim,   hidden), nn.LayerNorm(hidden), nn.GELU())

        # Gate network: takes concatenation of all projections → 3 weights
        self.gate = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),   # 3 modality weights
        )

        # Final MLP after gated sum
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
        )

        # Output heads
        self.reg_head = nn.Linear(hidden // 2, 1)
        self.cls_head = nn.Linear(hidden // 2, 1)

        self._hidden = hidden

    def forward(self, price, sent, gat):
        """
        price: (B, price_dim)
        sent:  (B, sent_dim)
        gat:   (B, gat_dim)
        """
        p = self.proj_price(price)   # (B, H)
        s = self.proj_sent(sent)
        g = self.proj_gat(gat)

        # Gate weights
        cat    = torch.cat([p, s, g], dim=-1)      # (B, 3H)
        gates  = F.softmax(self.gate(cat), dim=-1)  # (B, 3)

        # Weighted sum
        fused  = (gates[:, 0:1] * p +
                  gates[:, 1:2] * s +
                  gates[:, 2:3] * g)               # (B, H)

        out    = self.mlp(fused)
        reg    = self.reg_head(out).squeeze(-1)
        cls    = self.cls_head(out).squeeze(-1)
        conf   = compute_confidence(torch.sigmoid(cls), reg)

        return reg, cls, conf, gates   # return gates for interpretability


# ─────────────────────────────────────────────────────────────────────────────
# FUSION VARIANT B — CONCAT MLP (BASELINE)
# ─────────────────────────────────────────────────────────────────────────────
class ConcatFusion(nn.Module):
    """
    Simple baseline: concatenate all modalities → deep MLP.
    """
    def __init__(self, price_dim=64, sent_dim=SENT_DIM,
                 gat_dim=GAT_DIM, hidden=FUSION_DIM, dropout=DROPOUT):
        super().__init__()
        total = price_dim + sent_dim + gat_dim

        self.mlp = nn.Sequential(
            nn.Linear(total,   hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
        )
        self.reg_head = nn.Linear(hidden // 2, 1)
        self.cls_head = nn.Linear(hidden // 2, 1)

    def forward(self, price, sent, gat):
        x   = torch.cat([price, sent, gat], dim=-1)
        out = self.mlp(x)
        reg = self.reg_head(out).squeeze(-1)
        cls = self.cls_head(out).squeeze(-1)
        conf = compute_confidence(torch.sigmoid(cls), reg)
        return reg, cls, conf, None


# ─────────────────────────────────────────────────────────────────────────────
# FUSION VARIANT C — CROSS-MODAL TRANSFORMER
# ─────────────────────────────────────────────────────────────────────────────
class CrossModalTransformer(nn.Module):
    """
    Treats each modality as a token in a 3-token sequence.
    Multi-head self-attention across modalities lets each branch
    attend to the others. Final representation = mean of all tokens.
    """
    def __init__(self, price_dim=64, sent_dim=SENT_DIM,
                 gat_dim=GAT_DIM, hidden=FUSION_DIM,
                 n_heads=N_HEADS, n_layers=N_CM_LAYERS, dropout=DROPOUT):
        super().__init__()
        # Project all modalities to same token dim
        self.proj_price = nn.Linear(price_dim, hidden)
        self.proj_sent  = nn.Linear(sent_dim,  hidden)
        self.proj_gat   = nn.Linear(gat_dim,   hidden)

        # Modality positional embeddings (learned)
        self.modal_emb = nn.Parameter(torch.randn(3, hidden) * 0.02)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model    = hidden,
            nhead      = n_heads,
            dim_feedforward = hidden * 4,
            dropout    = dropout,
            batch_first= True,
            norm_first = True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Output MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.reg_head = nn.Linear(hidden // 2, 1)
        self.cls_head = nn.Linear(hidden // 2, 1)

    def forward(self, price, sent, gat):
        # Project to token dim
        p = self.proj_price(price).unsqueeze(1)   # (B, 1, H)
        s = self.proj_sent(sent).unsqueeze(1)
        g = self.proj_gat(gat).unsqueeze(1)

        # Stack into token sequence + add modality embeddings
        tokens = torch.cat([p, s, g], dim=1)       # (B, 3, H)
        tokens = tokens + self.modal_emb.unsqueeze(0)

        # Cross-modal attention
        attended = self.transformer(tokens)         # (B, 3, H)

        # Aggregate: mean over modality tokens
        agg = attended.mean(dim=1)                  # (B, H)

        out  = self.mlp(agg)
        reg  = self.reg_head(out).squeeze(-1)
        cls  = self.cls_head(out).squeeze(-1)
        conf = compute_confidence(torch.sigmoid(cls), reg)
        return reg, cls, conf, None


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def make_dataset(price_emb, sent, gat, y_reg, y_cls):
    return TensorDataset(
        torch.tensor(price_emb, dtype=torch.float32),
        torch.tensor(sent,      dtype=torch.float32),
        torch.tensor(gat,       dtype=torch.float32),
        torch.tensor(y_reg,     dtype=torch.float32),
        torch.tensor(y_cls,     dtype=torch.float32),
    )


def train_fusion(model, train_ds, val_ds, variant_name: str,
                 epochs=EPOCHS, patience=PATIENCE, lr=LR, batch=BATCH):
    model = model.to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr/20)

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False)

    best_val  = float('inf')
    best_state= None
    no_improve= 0
    best_epoch= 0
    history   = []

    print(f"\n  Training {variant_name}")
    print(f"  {'Epoch':>5}  {'Train':>10}  {'Val':>10}  {'DirAcc':>8}  {'Corr':>7}  {'Conf':>7}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*7}  {'─'*7}")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for price, sent, gat, yr, yc in train_dl:
            price, sent, gat = price.to(DEVICE), sent.to(DEVICE), gat.to(DEVICE)
            yr, yc = yr.to(DEVICE), yc.to(DEVICE)

            reg, cls, conf, _ = model(price, sent, gat)
            loss_reg = F.huber_loss(reg, yr)
            loss_cls = F.binary_cross_entropy_with_logits(cls, yc)
            loss     = loss_reg + 0.5 * loss_cls

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()

        sched.step()
        train_loss /= len(train_dl)

        # Validate
        model.eval()
        val_loss = 0.0
        all_reg, all_cls_p, all_yr, all_yc, all_conf = [], [], [], [], []
        with torch.no_grad():
            for price, sent, gat, yr, yc in val_dl:
                price, sent, gat = price.to(DEVICE), sent.to(DEVICE), gat.to(DEVICE)
                yr, yc = yr.to(DEVICE), yc.to(DEVICE)
                reg, cls, conf, _ = model(price, sent, gat)
                loss_reg = F.huber_loss(reg, yr)
                loss_cls = F.binary_cross_entropy_with_logits(cls, yc)
                val_loss += (loss_reg + 0.5 * loss_cls).item()
                all_reg.append(reg.cpu()); all_yr.append(yr.cpu())
                all_cls_p.append(torch.sigmoid(cls).cpu())
                all_yc.append(yc.cpu()); all_conf.append(conf.cpu())

        val_loss /= len(val_dl)
        reg_arr  = torch.cat(all_reg).numpy()
        yr_arr   = torch.cat(all_yr).numpy()
        cls_arr  = torch.cat(all_cls_p).numpy()
        yc_arr   = torch.cat(all_yc).numpy()
        conf_arr = torch.cat(all_conf).numpy()

        dir_acc = ((cls_arr > 0.5).astype(int) == yc_arr.astype(int)).mean() * 100
        corr    = float(np.corrcoef(reg_arr, yr_arr)[0, 1]) if reg_arr.std() > 1e-8 else 0.0
        avg_conf= conf_arr.mean()

        print(f"  {epoch:>5}  {train_loss:>10.5f}  {val_loss:>10.5f}  "
              f"{dir_acc:>7.2f}%  {corr:>7.3f}  {avg_conf:>6.1f}")

        history.append({'epoch': epoch, 'val_loss': val_loss,
                        'dir_acc': dir_acc, 'corr': corr, 'conf': avg_conf})

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_epoch, history


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, test_ds, variant_name: str, symbol: str):
    model.eval()
    dl = DataLoader(test_ds, batch_size=64, shuffle=False)

    all_reg, all_cls_p, all_yr, all_yc, all_conf, all_gates = [], [], [], [], [], []
    with torch.no_grad():
        for price, sent, gat, yr, yc in dl:
            price, sent, gat = price.to(DEVICE), sent.to(DEVICE), gat.to(DEVICE)
            yr, yc = yr.to(DEVICE), yc.to(DEVICE)
            reg, cls, conf, gates = model(price, sent, gat)
            all_reg.append(reg.cpu()); all_yr.append(yr.cpu())
            all_cls_p.append(torch.sigmoid(cls).cpu())
            all_yc.append(yc.cpu()); all_conf.append(conf.cpu())
            if gates is not None:
                all_gates.append(gates.cpu())

    reg_arr  = torch.cat(all_reg).numpy()
    yr_arr   = torch.cat(all_yr).numpy()
    cls_arr  = torch.cat(all_cls_p).numpy()
    yc_arr   = torch.cat(all_yc).numpy()
    conf_arr = torch.cat(all_conf).numpy()

    dir_acc  = ((cls_arr > 0.5).astype(int) == yc_arr.astype(int)).mean() * 100
    corr     = float(np.corrcoef(reg_arr, yr_arr)[0, 1]) if reg_arr.std() > 1e-8 else 0.0
    avg_conf = conf_arr.mean()
    n        = len(reg_arr)

    # Gate analysis (FusionA only)
    gate_means = None
    if all_gates:
        gates_arr  = torch.cat(all_gates).numpy()   # (N, 3)
        gate_means = gates_arr.mean(axis=0)          # [price, sent, gat]

    results = {
        'variant':   variant_name,
        'symbol':    symbol,
        'dir_acc':   round(float(dir_acc), 2),
        'corr':      round(float(corr), 3),
        'avg_conf':  round(float(avg_conf), 1),
        'n':         n,
        'gate_means': gate_means.tolist() if gate_means is not None else None,
    }
    return results, reg_arr, cls_arr, conf_arr


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION TABLE
# ─────────────────────────────────────────────────────────────────────────────
def print_ablation(all_results: list[dict]):
    print(f"\n{'█'*68}")
    print("  SPRINT 4 — ABLATION TABLE")
    print(f"{'█'*68}\n")

    # Header
    print(f"  {'Symbol':<12} {'Variant':<24} {'Dir Acc':>8} {'Corr':>7} {'Confidence':>11} {'Winner':>7}")
    print(f"  {'─'*12} {'─'*24} {'─'*8} {'─'*7} {'─'*11} {'─'*7}")

    # Group by symbol
    symbols = list(dict.fromkeys(r['symbol'] for r in all_results))
    summary = {}

    for sym in symbols:
        rows = [r for r in all_results if r['symbol'] == sym]
        # Find best variant per metric
        best_acc  = max(rows, key=lambda r: r['dir_acc'])
        best_corr = max(rows, key=lambda r: r['corr'])

        for r in rows:
            is_best = (r == best_acc or r == best_corr)
            winner  = '★' if r == best_acc else ' '
            print(f"  {r['symbol']:<12} {r['variant']:<24} "
                  f"{r['dir_acc']:>7.2f}%  {r['corr']:>7.3f}  "
                  f"{r['avg_conf']:>10.1f}  {winner:>7}")

            # Gate analysis for FusionA
            if r.get('gate_means'):
                g = r['gate_means']
                print(f"  {'':12} {'  └ Gate weights:':24} "
                      f"price={g[0]:.3f}  sent={g[1]:.3f}  gat={g[2]:.3f}")

        # Best overall for this symbol
        best = max(rows, key=lambda r: r['dir_acc'] + r['corr'])
        summary[sym] = best
        print(f"  {'─'*12} {'─'*24} {'─'*8} {'─'*7} {'─'*11} {'─'*7}")

    # Overall winner
    print(f"\n  BEST VARIANT PER SYMBOL")
    print(f"  {'─'*50}")
    variant_wins = {v: 0 for v in VARIANTS}
    for sym, best in summary.items():
        variant_wins[best['variant']] = variant_wins.get(best['variant'], 0) + 1
        print(f"  {sym:<12} → {best['variant']:<24} "
              f"acc={best['dir_acc']:.2f}%  corr={best['corr']:.3f}  "
              f"conf={best['avg_conf']:.1f}")

    print(f"\n  VARIANT WINS: " +
          "  ".join(f"{v.split('_')[0]}={n}" for v, n in variant_wins.items()))

    overall_winner = max(variant_wins, key=variant_wins.get)
    print(f"\n  ★  RECOMMENDED: {overall_winner}")
    print(f"     → Use this variant's checkpoints for Sprint 5 live trading")
    print(f"\n{'█'*68}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# SAVE CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(model, symbol: str, variant: str, results: dict, price_dim: int):
    fname = f'{symbol}_fusion_{variant}_best.pt'
    torch.save({
        'model_state':     model.state_dict(),
        'variant':         variant,
        'symbol':          symbol,
        'price_embed_dim': price_dim,   # actual dim of model.encode() output
        'results':         results,
    }, fname)
    print(f"  ✓ Saved {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY PRINTOUT
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(price_dim: int = 64):
    """Print architecture overview. price_dim defaults to V2's actual output dim."""
    fusion_in = price_dim + SENT_DIM + GAT_DIM
    print(f"\n  SPRINT 4 — Cross-Modal Fusion Architecture")
    print(f"  {'─'*60}")
    print(f"  Frozen inputs")
    print(f"    Price emb  : APEX-ST Sprint 2 penultimate layer  ({price_dim}-d)")
    print(f"    Sentiment  : FinBERT projector Sprint 3           ({SENT_DIM}-d)")
    print(f"    Graph      : GAT embedding Sprint 3               ({GAT_DIM}-d)")
    print(f"    Total input: {fusion_in}-d  (all frozen, no grad)")
    print(f"\n  Fusion variants (trained)")

    mA = GatedFusion(price_dim=price_dim)
    pA = sum(p.numel() for p in mA.parameters())
    print(f"    FusionA Gated       : {pA:>8,} params  "
          f"(3 modality gates + weighted sum + MLP)")

    mB = ConcatFusion(price_dim=price_dim)
    pB = sum(p.numel() for p in mB.parameters())
    print(f"    FusionB MLP         : {pB:>8,} params  "
          f"(concat {fusion_in}-d → MLP)")

    mC = CrossModalTransformer(price_dim=price_dim)
    pC = sum(p.numel() for p in mC.parameters())
    print(f"    FusionC Transformer : {pC:>8,} params  "
          f"({N_HEADS}-head × {N_CM_LAYERS}-layer cross-modal attn)")

    print(f"\n  Outputs  : reg (return) + cls (direction) + confidence (0-100)")
    print(f"  Training : frozen branches, only fusion params trained")
    print(f"  {'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# PER-SYMBOL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def process_symbol(symbol: str, all_results: list):
    print(f"\n{BANNER}")
    print(f"  SPRINT 4 — {symbol}")
    print(BANNER)

    # Load Sprint 2 model for price embeddings.
    # load_price_encoder now returns (model, actual_embed_dim) — the true
    # dimensionality of model.encode() output, probed via a dummy forward pass.
    apex_model, price_dim = load_price_encoder(symbol)

    if apex_model is None:
        print(f"  ⚠  No Sprint 2 checkpoint — using synthetic price embeddings "
              f"(64-d, V2 default)")
        price_dim = 64   # match V2 default so fusion dims are consistent

    # Load modality data for all splits
    splits_data = {}
    for split in ['train', 'val', 'test']:
        data, missing = load_modality_data(symbol, split)
        if data is None:
            print(f"  ❌ Missing files for {symbol}/{split}:")
            for m in missing: print(f"     {m}")
            return False
        splits_data[split] = data

    # Extract price embeddings (frozen) via model.encode() — real signal
    print(f"  Extracting frozen price embeddings from Sprint 2 "
          f"(embed_dim={price_dim})...")
    price_embs = {}
    for split in ['train', 'val', 'test']:
        X, sent, gat, yr, yc = splits_data[split]
        if apex_model is not None:
            pe = extract_price_embeddings(apex_model, X)
            # Sanity check — should never mismatch now that we probe at load time
            assert pe.shape[1] == price_dim, (
                f"Price embed dim mismatch after encode(): "
                f"got {pe.shape[1]}, expected {price_dim}"
            )
        else:
            pe = np.random.randn(len(X), price_dim).astype(np.float32) * 0.1
        price_embs[split] = pe
        print(f"    {split:>5}: price={pe.shape}  sent={splits_data[split][1].shape}  "
              f"gat={splits_data[split][2].shape}")

    # Build datasets — re-align all modalities to shortest array length
    # GAT embeddings may have slightly different row count than Sprint 1/2
    # arrays depending on walk-forward split boundary arithmetic.
    def make_ds(split):
        _, sent, gat, yr, yc = splits_data[split]
        pe = price_embs[split]
        n  = min(len(pe), len(sent), len(gat), len(yr), len(yc))
        if n < len(pe):
            print(f"    ⚠  {split}: aligning lengths → "
                  f"price={len(pe)} sent={len(sent)} gat={len(gat)} → trimmed to {n}")
        return make_dataset(pe[:n], sent[:n], gat[:n], yr[:n], yc[:n])

    train_ds = make_ds('train')
    val_ds   = make_ds('val')
    test_ds  = make_ds('test')

    # Train all three variants — pass price_dim so each fusion model is built
    # with the correct input dimension for this symbol's checkpoint
    variant_models = {
        'FusionA_Gated':       GatedFusion(price_dim=price_dim),
        'FusionB_MLP':         ConcatFusion(price_dim=price_dim),
        'FusionC_Transformer': CrossModalTransformer(price_dim=price_dim),
    }

    sym_results = []
    for vname, model in variant_models.items():
        trained_model, best_epoch, history = train_fusion(
            model, train_ds, val_ds, vname)
        results, reg_arr, cls_arr, conf_arr = evaluate(
            trained_model, test_ds, vname, symbol)

        print(f"\n  ┌─────────────────────────────────────────────┐")
        print(f"  │  {vname:<43}│")
        print(f"  ├─────────────────────────────────────────────┤")
        print(f"  │  Dir Acc    : {results['dir_acc']:>6.2f}%                        │")
        print(f"  │  Corr       : {results['corr']:>6.3f}                         │")
        print(f"  │  Confidence : {results['avg_conf']:>6.1f} / 100                   │")
        if results.get('gate_means'):
            g = results['gate_means']
            print(f"  │  Gate: price={g[0]:.2f} sent={g[1]:.2f} gat={g[2]:.2f}       │")
        print(f"  │  Best epoch : {best_epoch:<6}                          │")
        print(f"  └─────────────────────────────────────────────┘")

        save_checkpoint(trained_model, symbol, vname, results, price_dim)
        all_results.append(results)
        sym_results.append(results)

    # Best variant for this symbol
    best = max(sym_results, key=lambda r: r['dir_acc'] + r['corr'])
    print(f"\n  ★  Best for {symbol}: {best['variant']}  "
          f"(acc={best['dir_acc']:.2f}%  corr={best['corr']:.3f})")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('symbols',         nargs='*', default=SYMBOLS)
    p.add_argument('--summary',       action='store_true')
    p.add_argument('--ablation-only', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    # Strip shell comment tokens
    raw = args.symbols
    filtered = []
    for tok in raw:
        if tok.startswith('#'):
            break
        filtered.append(tok)
    symbols = [s.upper() for s in filtered] if filtered else SYMBOLS

    print("\n" + "█" * 68)
    print("  SPRINT 4 — CROSS-MODAL FUSION (APEX-ST FULL MODEL)")
    print(f"  Variants : {', '.join(VARIANTS)}")
    print(f"  Mode     : Freeze Sprint 2+3, train fusion only")
    print(f"  Device   : {DEVICE}")
    print("█" * 68)

    print_summary(price_dim=64)  # 64 = V2 default; overridden per-symbol at runtime

    if args.summary:
        return

    if args.ablation_only:
        abl_path = Path('sprint4_ablation.json')
        if abl_path.exists():
            with open(abl_path) as f:
                all_results = json.load(f)
            print_ablation(all_results)
        else:
            print("  No ablation results found. Run training first.")
        return

    all_results = []
    status      = {}

    for sym in symbols:
        ok = process_symbol(sym, all_results)
        status[sym] = '✅' if ok else '❌'

    # Save ablation results
    if all_results:
        with open('sprint4_ablation.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print("\n  ✓ Ablation results saved → sprint4_ablation.json")

    # Print ablation table
    if all_results:
        summary = print_ablation(all_results)

    # Final status
    print(f"\n{BANNER}")
    print("  SPRINT 4 TRAINING SUMMARY")
    print(BANNER)
    for sym, s in status.items():
        print(f"  {s}  {sym}")
    passed = sum(1 for v in status.values() if v == '✅')
    print(f"\n  {passed}/{len(symbols)} symbols trained")
    print("""
  Checkpoints: {SYMBOL}_fusion_{variant}_best.pt
  Ablation   : sprint4_ablation.json

  For inference with best variant:
    ckpt  = torch.load('RELIANCE_fusion_FusionA_Gated_best.pt')
    pdim  = ckpt['price_embed_dim']   # 64 for V2, 256 for V1 baseline
    model = GatedFusion(price_dim=pdim)
    model.load_state_dict(ckpt['model_state'])
    reg, cls, conf, gates = model(price_emb, sent_emb, gat_emb)
    print(f"Confidence: {{conf.item():.1f}}/100")

  Next → Sprint 5: python sprint5_ensemble.py
    """)
    print(BANNER)


if __name__ == '__main__':
    main()
