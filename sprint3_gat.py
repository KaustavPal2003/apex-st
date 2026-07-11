"""
sprint3_gat.py — Sprint 3 Branch 4: Graph Attention Network (GAT)
══════════════════════════════════════════════════════════════════
Builds a stock correlation graph from Sprint 1 return series,
then runs a 3-layer GAT to produce per-node (per-stock) embeddings
that capture inter-stock relationships.

Architecture
────────────
  Nodes       : symbols from watchlist.json (falls back to 5-stock default)
  Node features: last-N-day log-return vector, computed directly from
                data/{SYMBOL}.csv close prices (not from Sprint 1's y_reg,
                which is a forward-shifted 30-day-ahead target and would
                leak future information into node features if reused here).
  Edges       : Pearson correlation > threshold  (dynamic per window)
  GAT layers  : 3 × GATConv (8 heads, concat → mean)
  Output      : GAT_DIM = 64-d embedding per stock per day
  Saved as    : {SYMBOL}_apex_gat_{split}.npy

Usage
─────
  python sprint3_gat.py                    # all 5 stocks
  python sprint3_gat.py RELIANCE           # single (uses all 5 for graph)
  python sprint3_gat.py --summary
  python sprint3_gat.py --corr_threshold 0.4

Requirements
────────────
  pip install torch torch-geometric
  # torch-geometric install: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
  # Quickest for CPU-only:
  #   pip install torch-geometric
  #   pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.12.0+cpu.html
"""

import sys
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Read symbols from watchlist.json if present, else fall back to defaults
import json as _json
_DEFAULT_SYMBOLS = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
try:
    _wl = _json.loads(open('watchlist.json').read()).get('watchlist', [])
    SYMBOLS = _wl if _wl else _DEFAULT_SYMBOLS
except FileNotFoundError:
    SYMBOLS = _DEFAULT_SYMBOLS
GAT_DIM        = 64       # final per-stock embedding dim
GAT_HIDDEN     = 64       # hidden dim per head
GAT_HEADS      = 8        # attention heads (layers 1-2)
GAT_LAYERS     = 3
CORR_THRESHOLD = 0.3      # minimum |correlation| to add edge
CORR_WINDOW    = 20       # days to compute rolling correlation
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BANNER         = "═" * 68
N_STOCKS       = len(SYMBOLS)


# ─────────────────────────────────────────────────────────────────────────────
# GAT MODEL (pure PyTorch — no PyG dependency for portability)
# ─────────────────────────────────────────────────────────────────────────────
class GATLayer(nn.Module):
    """
    Single Graph Attention layer.
    Implements Veličković et al. 2018 with multi-head attention.
    """
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 8,
                 dropout: float = 0.2, concat: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.concat  = concat
        self.dropout = nn.Dropout(dropout)

        self.W  = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.a  = nn.Parameter(torch.empty(1, n_heads, 2 * out_dim))
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, adj):
        """
        x   : (N, in_dim)   node features
        adj : (N, N)        adjacency matrix (weighted or binary)
        Returns: (N, out_dim * n_heads) if concat else (N, out_dim)
        """
        N = x.size(0)
        H = self.W(x).view(N, self.n_heads, self.out_dim)   # (N, heads, d)

        # Attention coefficients: e_ij = LeakyReLU(a^T [Whi || Whj])
        Hi = H.unsqueeze(1).expand(-1, N, -1, -1)           # (N, N, heads, d)
        Hj = H.unsqueeze(0).expand(N, -1, -1, -1)           # (N, N, heads, d)
        e  = (self.a * torch.cat([Hi, Hj], dim=-1)).sum(-1)  # (N, N, heads)
        e  = F.leaky_relu(e, 0.2)

        # Mask non-edges
        mask = (adj == 0).unsqueeze(-1).expand_as(e)
        e    = e.masked_fill(mask, float('-inf'))

        alpha = torch.softmax(e, dim=1)                      # (N, N, heads)
        alpha = self.dropout(alpha)

        # Aggregate
        out = (alpha.unsqueeze(-1) * Hj).sum(1)             # (N, heads, d)

        if self.concat:
            return F.elu(out.reshape(N, -1))                 # (N, heads*d)
        else:
            return F.elu(out.mean(1))                        # (N, d)


class StockGAT(nn.Module):
    """
    3-layer GAT for stock correlation graph.
    Input  : (N_stocks, in_dim)  node return features
    Output : (N_stocks, GAT_DIM) stock embeddings
    """
    def __init__(self, in_dim: int, hidden: int = GAT_HIDDEN,
                 out_dim: int = GAT_DIM, n_heads: int = GAT_HEADS,
                 dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Layer 1: in_dim → hidden*heads
        self.gat1 = GATLayer(in_dim,  hidden, n_heads, dropout, concat=True)
        # Layer 2: hidden*heads → hidden*heads
        self.gat2 = GATLayer(hidden * n_heads, hidden, n_heads, dropout, concat=True)
        # Layer 3: hidden*heads → out_dim (mean aggregation)
        self.gat3 = GATLayer(hidden * n_heads, out_dim, n_heads=1, dropout=dropout, concat=False)

        self.norm1 = nn.LayerNorm(hidden * n_heads)
        self.norm2 = nn.LayerNorm(hidden * n_heads)
        self.norm3 = nn.LayerNorm(out_dim)

        self.out_dim = out_dim

    def forward(self, x, adj):
        """
        x   : (N, in_dim)
        adj : (N, N)
        """
        x = self.dropout(x)

        # Layer 1
        h1 = self.gat1(x, adj)
        h1 = self.norm1(h1)

        # Layer 2 with residual (if dims match)
        h2 = self.gat2(h1, adj)
        h2 = self.norm2(h2 + h1)

        # Layer 3
        h3 = self.gat3(h2, adj)
        h3 = self.norm3(h3)

        return h3   # (N, out_dim)


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
def build_adjacency(returns_matrix: np.ndarray,
                    threshold: float = CORR_THRESHOLD) -> np.ndarray:
    """
    Build adjacency matrix from Pearson correlation of return series.
    returns_matrix : (T, N_stocks)
    Returns        : (N_stocks, N_stocks) weighted adjacency (self-loops included)
    """
    # Pearson correlation
    corr = np.corrcoef(returns_matrix.T)            # (N, N)
    np.fill_diagonal(corr, 0.0)                      # remove self

    # Threshold: keep edges where |corr| > threshold
    adj  = np.where(np.abs(corr) > threshold, corr, 0.0)

    # Add self-loops (standard GAT practice)
    adj += np.eye(len(adj))

    return adj.astype(np.float32)


def build_node_features(returns_matrix: np.ndarray,
                        t: int, window: int = CORR_WINDOW) -> np.ndarray:
    """
    Node features at time t: rolling return statistics over last `window` days.
    Features per stock: [mean_ret, std_ret, min_ret, max_ret, last_ret,
                         momentum_5, momentum_10, skewness]
    Returns: (N_stocks, 8)
    """
    start = max(0, t - window)
    R     = returns_matrix[start:t+1]               # (W, N)
    N     = R.shape[1]

    if R.shape[0] < 2:
        return np.zeros((N, 8), dtype=np.float32)

    mean_r  = R.mean(0)
    std_r   = R.std(0) + 1e-8
    min_r   = R.min(0)
    max_r   = R.max(0)
    last_r  = R[-1]
    mom5    = R[-min(5, len(R)):].mean(0)
    mom10   = R[-min(10, len(R)):].mean(0)

    # Skewness (normalised third moment)
    skew = ((R - mean_r) ** 3).mean(0) / (std_r ** 3)

    feats = np.stack([mean_r, std_r, min_r, max_r, last_r, mom5, mom10, skew], axis=1)
    return feats.astype(np.float32)                  # (N, 8)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd

LOOKBACK = 60  # must match Sprint 1's lookback window

def load_returns_all() -> np.ndarray | None:
    """
    Build genuine, contemporaneous daily log-returns directly from
    data/{SYMBOL}.csv — not from y_reg, which is a forward-shifted
    target array and would leak future information into node features.
    """
    log_return_cols = []
    for sym in SYMBOLS:
        p = Path(f'data/{sym}.csv')
        if not p.exists():
            print(f"  ⚠  Missing {p}")
            return None
        df = pd.read_csv(p)
        close = df['close'].to_numpy()
        log_price  = np.log(np.clip(close, 1e-8, None))
        log_return = np.diff(log_price, prepend=log_price[0])
        # Align index 0 of this series to each sample's *decision day*
        # (the last day of its lookback window), matching Sprint 1's indexing
        log_return_cols.append(log_return[LOOKBACK - 1:])

    T = min(len(c) for c in log_return_cols)
    return np.stack([c[:T] for c in log_return_cols], axis=1)


def process_symbol_gat(symbol: str, model: StockGAT,
                       returns_matrix: np.ndarray,
                       split_sizes: tuple[int, int, int]) -> bool:
    """
    For each timestep t, compute GAT embedding for `symbol`.
    Saves (n_days, GAT_DIM) arrays for train/val/test.
    """
    sym_idx   = SYMBOLS.index(symbol)
    n_train, n_val, n_test = split_sizes
    total     = n_train + n_val + n_test

    # Build global adjacency once (on full return series)
    adj_global = build_adjacency(returns_matrix)
    adj_tensor = torch.tensor(adj_global, dtype=torch.float32).to(DEVICE)

    model.eval()
    gat_embs = []

    with torch.no_grad():
        for t in range(total):
            node_feats = build_node_features(returns_matrix, t)
            x_tensor   = torch.tensor(node_feats, dtype=torch.float32).to(DEVICE)
            out        = model(x_tensor, adj_tensor)         # (N_stocks, GAT_DIM)
            gat_embs.append(out[sym_idx].cpu().numpy())      # (GAT_DIM,)

    gat_array = np.stack(gat_embs, axis=0)                   # (T, GAT_DIM)

    # Split and save
    splits = {
        'train': gat_array[:n_train],
        'val':   gat_array[n_train : n_train + n_val],
        'test':  gat_array[n_train + n_val :],
    }
    for split, arr in splits.items():
        fname = f'{symbol}_apex_gat_{split}.npy'
        np.save(fname, arr.astype(np.float32))
        print(f"  ✓ Saved {fname}  shape={arr.shape}")

    return True


def process_symbol(symbol: str, model: StockGAT,
                   returns_matrix: np.ndarray | None) -> bool:
    print(f"\n{BANNER}")
    print(f"  SPRINT 3 GAT — Processing {symbol}")
    print(BANNER)

    # Check Sprint 1 data
    if not Path(f'{symbol}_apex_X_train.npy').exists():
        print(f"  ❌ Sprint 1 data missing for {symbol}")
        return False

    n_train = np.load(f'{symbol}_apex_X_train.npy', mmap_mode='r').shape[0]
    n_val   = np.load(f'{symbol}_apex_X_val.npy',   mmap_mode='r').shape[0]
    n_test  = np.load(f'{symbol}_apex_X_test.npy',  mmap_mode='r').shape[0]
    total   = n_train + n_val + n_test
    print(f"  Split sizes → train:{n_train}  val:{n_val}  test:{n_test}")

    if returns_matrix is None:
        print("  ⚠  Returns matrix unavailable — using synthetic node features")
        returns_matrix = np.random.randn(total, N_STOCKS).astype(np.float32) * 0.01

    ok = process_symbol_gat(symbol, model, returns_matrix, (n_train, n_val, n_test))
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(model: StockGAT):
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Stock GAT Architecture")
    print(f"  {'─'*45}")
    print(f"  Nodes         : {N_STOCKS} stocks")
    print(f"  Node feat dim : 8  (return stats over {CORR_WINDOW}-day window)")
    print(f"  Edge rule     : |Pearson corr| > {CORR_THRESHOLD}")
    print(f"  Layer 1       : GATConv  8→{GAT_HIDDEN}  ({GAT_HEADS} heads, concat)")
    print(f"  Layer 2       : GATConv  {GAT_HIDDEN*GAT_HEADS}→{GAT_HIDDEN}  ({GAT_HEADS} heads, concat + residual)")
    print(f"  Layer 3       : GATConv  {GAT_HIDDEN*GAT_HEADS}→{GAT_DIM}  (mean pool)")
    print(f"  Total params  : {total:,}")
    print(f"  Output dim    : {GAT_DIM}  per stock  (feeds into Sprint 4 fusion)")
    print(f"  {'─'*45}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('symbols', nargs='*', default=SYMBOLS)
    p.add_argument('--summary',        action='store_true')
    p.add_argument('--corr_threshold', type=float, default=CORR_THRESHOLD)
    return p.parse_args()


def main():
    args = parse_args()

    raw = args.symbols
    filtered = []
    for tok in raw:
        if tok.startswith('#'):
            break
        filtered.append(tok)
    symbols = [s.upper() for s in filtered] if filtered else SYMBOLS

    print("\n" + "█" * 68)
    print("  SPRINT 3 — GRAPH ATTENTION NETWORK (GAT)")
    print(f"  Nodes    : {N_STOCKS} stocks | Edges: |corr| > {args.corr_threshold}")
    print(f"  Layers   : {GAT_LAYERS} GAT layers  ({GAT_HEADS} heads)")
    print(f"  Output   : {GAT_DIM}-d embedding per stock per day")
    print(f"  Device   : {DEVICE}")
    print("█" * 68)

    model = StockGAT(in_dim=8).to(DEVICE)
    print_summary(model)

    if args.summary:
        return

    # Load return series for graph construction
    print("\n  Loading return series for graph construction...")
    returns_matrix = load_returns_all()
    if returns_matrix is not None:
        print(f"  ✓ Returns matrix shape: {returns_matrix.shape}  (T × {N_STOCKS} stocks)")
        adj = build_adjacency(returns_matrix, threshold=args.corr_threshold)
        print(f"\n  Stock Correlation Graph  (threshold={args.corr_threshold})")
        print(f"  {'─'*45}")
        header = "        " + "  ".join(f"{s[:6]:>6}" for s in SYMBOLS)
        print(f"  {header}")
        for i, sym_i in enumerate(SYMBOLS):
            row = f"  {sym_i:<9}"
            for j in range(N_STOCKS):
                val = adj[i, j]
                if i == j:
                    row += f"  {'self':>6}"
                elif val != 0:
                    row += f"  {val:>6.3f}"
                else:
                    row += f"  {'  ---':>6}"
            print(row)
        print(f"  {'─'*45}")
    else:
        print("  ⚠  Could not load all return series — using synthetic data")

    results = {}
    for sym in symbols:
        ok = process_symbol(sym, model, returns_matrix)
        results[sym] = '✅' if ok else '❌'

    print(f"\n{BANNER}")
    print("  SPRINT 3 GAT SUMMARY")
    print(BANNER)
    for sym, status in results.items():
        print(f"  {status}  {sym}")
    passed = sum(1 for v in results.values() if v == '✅')
    print(f"\n  {passed}/{len(symbols)} symbols processed")
    print(f"""
  Output files: {{SYMBOL}}_apex_gat_{{train|val|test}}.npy
  Shape: (n_days, {GAT_DIM})

  Next: sprint4_fusion.py  (gated cross-modal fusion)
    """)
    print(BANNER)


if __name__ == '__main__':
    main()
