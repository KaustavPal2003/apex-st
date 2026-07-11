"""
sprint3_finbert.py — Sprint 3 Branch 3: FinBERT Sentiment
══════════════════════════════════════════════════════════
Loads FinBERT (ProsusAI/finbert) from HuggingFace, encodes
news headlines per symbol per day, produces a fixed-dim
sentiment embedding that plugs into the APEX-ST fusion layer.

Pipeline
────────
  Raw headlines (CSV or list)
    → FinBERT tokeniser
    → Pooled [CLS] embeddings   (768-d)
    → Daily aggregation (mean)
    → Projection MLP            (→ SENT_DIM = 64)
    → Saved as {SYMBOL}_apex_sent_{split}.npy

Usage
─────
  python sprint3_finbert.py                        # process all 5 symbols
  python sprint3_finbert.py RELIANCE               # single symbol
  python sprint3_finbert.py --dummy                # synthetic data (no GPU/HF needed)
  python sprint3_finbert.py --summary              # show architecture

Requirements
────────────
  pip install torch transformers sentencepiece
  # HuggingFace model auto-downloads on first run (~500 MB)

Headline CSV format  (one file per symbol, optional)
────────────────────
  date,headline
  2022-01-03,"Reliance Q3 profit jumps 35 pct on retail surge"
  2022-01-04,"Mukesh Ambani eyes green energy expansion"
  ...
  File path: {SYMBOL}_headlines.csv   (auto-detected)
  If absent: synthetic zero-sentiment embeddings are used.
"""

import sys
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Read symbols from watchlist.json if present, else fall back to defaults
import json as _json
_DEFAULT_SYMBOLS = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
try:
    _wl = _json.loads(open('watchlist.json').read()).get('watchlist', [])
    SYMBOLS = _wl if _wl else _DEFAULT_SYMBOLS
except FileNotFoundError:
    SYMBOLS = _DEFAULT_SYMBOLS
SENT_DIM  = 64          # projected sentiment embedding dim
BERT_DIM  = 768         # FinBERT hidden size
MAX_LEN   = 128         # max tokens per headline
BATCH_HF  = 32          # headlines per HF forward pass
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BANNER    = "═" * 68
HF_MODEL  = "ProsusAI/finbert"


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT PROJECTION HEAD
# ─────────────────────────────────────────────────────────────────────────────
class SentimentProjector(nn.Module):
    """
    Projects daily-aggregated FinBERT embeddings (768-d) down to SENT_DIM.
    Also has a 3-class sentiment classifier head (positive / negative / neutral)
    for auxiliary supervision.
    """
    def __init__(self, bert_dim: int = BERT_DIM, sent_dim: int = SENT_DIM,
                 dropout: float = 0.2):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(bert_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, sent_dim),
            nn.LayerNorm(sent_dim),
        )
        # Auxiliary head: positive / negative / neutral
        self.sentiment_cls = nn.Linear(sent_dim, 3)

    def forward(self, bert_emb):
        """bert_emb: (B, bert_dim) → (B, sent_dim)"""
        proj = self.projector(bert_emb)
        logits = self.sentiment_cls(proj)
        return proj, logits


# ─────────────────────────────────────────────────────────────────────────────
# FINBERT ENCODER  (lazy-loaded so --summary works without HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────
class FinBERTEncoder:
    """Wraps HuggingFace FinBERT; lazy-loads on first encode call."""

    def __init__(self, model_name: str = HF_MODEL, device=DEVICE):
        self.model_name = model_name
        self.device     = device
        self._tokenizer = None
        self._model     = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "transformers not installed.\n"
                "  pip install transformers sentencepiece"
            )
        print(f"  Loading FinBERT from HuggingFace ({self.model_name})...")
        print("  (first run downloads ~500 MB — cached afterwards)")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model     = AutoModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        print("  ✓ FinBERT loaded")

    @torch.no_grad()
    def encode(self, headlines: list[str]) -> np.ndarray:
        """
        Encode a list of headline strings.
        Returns: np.ndarray (N, 768) of [CLS] embeddings.
        """
        self._load()
        all_embs = []
        for i in range(0, len(headlines), BATCH_HF):
            batch = headlines[i : i + BATCH_HF]
            enc   = self._tokenizer(
                batch,
                padding       = True,
                truncation    = True,
                max_length    = MAX_LEN,
                return_tensors= 'pt',
            ).to(self.device)
            out   = self._model(**enc)
            cls   = out.last_hidden_state[:, 0, :].cpu().numpy()  # (B, 768)
            all_embs.append(cls)
        return np.vstack(all_embs)   # (N, 768)


# ─────────────────────────────────────────────────────────────────────────────
# HEADLINE LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_headlines(symbol: str) -> dict | None:
    """
    Load headlines from {SYMBOL}_headlines.csv.
    Expected columns: date (YYYY-MM-DD), headline (str).
    Returns dict {date_str: [headlines]} or None if file absent.
    """
    csv_path = Path(f'{symbol}_headlines.csv')
    if not csv_path.exists():
        return None
    try:
        import csv
        from collections import defaultdict
        data = defaultdict(list)
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data[row['date'].strip()].append(row['headline'].strip())
        print(f"  ✓ Loaded headlines: {sum(len(v) for v in data.values())} "
              f"across {len(data)} trading days")
        return dict(data)
    except Exception as e:
        print(f"  ⚠ Could not parse {csv_path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC SENTIMENT (fallback when no headlines available)
# ─────────────────────────────────────────────────────────────────────────────
def make_synthetic_sentiment(n_days: int, seed: int = 42) -> np.ndarray:
    """
    Generates plausible synthetic FinBERT embeddings using a random walk
    in the BERT embedding space. Used when real headlines are absent.
    Shape: (n_days, BERT_DIM)
    """
    rng  = np.random.RandomState(seed)
    base = rng.randn(BERT_DIM).astype(np.float32)
    base /= np.linalg.norm(base)

    embs = []
    current = base.copy()
    for _ in range(n_days):
        noise   = rng.randn(BERT_DIM).astype(np.float32) * 0.05
        current = current + noise
        current /= (np.linalg.norm(current) + 1e-8)
        embs.append(current.copy())

    return np.stack(embs)   # (n_days, 768)


# ─────────────────────────────────────────────────────────────────────────────
# DAILY AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_daily(date_to_headlines: dict, trading_dates: list[str],
                    encoder: FinBERTEncoder) -> np.ndarray:
    """
    For each trading date, encode all headlines and mean-pool them.
    If no headlines exist for a date, use zero vector (model learns to ignore).
    Returns: (T, BERT_DIM)
    """
    daily_embs = []
    for date in trading_dates:
        if date in date_to_headlines:
            hlines = date_to_headlines[date]
            embs   = encoder.encode(hlines)          # (k, 768)
            daily_embs.append(embs.mean(axis=0))
        else:
            daily_embs.append(np.zeros(BERT_DIM, dtype=np.float32))
    return np.stack(daily_embs)   # (T, 768)


# ─────────────────────────────────────────────────────────────────────────────
# PROJECTION & SAVE
# ─────────────────────────────────────────────────────────────────────────────
def project_and_save(symbol: str, bert_embs: np.ndarray,
                     projector: SentimentProjector,
                     split_sizes: tuple[int, int, int]):
    """
    Project BERT embeddings → SENT_DIM, split into train/val/test,
    save as .npy files matching Sprint 1/2 naming convention.
    """
    t_train, t_val, t_test = split_sizes
    total = t_train + t_val + t_test

    if len(bert_embs) < total:
        # Pad with zeros if we have fewer days than needed
        pad = np.zeros((total - len(bert_embs), BERT_DIM), dtype=np.float32)
        bert_embs = np.vstack([bert_embs, pad])
    elif len(bert_embs) > total:
        bert_embs = bert_embs[-total:]   # take the most recent

    tensor = torch.tensor(bert_embs, dtype=torch.float32).to(DEVICE)

    projector.eval()
    with torch.no_grad():
        proj, _ = projector(tensor)   # (T, SENT_DIM)
    proj = proj.cpu().numpy()

    splits = {
        'train': proj[:t_train],
        'val':   proj[t_train : t_train + t_val],
        'test':  proj[t_train + t_val :],
    }

    for split, arr in splits.items():
        fname = f'{symbol}_apex_sent_{split}.npy'
        np.save(fname, arr.astype(np.float32))
        print(f"  ✓ Saved {fname}  shape={arr.shape}")

    return splits


# ─────────────────────────────────────────────────────────────────────────────
# PER-SYMBOL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def process_symbol(symbol: str, encoder: FinBERTEncoder,
                   projector: SentimentProjector,
                   dummy: bool = False):
    print(f"\n{BANNER}")
    print(f"  SPRINT 3 FinBERT — Processing {symbol}")
    print(BANNER)

    # Load Sprint 1 X shapes to infer split sizes
    x_train_path = Path(f'{symbol}_apex_X_train.npy')
    if not x_train_path.exists():
        print(f"  ❌ Sprint 1 data not found for {symbol}")
        print(f"     Run: python apex_synth_runner_v2.py {symbol} --sprint1-only")
        return False

    n_train = np.load(f'{symbol}_apex_X_train.npy', mmap_mode='r').shape[0]
    n_val   = np.load(f'{symbol}_apex_X_val.npy',   mmap_mode='r').shape[0]
    n_test  = np.load(f'{symbol}_apex_X_test.npy',  mmap_mode='r').shape[0]
    total   = n_train + n_val + n_test
    print(f"  Split sizes → train:{n_train}  val:{n_val}  test:{n_test}  total:{total}")

    if dummy:
        print("  [DUMMY MODE] Generating synthetic FinBERT embeddings...")
        bert_embs = make_synthetic_sentiment(total, seed=hash(symbol) % 2**31)
    else:
        headlines = load_headlines(symbol)
        if headlines is None:
            print(f"  ⚠  No headlines CSV found ({symbol}_headlines.csv)")
            print(f"     Falling back to synthetic sentiment embeddings.")
            bert_embs = make_synthetic_sentiment(total, seed=hash(symbol) % 2**31)
        else:
            # Load the decision-day dates produced by apex_feature_engineering.py
            # so each headline lands on the exact trading day it belongs to.
            dates_path = Path(f'{symbol}_apex_dates_train.npy')
            if not dates_path.exists():
                print(f"  Warning: Dates file not found ({symbol}_apex_dates_train.npy)")
                print(f"     Run: python apex_synth_runner_v2.py {symbol} --sprint1-only")
                print(f"     Falling back to synthetic sentiment embeddings.")
                bert_embs = make_synthetic_sentiment(total, seed=hash(symbol) % 2**31)
            else:
                dates_train = np.load(f'{symbol}_apex_dates_train.npy')
                dates_val   = np.load(f'{symbol}_apex_dates_val.npy')
                dates_test  = np.load(f'{symbol}_apex_dates_test.npy')
                trading_dates = [
                    str(d)[:10]
                    for d in np.concatenate([dates_train, dates_val, dates_test])
                ]
                print(f"  Date-aligning headlines across {len(trading_dates)} trading days...")
                bert_embs = aggregate_daily(headlines, trading_dates, encoder)

    project_and_save(symbol, bert_embs, projector, (n_train, n_val, n_test))
    return True


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def print_summary():
    proj  = SentimentProjector()
    total = sum(p.numel() for p in proj.parameters())
    print(f"\n  FinBERT Sentiment Branch")
    print(f"  {'─'*45}")
    print(f"  Encoder       : {HF_MODEL}  (frozen)")
    print(f"  BERT dim      : {BERT_DIM}")
    print(f"  Max tokens    : {MAX_LEN}")
    print(f"  Projector     : {BERT_DIM} → 256 → {SENT_DIM}  ({total:,} params)")
    print(f"  Aux head      : {SENT_DIM} → 3 (pos/neg/neu)")
    print(f"  Output dim    : {SENT_DIM}  (feeds into Sprint 4 fusion)")
    print(f"  {'─'*45}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('symbols', nargs='*', default=SYMBOLS)
    p.add_argument('--dummy',   action='store_true',
                   help='Use synthetic embeddings (no HuggingFace needed)')
    p.add_argument('--summary', action='store_true')
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
    print("  SPRINT 3 — FINBERT SENTIMENT BRANCH")
    print(f"  Model    : {HF_MODEL}")
    print(f"  Output   : {SENT_DIM}-d sentiment embedding per day")
    print(f"  Device   : {DEVICE}")
    print("█" * 68)

    print_summary()

    if args.summary:
        return

    encoder   = FinBERTEncoder()
    projector = SentimentProjector().to(DEVICE)

    results = {}
    for sym in symbols:
        ok = process_symbol(sym, encoder, projector, dummy=args.dummy)
        results[sym] = '✅' if ok else '❌'

    print(f"\n{BANNER}")
    print("  SPRINT 3 FinBERT SUMMARY")
    print(BANNER)
    for sym, status in results.items():
        print(f"  {status}  {sym}")
    passed = sum(1 for v in results.values() if v == '✅')
    print(f"\n  {passed}/{len(symbols)} symbols processed")
    print(f"""
  Output files: {{SYMBOL}}_apex_sent_{{train|val|test}}.npy
  Shape: (n_days, {SENT_DIM})

  Next: sprint3_gat.py  (inter-stock correlation graph)
  Then: sprint4_fusion.py  (gated cross-modal fusion)
    """)
    print(BANNER)


if __name__ == '__main__':
    main()
