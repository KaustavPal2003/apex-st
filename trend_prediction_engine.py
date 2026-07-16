# trend_prediction_engine.py — APEX-ST Auxiliary Baseline
"""
Macro-Trend Baseline for APEX-ST.

Predicts the SIGN of the 30-day forward log-return using a lightweight
XGBoost classifier trained on moving-average ratios, rolling OLS slope,
and a simple RSI-style feature.

This is intentionally a separate, much simpler model than the main
Sprint 1-5 pipeline (wavelet + HMM + KPCA + GRU + FinBERT + GAT + fusion +
ensemble). It exists as a cheap baseline to sanity-check whether the
heavy pipeline is earning its complexity — consistent with this repo's
practice of reporting honest, leak-checked numbers rather than only the
best-looking ones (see README "Honest result" / "Bugs found and fixed").

Fixes applied vs. the original draft of this script:
  1. Removed the fabricated step-direction comparison (a zero-filled
     array that was never actually compared to anything, printed next to
     a claim implying it had been). Step-direction accuracy is now
     genuinely computed against a naive persistence baseline.
  2. Switched from a single hardcoded RELIANCE.NS / yfinance-only symbol
     to the repo's actual 15-symbol watchlist, with local CSV data
     preferred over live API calls, yfinance only as a fallback.
  3. Matched repo naming/output conventions: plain symbol names (no
     ".NS" suffix in filenames), auto-detected OHLCV column names,
     output file named "{SYMBOL}_trend_metrics.json" to sit alongside
     the existing "{SYMBOL}_conformal.json" / "{SYMBOL}_cusum_state.json"
     artifacts.
  4. Train/eval split is now INDEX-BASED (80% train / 20% holdout by row
     count), matching the exact convention used in the real
     apex_feature_engineering.py (`train_end_idx = int(n * 0.80)`),
     instead of hardcoded absolute calendar dates. The original date-range
     approach silently broke whenever local data didn't extend all the
     way to "today" — an index-based split works regardless of how fresh
     or stale the local CSVs are.
  5. Wrapped per-symbol execution in try/except so one bad symbol
     doesn't kill the whole run.

NOTE ON INTEGRATION: this script looks for per-symbol OHLCV at
data/{SYMBOL}.csv (referenced by the bug-fix in
apex_feature_engineering.py / the GAT correlation-graph fix) and
auto-detects the date/close column names from common variants
(timestamp/date/Date/Datetime, close/Close/adj_close/etc.). Falls back
to yfinance if the file isn't present locally. If your data folder uses
a schema outside the detected variants, or one CSV holds all symbols
instead of one-per-symbol, swap out `load_price_series()` — that's the
only function that needs to change.
"""

import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
TREND_WINDOW = 30       # Look-ahead macro window size to calculate the true trend
TRAIN_FRACTION = 0.80   # Matches apex_feature_engineering.py's default train_end_idx
MIN_ROWS_REQUIRED = 150  # Minimum usable (post-dropna) rows to bother training/evaluating

# Matches the README watchlist exactly (plain NSE symbols, no suffix)
WATCHLIST = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "BAJAJ-AUTO", "BAJFINANCE",
    "CIPLA", "DIVISLAB", "EICHERMOT", "GRASIM", "INDUSINDBK",
    "JSWSTEEL", "NESTLEIND", "SUNPHARMA", "TATASTEEL", "TITAN",
]

DATA_DIR = os.getenv("APEX_DATA_DIR", "data")  # per-symbol CSVs, if present
# Only used for the yfinance fallback path (no local file found).
FALLBACK_FETCH_START = "2015-01-01"


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
# Column-name variants seen across different local data conventions.
# The first match found in each list wins.
_DATE_COL_CANDIDATES = ["timestamp", "Timestamp", "date", "Date", "Datetime", "datetime"]
_CLOSE_COL_CANDIDATES = ["close", "Close", "CLOSE", "close_price", "adj_close", "Adj Close"]


def _detect_column(df: pd.DataFrame, candidates: list, kind: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find a {kind} column in local CSV. "
        f"Columns present: {list(df.columns)}. "
        f"Expected one of: {candidates}"
    )


def load_price_series(symbol: str) -> pd.Series:
    """
    Load the FULL available close-price history for `symbol` (no date
    filtering — the caller decides how to split train/eval by row index,
    so this works regardless of how far the local data actually extends).

    Prefers a local data/{symbol}.csv, auto-detecting the date column and
    close-price column from common naming variants. Falls back to
    yfinance with a ".NS" suffix (fetching from FALLBACK_FETCH_START up
    to today) if no local file is found.
    """
    local_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if os.path.exists(local_path):
        raw = pd.read_csv(local_path)
        date_col = _detect_column(raw, _DATE_COL_CANDIDATES, "date")
        close_col = _detect_column(raw, _CLOSE_COL_CANDIDATES, "close")

        df = raw[[date_col, close_col]].copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.dropna(subset=[date_col, close_col])
        df = df.drop_duplicates(subset=[date_col]).set_index(date_col).sort_index()
        if df.empty:
            raise ValueError(f"Local CSV for {symbol} has no usable rows.")
        return df[close_col].squeeze()

    # Fallback: live fetch via yfinance
    import yfinance as yf
    yf_symbol = f"{symbol}.NS"
    df = yf.download(yf_symbol, start=FALLBACK_FETCH_START, progress=False)
    if df.empty:
        raise ValueError(f"Failed to retrieve data for {yf_symbol} (no local CSV either).")
    return df["Close"].squeeze()


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def calculate_rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Calculates the structural trend using a moving linear regression slope.
    This strips out day-to-day noise and estimates overall momentum.
    """
    slopes = [np.nan] * (window - 1)
    x = np.arange(window)
    x_var = np.var(x)

    for i in range(window - 1, len(series)):
        y = series.iloc[i - window + 1: i + 1].values
        slope = np.cov(x, y)[0, 1] / x_var
        slopes.append(slope)

    return pd.Series(slopes, index=series.index)


def build_features(close_prices: pd.Series) -> pd.DataFrame:
    """Shared feature builder used identically at train and eval time."""
    features = pd.DataFrame(index=close_prices.index)
    features["ma_ratio_10_50"] = close_prices.rolling(10).mean() / close_prices.rolling(50).mean()
    features["ma_ratio_30_100"] = close_prices.rolling(30).mean() / close_prices.rolling(100).mean()
    features["rolling_slope_20"] = calculate_rolling_slope(close_prices, window=20)
    features["rsi_sim"] = (
        (close_prices - close_prices.rolling(14).min())
        / (close_prices.rolling(14).max() - close_prices.rolling(14).min() + 1e-8)
    )
    return features


def build_trend_target(close_prices: pd.Series) -> pd.Series:
    """
    1 if the OLS slope over the NEXT TREND_WINDOW days is positive, else 0.
    shift(-TREND_WINDOW) aligns today's label to the future window, so no
    current-or-past information leaks into the label (mirrors the y_reg
    fix documented in the main pipeline's README).
    """
    future_slopes = calculate_rolling_slope(close_prices, window=TREND_WINDOW).shift(-TREND_WINDOW)
    return (future_slopes > 0).astype(int)


def prepare_symbol_dataset(symbol: str) -> pd.DataFrame:
    """
    Loads full history, builds features + target + close price, and drops
    boundary NaN rows (from rolling windows and the forward-looking
    target). Returns a single time-ordered DataFrame ready to split by
    row index — features and target are computed over the FULL series
    before any split, matching how apex_feature_engineering.py computes
    y_reg/y_cls over the full series before its WalkForwardSplitter cuts
    it into folds.
    """
    close_prices = load_price_series(symbol)
    features = build_features(close_prices)
    target = build_trend_target(close_prices)

    dataset = features.copy()
    dataset["target"] = target
    dataset["close"] = close_prices
    dataset = dataset.dropna()
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# 1. TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_trend_model(symbol: str, train_df: pd.DataFrame) -> xgb.XGBClassifier:
    print(f"=== [1/3] Training Trend Engine on Historical Data for {symbol} ===")

    X = train_df.drop(columns=["target", "close"]).values
    y = train_df["target"].values

    trend_xgb = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.03,
        random_state=42,
        eval_metric="logloss",
    )
    trend_xgb.fit(X, y)

    print(f" ✓ Successfully trained trend model on {len(y)} historical days "
          f"({train_df.index[0].strftime('%Y-%m-%d')} to {train_df.index[-1].strftime('%Y-%m-%d')}).")
    return trend_xgb


# ─────────────────────────────────────────────────────────────────────────────
# 2. HOLDOUT EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_trend_model(symbol: str, eval_df: pd.DataFrame, trained_model: xgb.XGBClassifier) -> dict | None:
    print(f"\n=== [2/3] Evaluating on Holdout Window for {symbol} ===")

    if len(eval_df) == 0:
        print(" ❌ Holdout split is empty.")
        return None

    X_eval = eval_df.drop(columns=["target", "close"]).values
    y_eval_true = eval_df["target"].values
    eval_preds = trained_model.predict(X_eval)

    # ─────────────────────────────────────────────────────────────────────
    # 3. PERFORMANCE ASSESSMENT
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n=== [3/3] Performance Assessment for {symbol} ===")

    trend_accuracy = float((eval_preds == y_eval_true).mean() * 100)

    # Genuine naive-persistence baseline for step-level direction (previously
    # this was a zero-filled array never actually compared to anything).
    # Persistence predicts "up" if yesterday's close > the day before,
    # "down" otherwise — the standard no-skill benchmark.
    eval_closes = eval_df["close"]
    actual_step_dir = (eval_closes.diff().fillna(0) > 0).astype(int).values
    pred_step_dir = (eval_closes.shift(1).diff().fillna(0) > 0).astype(int).values
    step_accuracy = float((pred_step_dir == actual_step_dir).mean() * 100)

    idx = eval_df.index
    print(f"  Holdout Window             : {idx[0].strftime('%Y-%m-%d')} to {idx[-1].strftime('%Y-%m-%d')}")
    print(f"  Evaluated Data Points      : {len(y_eval_true)} active trading days")
    print(f"  ---------------------------------------------------------------")
    print(f"  TREND PREDICTION ACCURACY  : {trend_accuracy:.2f}%  (30-day macro regime, XGBoost)")
    print(f"  STEP-DIRECTION BASELINE    : {step_accuracy:.2f}%  (1-day persistence, no-skill benchmark)")
    print(f"  ---------------------------------------------------------------")
    print("  Note: these two metrics answer different questions and are not")
    print("        directly comparable in difficulty — the persistence baseline")
    print("        is shown only as a sanity-check reference point, not proof")
    print("        the trend model outperforms it.")

    metrics_summary = {
        "symbol": symbol,
        "evaluation_timestamp": datetime.now().isoformat(),
        "evaluated_days": int(len(y_eval_true)),
        "trend_prediction_accuracy": round(trend_accuracy, 2),
        "step_persistence_baseline_accuracy": round(step_accuracy, 2),
        "window_start": idx[0].strftime("%Y-%m-%d"),
        "window_end": idx[-1].strftime("%Y-%m-%d"),
    }

    out_path = f"{symbol}_trend_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f" ✓ Saved → {out_path}")

    return metrics_summary


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — runs across the full watchlist, matching the rest of the pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main():
    results = []
    for symbol in WATCHLIST:
        print("\n" + "=" * 68)
        print(f" TREND ENGINE — {symbol}")
        print("=" * 68)
        try:
            dataset = prepare_symbol_dataset(symbol)
            if len(dataset) < MIN_ROWS_REQUIRED:
                print(f" ❌ Only {len(dataset)} usable rows after feature/target "
                      f"warm-up (need ≥{MIN_ROWS_REQUIRED}). Skipping.")
                continue

            train_end_idx = int(len(dataset) * TRAIN_FRACTION)
            train_df = dataset.iloc[:train_end_idx]
            eval_df = dataset.iloc[train_end_idx:]

            model = train_trend_model(symbol, train_df)
            summary = evaluate_trend_model(symbol, eval_df, model)
            if summary:
                results.append(summary)
        except Exception as e:
            print(f" ❌ {symbol} failed: {e}")
            continue

    if not results:
        print("\n ❌ No symbols produced a valid evaluation.")
        return

    mean_trend_acc = float(np.mean([r["trend_prediction_accuracy"] for r in results]))
    mean_step_acc = float(np.mean([r["step_persistence_baseline_accuracy"] for r in results]))

    print("\n" + "=" * 68)
    print(" SUMMARY ACROSS WATCHLIST")
    print("=" * 68)
    print(f"  Symbols evaluated          : {len(results)}/{len(WATCHLIST)}")
    print(f"  Mean trend accuracy        : {mean_trend_acc:.2f}%")
    print(f"  Mean step-persistence acc. : {mean_step_acc:.2f}%")
    print("  (Compare mean_trend_acc to 50% — same honesty standard as the")
    print("   main pipeline's directional-accuracy result in the README.)")

    with open("trend_engine_summary.json", "w") as f:
        json.dump(
            {
                "generated": datetime.now().isoformat(),
                "n_symbols": len(results),
                "mean_trend_prediction_accuracy": round(mean_trend_acc, 2),
                "mean_step_persistence_baseline_accuracy": round(mean_step_acc, 2),
                "per_symbol": results,
            },
            f,
            indent=2,
        )
    print(" ✓ Saved → trend_engine_summary.json")


if __name__ == "__main__":
    main()
