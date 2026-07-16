"""
trend_analysis.py — Sprint 7: Trend & Regime Analysis
═══════════════════════════════════════════════════════════════════════
Extracts three signals from EXISTING artefacts — no retraining needed:

  1. Slope Correlation
     Rolling 20-day linear regression slope of predicted vs actual
     30-day log-returns. Measures whether the model's predicted trend
     direction matches the actual trend direction over time.

  2. Regime Classification Matrix
     Maps each HMM regime (0-3) to Bullish / Bearish / Sideways / Volatile
     based on the regime's mean log-return and standard deviation.
     Reports the CURRENT regime for each symbol (last test-set day).

  3. Trend Prediction Accuracy
     Day-level direction accuracy: % of test-set days where the sign of
     the predicted 30-day log-return matches the sign of the actual
     30-day log-return. Simpler/coarser than (1) — no rolling window,
     just a straight per-day sign comparison.

Run:
    python trend_analysis.py

Output:
    results/slope_correlation.csv
    results/regime_classification.csv
    results/trend_prediction_accuracy.csv
    results/{SYM}_trend_prediction_accuracy.json
    Console summary table

Requires:
    {SYM}_ensemble_result.json   (Sprint 5 output)
    {SYM}_apex_pipeline.pkl      (Sprint 1 pipeline — contains fitted HMM)
    {SYM}_apex_X_test.npy        (test-split feature sequences)
    watchlist.json
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from scipy import stats

import __main__
from apex_synth_runner_v2 import RawCaptureKPCA, NystroemReducer
__main__.RawCaptureKPCA = RawCaptureKPCA
__main__.NystroemReducer = NystroemReducer

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
SLOPE_WINDOW  = 20   # rolling window for slope computation (trading days)
RESULTS_DIR   = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def load_watchlist():
    p = Path('watchlist.json')
    if p.exists():
        return json.loads(p.read_text()).get('watchlist', [])
    summary = json.loads(Path('sprint5_summary.json').read_text())
    return [s for s in summary if Path(f'{s}_ensemble_result.json').exists()]


def rolling_slope(series: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling linear regression slope over `window` days.
    Returns an array of the same length; first (window-1) values are NaN.
    """
    slopes = np.full(len(series), np.nan)
    x = np.arange(window, dtype=float)
    for i in range(window - 1, len(series)):
        y = series[i - window + 1: i + 1]
        slope, _, _, _, _ = stats.linregress(x, y)
        slopes[i] = slope
    return slopes


def classify_regime(mean_ret: float, std_ret: float) -> str:
    """
    Map regime statistics to a human-readable market state.

    Rules (calibrated to daily log-returns scaled to 30-day horizon):
      mean_ret > +0.002  AND std_ret < 0.025  → Bullish
      mean_ret < -0.002  AND std_ret < 0.025  → Bearish
      std_ret  > 0.030                         → Volatile (high uncertainty)
      else                                     → Sideways
    """
    if mean_ret > 0.002 and std_ret < 0.025:
        return 'Bullish 🟢'
    elif mean_ret < -0.002 and std_ret < 0.025:
        return 'Bearish 🔴'
    elif std_ret > 0.030:
        return 'Volatile ⚡'
    else:
        return 'Sideways 🟡'


# ─────────────────────────────────────────────────────────────────────
# 1. SLOPE CORRELATION
# ─────────────────────────────────────────────────────────────────────

def compute_slope_correlation(sym: str) -> dict | None:
    """
    Load actual and predicted 30-day log-returns from ensemble_result.json,
    compute rolling 20-day slopes for both, then measure:
      - Pearson correlation between predicted and actual slopes
      - % of days where both slopes have the same sign (direction agreement)
      - Overall trend direction accuracy (slope sign match rate)
    """
    p = Path(f'{sym}_ensemble_result.json')
    if not p.exists():
        return None

    er       = json.loads(p.read_text())
    y_actual = np.array(er.get('y_reg_test', []))
    y_pred   = np.array(er.get('reg_pred_test', []))

    if len(y_actual) < SLOPE_WINDOW + 5:
        return None

    slope_actual = rolling_slope(y_actual, SLOPE_WINDOW)
    slope_pred   = rolling_slope(y_pred,   SLOPE_WINDOW)

    # Only use rows where both slopes are valid (non-NaN)
    mask = ~np.isnan(slope_actual) & ~np.isnan(slope_pred)
    sa, sp = slope_actual[mask], slope_pred[mask]

    if len(sa) < 5:
        return None

    corr, pval         = stats.pearsonr(sa, sp)
    direction_agree    = np.mean(np.sign(sa) == np.sign(sp)) * 100
    n_valid            = int(mask.sum())

    # Overall trend: is the last 20-day slope positive?
    current_actual_trend  = 'UP ↑' if slope_actual[mask][-1] > 0 else 'DOWN ↓'
    current_pred_trend    = 'UP ↑' if slope_pred[mask][-1]   > 0 else 'DOWN ↓'
    trend_match           = '✅' if current_actual_trend == current_pred_trend else '❌'

    return {
        'symbol':              sym,
        'slope_correlation':   round(float(corr), 4),
        'slope_corr_pvalue':   round(float(pval), 4),
        'direction_agree_pct': round(float(direction_agree), 1),
        'n_valid_windows':     n_valid,
        'current_actual_trend': current_actual_trend,
        'current_pred_trend':   current_pred_trend,
        'trend_match':          trend_match,
    }


# ─────────────────────────────────────────────────────────────────────
# 2. REGIME CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────

DATA_DIR = Path('data')   # raw OHLCV, written by fetch_nse_data.py — data/<SYMBOL>.csv


def compute_regime_classification(sym: str) -> dict | None:
    """
    Rebuild the HMM observation matrix from RAW prices (log return, 5-day
    and 20-day rolling vol) — the same inputs RegimeDetector was fitted on —
    then decode regimes and subset to the test-period dates.

    NOTE: {SYM}_apex_X_test.npy is NOT used here. It holds post-KPCA/Nystroem
    reduced features, which are the wrong shape/scale for RegimeDetector's
    fitted HMM + scaler. Regime decoding must happen on raw price-derived
    observations, matched back to the test dates via {SYM}_apex_dates_test.npy.
    """
    pkl_path   = Path(f'{sym}_apex_pipeline.pkl')
    csv_path   = DATA_DIR / f'{sym}.csv'
    dates_path = Path(f'{sym}_apex_dates_test.npy')
    er_path    = Path(f'{sym}_ensemble_result.json')

    if not pkl_path.exists() or not csv_path.exists() or not dates_path.exists():
        return None

    try:
        with open(pkl_path, 'rb') as f:
            pipeline = pickle.load(f)
    except Exception as e:
        print(f"  ⚠  {sym}: could not load pipeline — {e}")
        return None

    regime_detector = getattr(pipeline, 'regime', None)
    if regime_detector is None and isinstance(pipeline, dict):
        regime_detector = pipeline.get('regime')
    if regime_detector is None or regime_detector.model is None:
        print(f"  ⚠  {sym}: RegimeDetector/HMM not found in pipeline artefact")
        return None

    # ── Load raw OHLCV, full history (needed for correct rolling-window stats) ──
    df_raw = pd.read_csv(csv_path)
    df_raw['timestamp'] = pd.to_datetime(df_raw['date'])
    df_raw = df_raw.sort_values('timestamp').reset_index(drop=True)

    # ── Load test-period decision dates ──────────────────────────────────────
    test_dates = np.load(str(dates_path), allow_pickle=True)
    test_dates = pd.to_datetime(pd.Series(test_dates))

    try:
        obs      = regime_detector._build_obs(df_raw)
        obs_sc   = regime_detector._obs_scaler.transform(obs)
        labels_full = regime_detector.model.predict(obs_sc)
    except Exception as e:
        print(f"  ⚠  {sym}: HMM predict failed — {e}")
        return None

    df_raw['_regime'] = labels_full
    mask_test = df_raw['timestamp'].isin(test_dates)
    df_test   = df_raw[mask_test]

    if df_test.empty:
        print(f"  ⚠  {sym}: no raw rows matched test dates")
        return None

    regimes = df_test['_regime'].values

    # Get actual returns for regime statistics
    if er_path.exists():
        er = json.loads(er_path.read_text())
        y_actual = np.array(er.get('y_reg_test', []))
    else:
        y_actual = np.zeros(len(regimes))

    n_states    = regime_detector.n_states
    regime_rows = []
    for state in range(n_states):
        smask = regimes == state
        if smask.sum() == 0:
            continue
        mean_r = float(y_actual[smask].mean()) if len(y_actual) == len(regimes) else 0.0
        std_r  = float(y_actual[smask].std())  if len(y_actual) == len(regimes) else 0.0
        freq   = float(smask.mean()) * 100
        label  = classify_regime(mean_r, std_r)
        regime_rows.append({
            'state':      state,
            'label':      label,
            'mean_ret':   round(mean_r, 5),
            'std_ret':    round(std_r,  5),
            'freq_pct':   round(freq,   1),
            'is_current': bool(regimes[-1] == state),
        })

    current_state = int(regimes[-1])
    current_label = next(
        (r['label'] for r in regime_rows if r['state'] == current_state),
        'Unknown'
    )

    return {
        'symbol':        sym,
        'current_state': current_state,
        'current_label': current_label,
        'regime_map':    regime_rows,
        'regime_series': regimes.tolist(),
    }

# ─────────────────────────────────────────────────────────────────────
# 3. TREND PREDICTION ACCURACY
# ─────────────────────────────────────────────────────────────────────

def compute_trend_prediction_accuracy(sym: str) -> dict | None:
    """
    Day-level trend/direction accuracy over the test set: for each day,
    compare sign(predicted 30-day log-return) to sign(actual 30-day
    log-return). Unlike compute_slope_correlation(), this operates on
    the raw daily values directly — no rolling window, no smoothing.

    Returns e.g.:
        {
          "symbol": "TCS",
          "evaluation_timestamp": "2026-07-16T09:41:03Z",
          "evaluated_days": 120,
          "trend_prediction_accuracy": 81.7
        }
    """
    p = Path(f'{sym}_ensemble_result.json')
    if not p.exists():
        return None

    er       = json.loads(p.read_text())
    y_actual = np.array(er.get('y_reg_test', []))
    y_pred   = np.array(er.get('reg_pred_test', []))

    if len(y_actual) == 0 or len(y_actual) != len(y_pred):
        return None

    actual_dir = np.sign(y_actual)
    pred_dir   = np.sign(y_pred)
    matches    = actual_dir == pred_dir
    accuracy   = float(matches.mean()) * 100

    return {
        'symbol':                   sym,
        'evaluation_timestamp':     datetime.now(timezone.utc)
                                        .isoformat(timespec='seconds')
                                        .replace('+00:00', 'Z'),
        'evaluated_days':           int(len(y_actual)),
        'trend_prediction_accuracy': round(accuracy, 1),
    }


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    symbols  = load_watchlist()
    sc_rows  = []
    reg_rows = []
    tpa_rows = []

    print("\n" + "═" * 70)
    print("  APEX-ST — Trend & Regime Analysis")
    print("═" * 70)

    # ── 1. Slope Correlation ──────────────────────────────────────────
    print(f"\n{'Symbol':12} {'Slope Corr':>11} {'p-val':>8} "
          f"{'Dir Agree%':>11} {'Actual':>7} {'Pred':>7} {'Match':>6}")
    print("-" * 68)

    for sym in symbols:
        r = compute_slope_correlation(sym)
        if r is None:
            print(f"{sym:12} — skipped")
            continue
        sc_rows.append(r)
        print(f"{sym:12} {r['slope_correlation']:>+11.4f} "
              f"{r['slope_corr_pvalue']:>8.4f} "
              f"{r['direction_agree_pct']:>10.1f}% "
              f"{r['current_actual_trend']:>8} "
              f"{r['current_pred_trend']:>8} "
              f"{r['trend_match']:>6}")

    if sc_rows:
        avg_corr   = np.mean([r['slope_correlation']   for r in sc_rows])
        avg_agree  = np.mean([r['direction_agree_pct'] for r in sc_rows])
        n_match    = sum(1 for r in sc_rows if r['trend_match'] == '✅')
        print("-" * 68)
        print(f"{'Mean':12} {avg_corr:>+11.4f} {'':>8} "
              f"{avg_agree:>10.1f}%  Current trend match: {n_match}/{len(sc_rows)}")

    # ── 2. Regime Classification ──────────────────────────────────────
    print(f"\n\n{'Symbol':12} {'Current Regime':>20} {'State':>6} "
          f"{'Mean Ret':>9} {'Std Ret':>8} {'Frequency':>10}")
    print("-" * 70)

    for sym in symbols:
        r = compute_regime_classification(sym)
        if r is None:
            print(f"{sym:12} — pipeline artefact missing")
            continue

        current = next(
            (rm for rm in r['regime_map'] if rm['is_current']),
            None
        )
        if current is None:
            print(f"{sym:12} — regime decode failed")
            continue

        reg_rows.append({
            'symbol':        sym,
            'current_label': r['current_label'],
            'current_state': r['current_state'],
            'mean_ret':      current['mean_ret'],
            'std_ret':       current['std_ret'],
            'freq_pct':      current['freq_pct'],
        })
        print(f"{sym:12} {r['current_label']:>20} "
              f"{r['current_state']:>6}  "
              f"{current['mean_ret']:>+9.5f} "
              f"{current['std_ret']:>8.5f} "
              f"{current['freq_pct']:>9.1f}%")

    # ── Regime summary ────────────────────────────────────────────────
    if reg_rows:
        labels     = [r['current_label'] for r in reg_rows]
        from collections import Counter
        counts     = Counter(labels)
        print("\n  Regime distribution across watchlist:")
        for label, cnt in counts.most_common():
            bar = '█' * cnt
            print(f"    {label:20} {bar} ({cnt} symbols)")

    # ── 3. Trend Prediction Accuracy ────────────────────────────────────
    print(f"\n\n{'Symbol':12} {'Evaluated Days':>15} {'Trend Pred Acc':>16}")
    print("-" * 46)

    for sym in symbols:
        r = compute_trend_prediction_accuracy(sym)
        if r is None:
            print(f"{sym:12} — skipped")
            continue
        tpa_rows.append(r)
        print(f"{sym:12} {r['evaluated_days']:>15} "
              f"{r['trend_prediction_accuracy']:>15.1f}%")

        # Per-symbol JSON, matches the requested example shape exactly
        with open(RESULTS_DIR / f"{sym}_trend_prediction_accuracy.json", 'w') as f:
            json.dump(r, f, indent=2)

    if tpa_rows:
        avg_acc = np.mean([r['trend_prediction_accuracy'] for r in tpa_rows])
        print("-" * 46)
        print(f"{'Mean':12} {'':>15} {avg_acc:>15.1f}%")

    # ── Save CSVs ─────────────────────────────────────────────────────
    if sc_rows:
        pd.DataFrame(sc_rows).to_csv(
            RESULTS_DIR / 'slope_correlation.csv', index=False
        )
        print(f"\n  ✓ Saved results/slope_correlation.csv")

    if reg_rows:
        pd.DataFrame(reg_rows).to_csv(
            RESULTS_DIR / 'regime_classification.csv', index=False
        )
        print(f"  ✓ Saved results/regime_classification.csv")

    if tpa_rows:
        pd.DataFrame(tpa_rows).to_csv(
            RESULTS_DIR / 'trend_prediction_accuracy.csv', index=False
        )
        print(f"  ✓ Saved results/trend_prediction_accuracy.csv")
        print(f"  ✓ Saved results/{{SYMBOL}}_trend_prediction_accuracy.json "
              f"({len(tpa_rows)} files)")

    print("\n" + "═" * 70 + "\n")


if __name__ == '__main__':
    main()
