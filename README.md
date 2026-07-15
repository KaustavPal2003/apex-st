# APEX-ST — NSE 30-Day Stock Prediction Pipeline

A multi-stage machine learning pipeline for directional and return prediction on NSE large/mid-cap stocks, built as a research project with academic honesty at its core.

## Honest result

**Mean directional accuracy: 50.03%** — statistically indistinguishable from a coin flip across 15 symbols and multiple independent runs. This is the correct, leak-free result after identifying and fixing four data leakage bugs during development. A pipeline that tells the truth is more valuable than one that inflates results.

---

## Architecture

```
Data fetch → Screen → Feature engineering → Price GRU (Sprint 1+2)
                   → FinBERT sentiment     (Sprint 3a)
                   → Graph Attention Net   (Sprint 3b)
                          ↓
                   Gated fusion            (Sprint 4)
                          ↓
                   XGBoost ensemble        (Sprint 5)
                   + Conformal calibration
                   + CUSUM drift detection
                          ↓
                   FastAPI inference API   (Sprint 6)
                   + Streamlit dashboard
```

## Pipeline stages

| Script | Stage | Description |
|---|---|---|
| `fetch_nse_data.py` | Data | Downloads 15+ years OHLCV |
| `bhavcopy_to_apex_multi.py` | Data | Alternative data path: merges yearly NSE bhavcopy CSVs into per-symbol `data/<SYMBOL>.csv` files |
| `stock_screener.py` | Screen | Relative strength vs Nifty 50; optional fundamentals gate |
| `apex_feature_engineering.py` | Sprint 1 | Wavelet decomp, HMM regime, 60+ indicators, Nystroem |
| `apex_synth_runner_v2.py` | Sprint 2 | Conv1D + BiGRU + Attention; walk-forward CV |
| `sprint3_finbert.py` | Sprint 3a | ProsusAI/finbert; date-aligned via `aggregate_daily()` |
| `sprint3_gat.py` | Sprint 3b | 8-head GAT on log-return correlation graph |
| `sprint4_fusion.py` | Sprint 4 | Gated / Concat / CrossAttention fusion variants |
| `sprint5_ensemble.py` | Sprint 5 | XGBoost stacking; split conformal prediction; CUSUM |
| `apex_inference.py` | Sprint 6 | FastAPI REST service; 6 endpoints |
| `dashboard.py` | Sprint 6 | Streamlit dashboard; live/offline dual mode |
| `sentinel_news_fetcher.py` | Data | GDELT DOC 2.0 headline fetcher |
| `live_edge_pipeline.py` | Orchestrator | Runs every stage above via subprocess, in order, under the "Live Edge" presentation naming — see [Live Edge orchestrator](#live-edge-orchestrator) below |

## Bugs found and fixed

Four data leakage bugs were identified and fixed during development:

1. **GAT correlation graph** — built from raw future price levels instead of daily log-returns, inflating correlations via bull-market drift. Fixed by reading from `data/{SYM}.csv` directly.
2. **`y_reg` target** — stored raw future closing price instead of 30-day log-return. Fixed by computing `log(future/curr)` in `_create_sequences()`.
3. **Sentiment date alignment** — headlines scrambled to wrong trading days via even-chunking. Fixed by wiring `aggregate_daily()` with `{SYM}_apex_dates_*.npy` arrays.
4. **Fundamentals screener** — `numpy.bool_ is not False` identity check silently passed every symbol. Fixed by wrapping in `bool()`.

## Key findings

- Price branch dominates fusion gate (mean 73% weight)
- Sentiment branch gets 16% weight despite only 5.6% average headline density
- 14/15 symbols well-calibrated at 90% conformal coverage
- INDUSINDBK persistently fails calibration (81.5% empirical vs 90% nominal) across 5+ independent runs
- Seed stability test confirms per-symbol accuracy swings of ±2-3 points are noise, not signal

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch data and screen
python fetch_nse_data.py --start 2010-01-01 --screen --skip-fundamentals

# 3. Run full pipeline
python apex_synth_runner_v2.py --epochs 18
python sprint3_finbert.py
python sprint3_gat.py
python sprint4_fusion.py
python sprint5_ensemble.py

# 4. Start inference service
uvicorn apex_inference:app --host 0.0.0.0 --port 8000

# 5. Start dashboard
streamlit run dashboard.py
```

## Alternative data ingestion: NSE bhavcopy

`fetch_nse_data.py` pulls history from Yahoo Finance, but the repo also ships
15 years of official NSE bhavcopy exports (`NSE_OHLCV_2010.csv` … `NSE_OHLCV_2025.csv`,
one file per year, all symbols). `bhavcopy_to_apex_multi.py` merges these into
the same per-symbol `data/<SYMBOL>.csv` layout that `apex_synth_runner_v2.py`
expects, as an alternative to the Yahoo Finance fetch:

```bash
# merge every yearly bhavcopy file in the current folder
python bhavcopy_to_apex_multi.py --glob "NSE_OHLCV_*.csv" --out-dir data

# or restrict to the watchlist explicitly
python bhavcopy_to_apex_multi.py NSE_OHLCV_2010.csv NSE_OHLCV_2011.csv \
    --symbols ADANIENT ADANIPORTS APOLLOHOSP --out-dir data
```

Tickers that were renamed on NSE mid-history (e.g. Adani Ports traded as
`MUNDRAPORT` before its 2011–12 rename) need both names searched or rows
after the rename get silently dropped. Use `CURRENT:HISTORICAL` to handle
this — the script looks up both names in the source CSVs and writes a
single continuous file under the current name:

```bash
python bhavcopy_to_apex_multi.py --glob "NSE_OHLCV_*.csv" \
    --symbols ADANIENT ADANIPORTS:MUNDRAPORT APOLLOHOSP --out-dir data
# -> data/ADANIPORTS.csv, populated from MUNDRAPORT rows pre-rename and
#    ADANIPORTS rows post-rename
```

Duplicate dates within a symbol are dropped (keeping the last row) and the
script prints row counts and date ranges per symbol so gaps are visible
before training starts.

## Live Edge orchestrator

`live_edge_pipeline.py` is a re-skin, not a rewrite: it calls the same
unmodified scripts listed in [Pipeline stages](#pipeline-stages), in the same
order, via subprocess, and only relabels the stages for presentation (e.g.
"Execution Pruning Engine + Circuit Breaker" = the existing conformal +
CUSUM logic already inside `sprint5_ensemble.py`). Nothing about the data,
features, model, or evaluation changes. The full stage-name mapping and
rationale is documented in [`LIVE_EDGE_ARCHITECTURE.md`](LIVE_EDGE_ARCHITECTURE.md).

Run the whole pipeline end-to-end with one command:

```bash
python live_edge_pipeline.py
```

At the end it reports the measured mean directional accuracy from
`sprint5_summary.json` if that file exists from a real run, falling back to
the documented 50.03% figure from [Honest result](#honest-result) otherwise
— it never prints a placeholder number. It then prints the same two
commands to start the live service:

```bash
uvicorn apex_inference:app --host 0.0.0.0 --port 8000
streamlit run dashboard.py
```

## Requirements

```
Python 3.11+
torch
fastapi
uvicorn[standard]
streamlit
xgboost==2.1.3
numpy
pandas
requests
yfinance
scikit-learn
transformers
scipy
pywavelets
hmmlearn
feedparser
```

## Deployment

- **Local**: Windows Task Scheduler auto-starts both services on boot
- **Cloud**: AWS EC2 t3.micro, Mumbai region (ap-south-1)
- **Live dashboard**: `http://13.233.140.171:8501`
- **Inference API**: `http://13.233.140.171:8000`

## Weekly refresh

Models are retrained every Monday after NSE market close (11 PM IST):
```bash
python fetch_nse_data.py --start 2010-01-01 --screen --skip-fundamentals
python apex_synth_runner_v2.py --epochs 18
python sprint3_finbert.py
python sprint3_gat.py
python sprint4_fusion.py
python sprint5_ensemble.py
```

## Academic documentation

LaTeX tables for all results are available in `apex_st_tables.tex`, including:
- Per-symbol prediction performance (Table 1)
- Conformal calibration results (Table 2)
- Data leakage bugs (Table 3)
- Seed stability analysis (Table 4)
- News coverage statistics (Table 5)
- Architecture summary (Table 6)
- Fusion gate weights (Table 7)

## Watchlist (15 symbols)

ADANIENT, ADANIPORTS, APOLLOHOSP, BAJAJ-AUTO, BAJFINANCE, CIPLA, DIVISLAB, EICHERMOT, GRASIM, INDUSINDBK, JSWSTEEL, NESTLEIND, SUNPHARMA, TATASTEEL, TITAN

## Limitations

- 30-day directional accuracy at random-baseline level with current data sources
- GDELT news coverage averages 5.6% density — insufficient for robust sentiment signal
- Inference API serves frozen last-test-row predictions, not live forward-looking forecasts
- Wavelet decomposition applied to full series before splitting (minor boundary effect, documented)
- No GPU available during development — all training on AMD Ryzen 5 4600G CPU

## License

MIT License — see LICENSE file.
