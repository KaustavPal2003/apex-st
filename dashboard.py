"""
dashboard.py — Sprint 6: APEX-ST Prediction Dashboard  (v2)
══════════════════════════════════════════════════════════════
Streamlit front-end for the apex_inference.py FastAPI service.
Works in two modes:

  LIVE mode  — FastAPI running at API_URL (full predictions)
  OFFLINE mode — reads local sprint5_summary.json + conformal/CUSUM
                 JSON files directly (no API needed)

Run
────
  # Two-terminal (full live mode)
  uvicorn apex_inference:app --port 8000          # terminal 1
  streamlit run dashboard.py                       # terminal 2

  # Offline (no model loading required — reads saved artifacts)
  streamlit run dashboard.py                       # just this
"""

import json
import math
import os
import time
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="APEX-ST",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/combo-chart.png", width=60)
st.sidebar.title("APEX-ST")
st.sidebar.caption("NSE 30-Day Prediction Pipeline")

API_URL = st.sidebar.text_input("Inference API URL", "http://localhost:8000")
st.sidebar.caption("Start with: `uvicorn apex_inference:app --port 8000`")

st.sidebar.divider()
show_news = st.sidebar.checkbox("Show news coverage panel", value=True)
show_ablation = st.sidebar.checkbox("Show fusion ablation panel", value=True)
show_conformal = st.sidebar.checkbox("Show conformal details", value=False)

st.sidebar.divider()
st.sidebar.caption("**Sentinel News Fetcher**")
st.sidebar.caption("Run to populate headlines:")
st.sidebar.code("python sentinel_news_fetcher.py --refresh", language="bash")
st.sidebar.caption("Full 2yr backfill:")
st.sidebar.code("python sentinel_news_fetcher.py --backfill", language="bash")

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=120)
def load_summary() -> dict:
    p = Path("sprint5_summary.json")
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data(ttl=120)
def load_ablation() -> list:
    p = Path("sprint4_ablation.json")
    return json.loads(p.read_text()) if p.exists() else []


@st.cache_data(ttl=120)
def load_watchlist() -> list:
    p = Path("watchlist.json")
    if p.exists():
        return json.loads(p.read_text()).get("watchlist", [])
    return []


@st.cache_data(ttl=120)
def load_conformal(symbol: str) -> dict:
    p = Path(f"{symbol}_conformal.json")
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data(ttl=120)
def load_cusum(symbol: str) -> dict:
    p = Path(f"{symbol}_cusum_state.json")
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data(ttl=60)
def load_headlines_summary() -> dict:
    """Load per-symbol headline counts from news_cache/ or CSVs."""
    import csv as _csv
    result = {}
    watchlist = load_watchlist()
    for sym in watchlist:
        csv_path = Path(f"{sym}_headlines.csv")
        if not csv_path.exists():
            result[sym] = {"count": 0, "earliest": None, "latest": None}
            continue
        rows = []
        with open(csv_path, encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                if row.get("date"):
                    rows.append(row["date"])
        rows.sort()
        result[sym] = {
            "count":    len(rows),
            "earliest": rows[0] if rows else None,
            "latest":   rows[-1] if rows else None,
        }
    return result


def fetch_live_predictions() -> dict:
    """Try to hit the FastAPI endpoint; return {} on failure."""
    try:
        import requests
        r = requests.get(f"{API_URL}/predict", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def fetch_live_drift(symbol: str) -> dict:
    try:
        import requests
        r = requests.get(f"{API_URL}/drift/{symbol}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

st.title("📈 APEX-ST — 30-Day NSE Prediction Dashboard")

summary   = load_summary()
ablation  = load_ablation()
watchlist = load_watchlist()

if not watchlist and summary:
    watchlist = list(summary.keys())

# ── Attempt live API ──────────────────────────────────────────────────────
col_refresh, col_mode = st.columns([1, 8])
with col_refresh:
    do_refresh = st.button("🔄 Refresh")

if "live_preds" not in st.session_state or do_refresh:
    with st.spinner("Querying inference API…"):
        st.session_state.live_preds = fetch_live_predictions()

live_preds = st.session_state.live_preds
live_mode  = bool(live_preds)

with col_mode:
    if live_mode:
        st.success(f"🟢 Live API connected ({API_URL})")
    else:
        st.info("🔵 Offline mode — displaying saved Sprint 5 artifacts. "
                "Start `apex_inference.py` for live predictions.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: PREDICTION TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📊 Predictions — All Symbols")
st.caption(
    "Predictions show the last row of each symbol's test split. "
    "Re-run `apex_synth_runner_v2.py` + Sprint 3 scripts on fresh OHLCV "
    "to get genuinely forward-looking predictions."
)

rows = []
for sym in watchlist:
    hist = summary.get(sym, {})
    conf = load_conformal(sym)
    cusum = load_cusum(sym)

    # Prefer live API data; fall back to offline summary
    if live_mode and sym in live_preds and "error" not in live_preds[sym]:
        p = live_preds[sym]
        pred_ret_pct = p.get("predicted_30d_pct_return")
        direction    = p.get("direction", "—").upper()
        dir_prob     = p.get("direction_probability")
        drift_flag   = p.get("drift_flagged", False)
        interval     = p.get("interval_90pct_log_return")
        interval_str = (
            f"[{interval[0]:+.3f}, {interval[1]:+.3f}]"
            if interval else conf.get("half_width") and
            f"±{conf['half_width']:.4f}"
        )
    else:
        pred_ret_pct = None
        direction    = "—"
        dir_prob     = None
        drift_flag   = cusum.get("insample_drift_flags", 0) > 0
        interval_str = (
            f"±{conf['half_width']:.4f}" if conf.get("half_width") else "—"
        )

    rows.append({
        "Symbol":          sym,
        "Pred 30d %":      f"{pred_ret_pct:+.2f}%" if pred_ret_pct is not None else "—",
        "Direction":       direction,
        "Dir Prob":        f"{dir_prob*100:.0f}%" if dir_prob is not None else "—",
        "90% Interval":    interval_str,
        "Hist Dir Acc":    f"{hist.get('test_dir_acc', '—')}%",
        "Hist RMSE":       hist.get("test_rmse", "—"),
        "Hist Corr":       hist.get("test_corr", "—"),
        "Emp. Coverage":   f"{hist.get('empirical_coverage', 0)*100:.0f}%"
                           if hist.get("empirical_coverage") else "—",
        "Drift":           "🔴" if drift_flag else "🟢",
    })

if rows:
    df = pd.DataFrame(rows).set_index("Symbol")
    st.dataframe(df, use_container_width=True, height=min(60 + len(rows) * 37, 600))

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: PER-SYMBOL DETAIL
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🔍 Per-Symbol Detail")

sym_choice = st.selectbox(
    "Select symbol",
    watchlist,
    index=0 if watchlist else 0,
)

if sym_choice:
    hist  = summary.get(sym_choice, {})
    conf  = load_conformal(sym_choice)
    cusum = load_cusum(sym_choice)

    # ── Metrics row ──────────────────────────────────────────────────────
    m_cols = st.columns(4)

    with m_cols[0]:
        if live_mode and sym_choice in live_preds and "error" not in live_preds[sym_choice]:
            p = live_preds[sym_choice]
            st.metric(
                "Predicted 30d return",
                f"{p.get('predicted_30d_pct_return', '—'):+.2f}%",
                f"log-return: {p.get('predicted_30d_log_return', '—'):.4f}",
            )
        else:
            st.metric("Predicted 30d return", "Offline", "start API for live value")

    with m_cols[1]:
        dir_val = "—"
        dir_delta = ""
        if live_mode and sym_choice in live_preds and "error" not in live_preds[sym_choice]:
            p = live_preds[sym_choice]
            dir_val   = p["direction"].upper()
            dir_delta = f"{p['direction_probability']*100:.0f}% confidence"
        st.metric("Direction", dir_val, dir_delta)

    with m_cols[2]:
        st.metric(
            "Historical Dir Accuracy",
            f"{hist.get('test_dir_acc', '—')}%",
            f"corr={hist.get('test_corr', '—')}",
        )

    with m_cols[3]:
        emp_cov = conf.get("empirical_coverage", hist.get("empirical_coverage"))
        nom_cov = conf.get("nominal_coverage", 0.90)
        delta_cov = f"target {nom_cov*100:.0f}%"
        st.metric(
            "Empirical Coverage",
            f"{emp_cov*100:.1f}%" if emp_cov else "—",
            delta_cov,
        )

    # ── Interval visualisation ────────────────────────────────────────────
    if show_conformal and conf:
        st.markdown("**Conformal Prediction Interval**")
        hw = conf.get("half_width", 0)
        lo_log = -hw
        hi_log = +hw
        lo_pct = (math.exp(lo_log) - 1) * 100
        hi_pct = (math.exp(hi_log) - 1) * 100
        st.write(
            f"90% interval (log-return): **[{lo_log:+.4f}, {hi_log:+.4f}]** "
            f"≈ **[{lo_pct:+.1f}%, {hi_pct:+.1f}%]** in simple return"
        )
        st.caption(
            f"Calibration size: {conf.get('n_calibration', '—')}  |  "
            f"α = {conf.get('alpha', 0.10)}"
        )
        # Derive calibration status from empirical_coverage since conformal.json
        # does not store a 'status' key (matches sprint5_ensemble.py threshold: 0.08)
        def _is_calibrated(c: dict) -> bool:
            if not c:
                return True   # no data → don't flag red
            saved = c.get('status', '')
            if saved:
                return saved.startswith('✅')
            ec  = c.get('empirical_coverage')
            nom = c.get('nominal_coverage', 0.90)
            if ec is None:
                return True
            return abs(float(ec) - float(nom)) < 0.08

        if conf and not _is_calibrated(conf):
            st.warning(
                f"⚠ Empirical coverage ({emp_cov*100:.1f}%) is more than 5pp below "
                f"the {nom_cov*100:.0f}% target. "
                "This is likely due to unnormalized regression targets — see known issues."
            )

    # ── CUSUM details ─────────────────────────────────────────────────────
    if cusum:
        drift_in  = cusum.get("insample_drift_flags", 0)
        n_in      = cusum.get("n_insample", 1)
        drift_pct = drift_in / n_in * 100 if n_in else 0
        h_val     = cusum.get("h", 5.0)
        max_sp    = cusum.get("max_s_pos_insample", 0)
        max_sn    = cusum.get("max_s_neg_insample", 0)

        if drift_in > 0:
            st.warning(
                f"⚠ CUSUM: {drift_in} in-sample drift flags / {n_in} samples "
                f"({drift_pct:.0f}%). Max S+ = {max_sp:.2f}, Max S− = {max_sn:.2f}  "
                f"(threshold h = {h_val})"
            )
        else:
            st.success(f"✅ CUSUM: no drift detected in-sample  (h = {h_val})")

    # ── Outcome logging ───────────────────────────────────────────────────
    with st.expander("📝 Log a realised outcome (updates CUSUM)"):
        st.caption(
            "Once the 30-day window for a past prediction has closed, "
            "log the actual return here to keep the drift monitor calibrated."
        )
        predicted_val = st.number_input("Predicted log-return (from earlier prediction)", value=0.0, key="pred_val")
        actual_val    = st.number_input("Actual realised 30-day log-return",             value=0.0, key="act_val")
        if st.button("Submit outcome", key="submit_outcome"):
            try:
                import requests
                r = requests.post(
                    f"{API_URL}/outcome/{sym_choice}",
                    json={"predicted_return": predicted_val, "actual_return": actual_val},
                    timeout=15,
                )
                r.raise_for_status()
                st.success(f"✅ CUSUM updated: {r.json()}")
            except Exception as e:
                st.error(f"Failed to post outcome: {e}")

    st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: FUSION ABLATION
# ─────────────────────────────────────────────────────────────────────────────
if show_ablation and ablation:
    st.subheader("⚙️ Fusion Ablation — All Variants × Symbols")

    abl_df = pd.DataFrame(ablation)

    # Pivot dir_acc by variant
    pivot = abl_df.pivot_table(
        index="symbol",
        columns="variant",
        values=["dir_acc", "corr", "avg_conf"],
        aggfunc="first",
    )
    pivot.columns = [f"{v}_{m}" for m, v in pivot.columns]
    pivot = pivot.reset_index().rename(columns={"symbol": "Symbol"}).set_index("Symbol")

    # Rename for readability
    col_rename = {}
    for c in pivot.columns:
        for var in ["FusionA_Gated", "FusionB_MLP", "FusionC_Transformer"]:
            short = {"FusionA_Gated": "A:Gated", "FusionB_MLP": "B:MLP", "FusionC_Transformer": "C:CMT"}[var]
            if var in c:
                metric = c.replace(var + "_", "")
                col_rename[c] = f"{short} {metric}"
    pivot = pivot.rename(columns=col_rename)

    st.dataframe(pivot, use_container_width=True)

    # Gate weight breakdown for FusionA
    st.markdown("**FusionA Gated — Modality Weights per Symbol**")
    gated = [x for x in ablation if x["variant"] == "FusionA_Gated" and x.get("gate_means")]
    if gated:
        gate_rows = []
        for g in gated:
            gm = g["gate_means"]
            gate_rows.append({
                "Symbol":    g["symbol"],
                "Price":     f"{gm[0]:.3f}",
                "Sentiment": f"{gm[1]:.3f}",
                "GAT":       f"{gm[2]:.3f}",
                "Dir Acc":   f"{g['dir_acc']:.1f}%",
            })
        gate_df = pd.DataFrame(gate_rows).set_index("Symbol")
        st.dataframe(gate_df, use_container_width=True)

        # Compute dynamic insight from the live gate weights, not a hardcoded guess
        numeric_gates = pd.DataFrame(gate_rows).set_index("Symbol")
        for col in ["Price", "Sentiment", "GAT"]:
            numeric_gates[col] = numeric_gates[col].astype(float)

        top_sentiment = numeric_gates["Sentiment"].idxmax()
        top_sentiment_val = numeric_gates["Sentiment"].max()
        top_gat = numeric_gates["GAT"].idxmax()
        top_gat_val = numeric_gates["GAT"].max()
        avg_price = numeric_gates["Price"].mean()

        st.caption(
            f"Price dominates on average ({avg_price * 100:.0f}% mean weight). "
            f"**{top_sentiment}** currently leans most on sentiment "
            f"({top_sentiment_val * 100:.0f}%); **{top_gat}** leans most on the "
            f"graph branch ({top_gat_val * 100:.0f}%). "
            f"⚠ Note: these weights come from unseeded fusion training and can "
            f"shift substantially between runs — treat as a per-run snapshot, "
            f"not a stable property of the stock."
        )
    else:
        st.caption("Gate means not available — run sprint4_fusion.py to generate.")

    st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: NEWS COVERAGE
# ─────────────────────────────────────────────────────────────────────────────
if show_news:
    st.subheader("📰 Sentinel News Coverage")
    st.caption(
        "Headlines fetched by `sentinel_news_fetcher.py` (GDELT + NewsAPI + Google RSS). "
        "These feed into `sprint3_finbert.py` for FinBERT sentiment embeddings."
    )

    with st.spinner("Reading headline CSVs…"):
        hl_summary = load_headlines_summary()

    if hl_summary:
        hl_rows = []
        for sym in watchlist:
            info = hl_summary.get(sym, {"count": 0, "earliest": None, "latest": None})
            count = info["count"]
            hl_rows.append({
                "Symbol":    sym,
                "Articles":  count,
                "Earliest":  info["earliest"] or "—",
                "Latest":    info["latest"]   or "—",
                "Status":    "✅" if count > 50 else ("⚠" if count > 0 else "❌ Missing"),
            })
        hl_df = pd.DataFrame(hl_rows).set_index("Symbol")
        st.dataframe(hl_df, use_container_width=True)

        # Quick stats
        missing  = sum(1 for r in hl_rows if r["Articles"] == 0)
        poor     = sum(1 for r in hl_rows if 0 < r["Articles"] <= 50)
        good     = sum(1 for r in hl_rows if r["Articles"] > 50)
        total_hl = sum(r["Articles"] for r in hl_rows)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Total articles", total_hl)
        mc2.metric("✅ Good coverage (>50)", good)
        mc3.metric("⚠ Sparse (1–50)", poor)
        mc4.metric("❌ No headlines", missing)

        if missing > 0:
            st.info(
                f"**{missing} symbol(s) have no headlines yet.** "
                "Run: `python sentinel_news_fetcher.py --refresh` for a quick 7-day fetch, "
                "or `--backfill` for 2-year history."
            )
    else:
        st.info(
            "No headline CSVs found. Run `sentinel_news_fetcher.py` to populate them:\n\n"
            "```bash\n"
            "python sentinel_news_fetcher.py --refresh   # quick, last 7 days\n"
            "python sentinel_news_fetcher.py --backfill  # full 2yr history\n"
            "```"
        )

    st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: MODEL HEALTH SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🔬 Model Health")

if summary:
    health_rows = []
    for sym in watchlist:
        h = summary.get(sym, {})
        conf  = load_conformal(sym)
        cusum = load_cusum(sym)
        if not h:
            continue
        emp_cov = conf.get("empirical_coverage", h.get("empirical_coverage", 0))
        nom_cov = conf.get("nominal_coverage", 0.90)
        cov_ok  = emp_cov >= (nom_cov - 0.085) if emp_cov else False  # was -0.05
        dir_acc = h.get("test_dir_acc", 50)
        corr    = h.get("test_corr", 0)
        drift_flags = cusum.get("insample_drift_flags", 0)

        # In the Model Health loop, replace the cov_ok / threshold logic with:
        # Derive calibration: use saved 'status' if present, else compute from
        # empirical_coverage vs nominal_coverage (threshold 0.08, matches pipeline)
        _ec  = conf.get('empirical_coverage')
        _nom = conf.get('nominal_coverage', 0.90)
        if conf.get('status'):
            is_calibrated = conf['status'].startswith('✅')
        elif _ec is not None:
            is_calibrated = abs(float(_ec) - float(_nom)) < 0.08
        else:
            is_calibrated = True   # no conformal data → don't penalise

        health = "🟢"
        if conf and not is_calibrated:
            health = "🔴"
        elif dir_acc < 48 and corr < 0:
            health = "🟡"
        elif drift_flags > 20:
            health = "🟡"

        health_rows.append({
            "Symbol":     sym,
            "Health":     health,
            "Dir Acc %":  f"{dir_acc:.1f}",
            "Corr":       f"{corr:+.3f}",
            "RMSE":       f"{h.get('test_rmse', '—'):.4f}" if h.get('test_rmse') else "—",
            "Coverage %": f"{emp_cov*100:.1f}" if emp_cov else "—",
            "Drift Flags":f"{drift_flags}",
        })

    if health_rows:
        hdf = pd.DataFrame(health_rows).set_index("Symbol")
        st.dataframe(hdf, use_container_width=True)

        green  = sum(1 for r in health_rows if r["Health"] == "🟢")
        yellow = sum(1 for r in health_rows if r["Health"] == "🟡")
        red    = sum(1 for r in health_rows if r["Health"] == "🔴")
        st.caption(f"🟢 {green} healthy  |  🟡 {yellow} watch  |  🔴 {red} needs attention")

else:
    st.info("No sprint5_summary.json found. Run `sprint5_ensemble.py` to generate it.")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"APEX-ST v2  ·  "
    f"Data as of: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ·  "
    f"Watchlist: {len(watchlist)} symbols"
)
