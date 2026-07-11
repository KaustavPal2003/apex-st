"""
apex_inference.py — Sprint 6: APEX-ST Inference Service
════════════════════════════════════════════════════════════════
FastAPI service that serves the trained APEX-ST ensemble as a
live prediction endpoint, with conformal intervals and CUSUM
drift monitoring.

Endpoints
──────────
  GET  /                         Health check + model status
  GET  /predict                  Latest predictions for all symbols
  GET  /predict/{symbol}         Single-symbol prediction
  GET  /drift/{symbol}           CUSUM drift state for a symbol
  POST /outcome/{symbol}         Log a realised outcome → updates CUSUM
  GET  /health                   Detailed model artefact health check

Run
────
  pip install fastapi uvicorn[standard] xgboost torch numpy
  uvicorn apex_inference:app --host 0.0.0.0 --port 8000

  Then open dashboard.py (streamlit run dashboard.py) and set
  API URL to http://localhost:8000.

Artefacts required (per symbol, in the working directory)
───────────────────────────────────────────────────────────
  {SYM}_ensemble_result.json     Real reg + cls predictions (Sprint 5 output) ← drives /predict
  {SYM}_conformal.json           Conformal calibration half-width + status
  {SYM}_cusum_state.json         CUSUM baseline (mu0, sigma0, h, k)
  watchlist.json                 Symbol list
  sprint5_summary.json           Cross-symbol summary (historical metrics)

Optional (loaded if present, not required for /predict to function):
  {SYM}_ensemble_xgb.json        XGBoost regression booster  (for future live-fusion path)
  {SYM}_ensemble_xgb_cls.json    XGBoost classification booster (for future live-fusion path)
  {SYM}_apex_X_test.npy          Test-split feature sequences
"""

from __future__ import annotations

import json
import math
import os
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("apex_inference")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKDIR = Path(os.getenv("APEX_WORKDIR", "."))

# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="APEX-ST Inference API",
    description="NSE 30-day stock direction + return prediction with conformal intervals",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class OutcomeRequest(BaseModel):
    predicted_return: float
    actual_return: float


class PredictionResponse(BaseModel):
    symbol: str
    as_of: str                          # ISO-8601 UTC timestamp of this prediction
    predicted_30d_log_return: float
    predicted_30d_pct_return: float     # (exp(log_return) - 1) * 100
    direction: str                      # "UP" | "DOWN"
    direction_probability: float        # cls model's confidence [0,1]
    interval_90pct_log_return: List[float]  # [lo, hi] conformal interval
    interval_90pct_pct: List[float]         # same in simple % return
    drift_flagged: bool
    cusum_s_pos: float
    cusum_s_neg: float
    model_status: str                   # "ok" | "stale" | "missing"


# ─────────────────────────────────────────────────────────────────────────────
# ARTEFACT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_watchlist() -> List[str]:
    p = WORKDIR / "watchlist.json"
    if p.exists():
        return json.loads(p.read_text()).get("watchlist", [])
    # Fall back to sprint5_summary keys
    sp = WORKDIR / "sprint5_summary.json"
    if sp.exists():
        return list(json.loads(sp.read_text()).keys())
    return []


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_npy(path: Path) -> Optional[np.ndarray]:
    try:
        return np.load(str(path))
    except Exception:
        return None


def _load_xgb(path: Path) -> Optional[xgb.Booster]:
    try:
        model = xgb.Booster()
        model.load_model(str(path))
        return model
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CACHE  (loaded once at startup, not on every request)
# ─────────────────────────────────────────────────────────────────────────────

class SymbolState:
    """All artefacts for one symbol, loaded at startup."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ok = False
        self.missing: List[str] = []

        # XGBoost boosters — loaded optionally for a future live-fusion path
        # where Sprint 4 fusion models run per-request and feed real meta-features.
        # Not required for /predict, which now reads ensemble_result.json directly.
        self.reg_model = _load_xgb(WORKDIR / f"{symbol}_ensemble_xgb.json")   # optional
        self.cls_model = _load_xgb(WORKDIR / f"{symbol}_ensemble_xgb_cls.json")  # optional

        # Calibration + drift artefacts
        self.conformal  = _load_json(WORKDIR / f"{symbol}_conformal.json")
        self.cusum_state = _load_json(WORKDIR / f"{symbol}_cusum_state.json")

        # Sprint 5 summary (historical test metrics)
        summary_path = WORKDIR / "sprint5_summary.json"
        self.summary = {}
        if summary_path.exists():
            full = _load_json(summary_path) or {}
            self.summary = full.get(symbol, {})

        # Test-split embeddings — used to produce the "latest" prediction
        # (last row of the test split = most recent observed window)
        self.X_test     = _load_npy(WORKDIR / f"{symbol}_apex_X_test.npy")
        self.sent_test  = _load_npy(WORKDIR / f"{symbol}_apex_sent_test.npy")
        self.gat_test   = _load_npy(WORKDIR / f"{symbol}_apex_gat_test.npy")

        # Per-symbol ensemble result — contains real reg_pred_test + cls_pred_test
        self.ensemble_result = _load_json(
            WORKDIR / f"{symbol}_ensemble_result.json"
        )
        if self.ensemble_result is None:
            log.warning(
                f"  ⚠   {symbol}: _ensemble_result.json missing — "
                f"re-run sprint5_ensemble.py to generate real predictions"
            )

        # Check required artefacts — only files that actually drive /predict
        # ensemble_result.json is the critical one: missing it means /predict
        # silently returns zeros even though /health would show ok=True.
        # reg/cls XGBoost boosters are optional (future live-fusion path).
        required = {
            "ensemble_result.json": self.ensemble_result,
            "conformal.json":       self.conformal,
            "cusum_state.json":     self.cusum_state,
        }
        for name, val in required.items():
            if val is None:
                self.missing.append(f"{symbol}_{name}")

        self.ok = len(self.missing) == 0
        if self.ok:
            log.info(f"  ✅  {symbol} — artefacts loaded")
        else:
            log.warning(f"  ⚠   {symbol} — missing: {self.missing}")

        # Live CUSUM state (mutable — updated by /outcome endpoint)
        self._cusum_s_pos: float = float(
            self.cusum_state.get("s_pos_current", 0.0) if self.cusum_state else 0.0
        )
        self._cusum_s_neg: float = float(
            self.cusum_state.get("s_neg_current", 0.0) if self.cusum_state else 0.0
        )
        self._cusum_h: float = float(
            self.cusum_state.get("h", 5.0) if self.cusum_state else 5.0
        )
        self._cusum_k: float = float(
            self.cusum_state.get("k", 0.5) if self.cusum_state else 0.5
        )
        self._cusum_mu: float = float(
            self.cusum_state.get("mu0", 0.0) if self.cusum_state else 0.0
        )
        self._cusum_sigma: float = float(
            self.cusum_state.get("sigma0", 1.0) if self.cusum_state else 1.0
        )
        self._drift_flagged: bool = False


# Global model registry — populated at startup
_registry: Dict[str, SymbolState] = {}
_startup_time: str = ""


@app.on_event("startup")
def load_all_models():
    global _startup_time
    _startup_time = datetime.now(timezone.utc).isoformat()

    log.info("═" * 60)
    log.info("  APEX-ST Inference Service — loading artefacts")
    log.info(f"  Working dir : {WORKDIR.resolve()}")
    log.info(f"  Device      : {DEVICE}")
    log.info("═" * 60)

    symbols = _load_watchlist()
    if not symbols:
        log.warning("  watchlist.json not found — no symbols loaded")
        return

    for sym in symbols:
        _registry[sym] = SymbolState(sym)

    n_ok = sum(1 for s in _registry.values() if s.ok)
    log.info(f"  Loaded {n_ok}/{len(_registry)} symbols successfully")
    log.info("═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def _make_prediction(state: SymbolState) -> PredictionResponse:
    """
    Produce a prediction for one symbol using the last row of the test split.

    The XGBoost meta-learner expects a feature vector derived from Sprint 4's
    three fusion-variant outputs.  Since we have the test-split predictions
    saved in sprint5_summary (reg_pred_test / cls_pred_test), we use those
    directly rather than re-running the fusion models on every request — the
    "latest" prediction is the last test-set row.

    For a genuinely forward-looking prediction on fresh unseen data, you would
    first re-run apex_synth_runner_v2.py + sprint3_finbert.py + sprint3_gat.py
    to generate new embeddings, then call the fusion models, then the XGBoost
    meta-learner.  That pipeline is not wired here to keep the inference server
    lightweight; this server serves the most recent trained result.
    """
    sym = state.symbol
    as_of = datetime.now(timezone.utc).isoformat()

    # ── Pull test-split predictions ───────────────────────────────────────
    # sprint5_summary.json stores aggregate metrics (test_dir_acc, test_rmse,
    # test_corr) but NOT the full prediction arrays.  Those are in the per-
    # symbol {SYM}_ensemble_xgb.json artefacts; the most recent single
    # prediction is obtained by running the XGBoost regressor + classifier on
    # the last row of the test-split embeddings.
    #
    # We load the test-split npy embeddings from SymbolState, take the last
    # row, and run inference.  If embeddings are unavailable (older artefacts
    # that pre-date the embedding save step), we fall back to the summary's
    # test_dir_acc to populate a plausible direction estimate.

    # ── Real last-test-row prediction from ensemble_result.json ──────────
    # sprint5_ensemble.py now saves reg_pred_test and cls_pred_test as
    # lists.  We take the LAST element — the most recent sample in the
    # test split, i.e. the prediction window ending closest to today.
    # This is a genuine model output, not a placeholder.
    er = state.ensemble_result
    if er and er.get("reg_pred_test") and er.get("cls_pred_test"):
        pred_log_return = float(er["reg_pred_test"][-1])
        cls_prob        = float(er["cls_pred_test"][-1])
        # cls_pred_test stores class-1 probabilities (from predict_proba[:,1])
        # so it's already in [0,1] — no sigmoid needed.
    else:
        # ensemble_result.json missing — re-run sprint5_ensemble.py
        log.warning(f"{sym}: ensemble_result.json missing, returning zeros")
        pred_log_return = 0.0
        cls_prob        = 0.5

    pred_pct_return = (math.exp(pred_log_return) - 1) * 100
    direction       = "UP" if cls_prob >= 0.5 else "DOWN"

    # ── Conformal interval ─────────────────────────────────────────────────
    hw = float(state.conformal.get("half_width", 0.025)) if state.conformal else 0.025
    lo_log = pred_log_return - hw
    hi_log = pred_log_return + hw
    lo_pct = (math.exp(lo_log) - 1) * 100
    hi_pct = (math.exp(hi_log) - 1) * 100

    # ── CUSUM live state ───────────────────────────────────────────────────
    drift_flagged = state._drift_flagged or (
        state._cusum_s_pos > state._cusum_h or
        abs(state._cusum_s_neg) > state._cusum_h
    )

    # ── Model status ───────────────────────────────────────────────────────
    model_status = "ok" if state.ok else "missing"

    return PredictionResponse(
        symbol=sym,
        as_of=as_of,
        predicted_30d_log_return=pred_log_return,
        predicted_30d_pct_return=round(pred_pct_return, 4),
        direction=direction,
        direction_probability=round(cls_prob, 4),
        interval_90pct_log_return=[round(lo_log, 4), round(hi_log, 4)],
        interval_90pct_pct=[round(lo_pct, 2), round(hi_pct, 2)],
        drift_flagged=drift_flagged,
        cusum_s_pos=round(state._cusum_s_pos, 4),
        cusum_s_neg=round(state._cusum_s_neg, 4),
        model_status=model_status,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    n_ok = sum(1 for s in _registry.values() if s.ok)
    return {
        "service": "APEX-ST Inference API",
        "version": "1.0.0",
        "status": "running",
        "symbols_loaded": n_ok,
        "symbols_total": len(_registry),
        "startup_time": _startup_time,
        "device": str(DEVICE),
    }


@app.get("/health")
def health():
    """Detailed per-symbol artefact health check."""
    report = {}
    for sym, state in _registry.items():
        # Derive calibration status from empirical_coverage since older
        # conformal.json files may not have the 'status' key written.
        # Threshold matches sprint5_ensemble.py: abs(ec - nominal) < 0.08
        conf_status = "missing"
        if state.conformal:
            ec  = state.conformal.get("empirical_coverage")
            nom = state.conformal.get("nominal_coverage", 0.9)
            if state.conformal.get("status"):
                conf_status = state.conformal["status"]   # use saved value if present
            elif ec is not None:
                conf_status = (
                    "✅ well-calibrated"
                    if abs(float(ec) - float(nom)) < 0.08
                    else "⚠ check calibration"
                )
        report[sym] = {
            "ok": state.ok,
            "missing_artefacts": state.missing,
            "conformal_status": conf_status,
            "empirical_coverage": (
                state.conformal.get("empirical_coverage")
                if state.conformal else None
            ),
            "drift_flagged": state._drift_flagged,
            "cusum_s_pos": round(state._cusum_s_pos, 4),
            "cusum_s_neg": round(state._cusum_s_neg, 4),
        }
    return {
        "healthy": sum(1 for r in report.values() if r["ok"]),
        "total": len(report),
        "symbols": report,
    }


@app.get("/predict")
def predict_all():
    """
    Return latest predictions for every loaded symbol.
    Response is a dict keyed by symbol — matches dashboard's fetch_live_predictions().
    """
    if not _registry:
        raise HTTPException(status_code=503, detail="No symbols loaded — check watchlist.json")

    result = {}
    for sym, state in _registry.items():
        if not state.ok:
            result[sym] = {
                "error": f"artefacts missing: {state.missing}",
                "symbol": sym,
            }
            continue
        try:
            result[sym] = _make_prediction(state).model_dump()
        except HTTPException as e:
            result[sym] = {"error": e.detail, "symbol": sym}
        except Exception as e:
            log.exception(f"Prediction failed for {sym}")
            result[sym] = {"error": str(e), "symbol": sym}

    return result


@app.get("/predict/{symbol}")
def predict_one(symbol: str):
    """Return the latest prediction for a single symbol."""
    sym = symbol.upper()
    if sym not in _registry:
        raise HTTPException(status_code=404, detail=f"Symbol {sym} not in watchlist")
    state = _registry[sym]
    if not state.ok:
        raise HTTPException(
            status_code=503,
            detail=f"{sym}: missing artefacts {state.missing}"
        )
    return _make_prediction(state)


@app.get("/drift/{symbol}")
def drift(symbol: str):
    """
    Return current CUSUM drift state for a symbol.
    Matches dashboard's fetch_live_drift(symbol).
    """
    sym = symbol.upper()
    if sym not in _registry:
        raise HTTPException(status_code=404, detail=f"Symbol {sym} not in watchlist")

    state = _registry[sym]
    cs = state.cusum_state or {}

    return {
        "symbol": sym,
        "drift_flagged": state._drift_flagged,
        "s_pos_current": round(state._cusum_s_pos, 4),
        "s_neg_current": round(state._cusum_s_neg, 4),
        "h": state._cusum_h,
        "k": state._cusum_k,
        "baseline_mean": state._cusum_mu,  # cusum_state.json key: mu0
        "baseline_std": state._cusum_sigma,  # cusum_state.json key: sigma0
        "insample_drift_flags": cs.get("insample_drift_flags", 0),
        "n_insample": cs.get("n_insample", 0),
        "max_s_pos_insample": cs.get("max_s_pos_insample", 0),
        "max_s_neg_insample": cs.get("max_s_neg_insample", 0),
    }


@app.post("/outcome/{symbol}")
def log_outcome(symbol: str, body: OutcomeRequest):
    """
    Log a realised 30-day outcome and update the live CUSUM state.

    Once a prediction window has closed (30 trading days after the prediction
    was made), call this endpoint with the predicted and actual log-returns.
    The CUSUM accumulator is updated in-memory; state is also persisted back
    to {SYMBOL}_cusum_state.json so it survives service restarts.

    Matches dashboard's POST /outcome/{symbol_choice} call.
    """
    sym = symbol.upper()
    if sym not in _registry:
        raise HTTPException(status_code=404, detail=f"Symbol {sym} not in watchlist")

    state = _registry[sym]
    pred  = body.predicted_return
    actual = body.actual_return

    # Standardise the residual using in-sample baseline statistics
    residual = actual - pred
    std = state._cusum_sigma if state._cusum_sigma > 1e-8 else 1.0
    z = (residual - state._cusum_mu) / std

    # One-sided CUSUM update (Page's test)
    k = state._cusum_k
    state._cusum_s_pos = max(0.0, state._cusum_s_pos + z - k)
    state._cusum_s_neg = min(0.0, state._cusum_s_neg + z + k)

    # Check threshold
    h = state._cusum_h
    new_flag = (state._cusum_s_pos > h or abs(state._cusum_s_neg) > h)
    if new_flag and not state._drift_flagged:
        log.warning(f"  🔴  CUSUM drift detected for {sym}  "
                    f"S+={state._cusum_s_pos:.2f}  S−={state._cusum_s_neg:.2f}")
    state._drift_flagged = new_flag

    # Persist updated state
    if state.cusum_state is not None:
        state.cusum_state["s_pos_current"] = round(state._cusum_s_pos, 6)  # live accumulator
        state.cusum_state["s_neg_current"] = round(state._cusum_s_neg, 6)  # live accumulator
        state.cusum_state["drift_flagged"] = new_flag
        cusum_path = WORKDIR / f"{sym}_cusum_state.json"
        try:
            cusum_path.write_text(
                json.dumps(state.cusum_state, indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            log.error(f"Failed to persist CUSUM state for {sym}: {e}")

    return {
        "symbol": sym,
        "predicted_return": pred,
        "actual_return": actual,
        "residual": round(residual, 6),
        "z_score": round(z, 4),
        "s_pos": round(state._cusum_s_pos, 4),
        "s_neg": round(state._cusum_s_neg, 4),
        "drift_flagged": new_flag,
        "message": (
            "⚠ Drift detected — consider retraining" if new_flag
            else "✅ CUSUM within bounds"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apex_inference:app", host="0.0.0.0", port=8000, reload=False)
