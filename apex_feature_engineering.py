# apex_feature_engineering.py  — APEX-ST Sprint 1
"""
Advanced Feature Engineering for APEX-ST model.
Adds three feature-engineering modules on top of standard TA features:

  1. WaveletDecomposer    — Discrete Wavelet Transform (Daubechies-4, 4 levels)
  2. RegimeDetector       — Hidden Markov Model (4 latent states)
  3. KPCAReducer          — Kernel PCA (RBF kernel, 120 components)
  4. AdvancedFeaturePipeline — Orchestrates all three + existing TA features

AdvancedFeaturePipeline below is the single entry point that orchestrates
all three alongside the existing TA feature builder.
"""

import numpy as np
import pandas as pd
import pywt
import pickle
import warnings
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# 1.  TECHNICAL INDICATOR BUILDER
#     (self-contained so this module works without the old pipeline present)
# ─────────────────────────────────────────────────────────────────────────────

class TechnicalIndicatorBuilder:
    """
    Compute the full set of TA indicators required by APEX-ST Phase 2.

    Trend  : SMA-5/20/50, EMA-12/26
    Momentum: RSI-14, MACD(12-26-9), Stochastic(14,3)
    Volatility: Bollinger Bands(20,2), ATR-14
    Volume: OBV, VWAP (daily approximation)
    """

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _true_range(df: pd.DataFrame) -> pd.Series:
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift(1)).abs()
        lc = (df['low']  - df['close'].shift(1)).abs()
        return pd.concat([hl, hc, lc], axis=1).max(axis=1)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Append all TA columns to df (in-place copy).
        df must contain: open, high, low, close, volume.
        Returns df with indicator columns added.
        """
        df = df.copy()
        c = df['close']
        v = df['volume']

        # ── Trend ──
        for w in (5, 20, 50):
            df[f'sma_{w}'] = c.rolling(w).mean()
        df['ema_12'] = self._ema(c, 12)
        df['ema_26'] = self._ema(c, 26)

        # ── Momentum ──
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs   = gain / (loss + 1e-9)
        df['rsi_14'] = 100 - 100 / (1 + rs)

        ema12 = self._ema(c, 12)
        ema26 = self._ema(c, 26)
        df['macd']        = ema12 - ema26
        df['macd_signal'] = self._ema(df['macd'], 9)
        df['macd_hist']   = df['macd'] - df['macd_signal']

        low14  = df['low'].rolling(14).min()
        high14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (c - low14) / (high14 - low14 + 1e-9)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # ── Volatility ──
        mid  = c.rolling(20).mean()
        std  = c.rolling(20).std()
        df['bb_upper'] = mid + 2 * std
        df['bb_lower'] = mid - 2 * std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (mid + 1e-9)
        df['bb_pct']   = (c - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)

        tr = self._true_range(df)
        df['atr_14'] = tr.rolling(14).mean()

        # ── Volume ──
        obv = [0]
        for i in range(1, len(df)):
            if c.iloc[i] > c.iloc[i - 1]:
                obv.append(obv[-1] + v.iloc[i])
            elif c.iloc[i] < c.iloc[i - 1]:
                obv.append(obv[-1] - v.iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv

        # VWAP approximation (daily reset not possible without intraday; use rolling)
        df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-9)

        # ── Rolling Z-score normalisation (60-period) ──
        for col in ['open', 'high', 'low', 'close', 'volume']:
            roll_mean = df[col].rolling(60).mean()
            roll_std  = df[col].rolling(60).std() + 1e-9
            df[f'{col}_z'] = (df[col] - roll_mean) / roll_std

        print(f"  ✓ Technical indicators computed ({len(df.columns)} total columns)")
        return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  WAVELET DECOMPOSER
# ─────────────────────────────────────────────────────────────────────────────

class WaveletDecomposer:
    """
    Discrete Wavelet Transform using Daubechies-4 wavelet, 4 decomposition levels.

    For each input price series it produces:
      - 4 detail coefficient arrays  (d1 … d4)  — high-frequency noise / short cycles
      - 1 approximation array        (a4)        — low-frequency trend
    Each array is interpolated back to the original series length so it can be
    concatenated with other tabular features.

    This gives 5 extra columns per input series (close, by default).
    """

    def __init__(self, wavelet: str = 'db4', level: int = 4):
        self.wavelet = wavelet
        self.level   = level
        pywt.Wavelet(wavelet)          # validate wavelet name early
        print(f"  WaveletDecomposer initialised  (wavelet={wavelet}, levels={level})")

    def _pad_to_power2(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        """Zero-pad to next power of 2 to avoid boundary artefacts."""
        n      = len(x)
        target = int(2 ** np.ceil(np.log2(n)))
        padded = np.pad(x, (0, target - n), mode='reflect')
        return padded, n

    def decompose_series(self, series: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Decompose a 1-D series and return all coefficient arrays
        resampled back to the original length.
        """
        series = np.asarray(series, dtype=np.float64)
        original_len = len(series)

        # Remove NaN by forward-fill before decomposing
        mask = np.isnan(series)
        if mask.any():
            idx = np.where(~mask)[0]
            series[mask] = np.interp(np.where(mask)[0], idx, series[idx])

        padded, n_orig = self._pad_to_power2(series)
        coeffs = pywt.wavedec(padded, self.wavelet, level=self.level)
        # coeffs[0] = approximation a4, coeffs[1..4] = details d4 … d1

        result = {}
        for i, c in enumerate(coeffs):
            key = f'a{self.level}' if i == 0 else f'd{self.level - i + 1}'
            # Upsample back to original length
            resampled = np.interp(
                np.linspace(0, 1, original_len),
                np.linspace(0, 1, len(c)),
                c
            )
            result[key] = resampled

        return result

    def transform(self, df: pd.DataFrame,
                  columns: List[str] = ('close',)) -> pd.DataFrame:
        """
        Add wavelet feature columns for each column in `columns`.
        New columns: {col}_wt_a4, {col}_wt_d4, …, {col}_wt_d1
        """
        df = df.copy()
        total_new = 0
        for col in columns:
            if col not in df.columns:
                print(f"  ⚠  Column '{col}' not found — skipping wavelet decomp")
                continue
            decomp = self.decompose_series(df[col].values)
            for key, values in decomp.items():
                df[f'{col}_wt_{key}'] = values
                total_new += 1

        print(f"  ✓ Wavelet decomposition: {total_new} new columns added "
              f"(wavelet={self.wavelet}, levels={self.level})")
        return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  REGIME DETECTOR  (Hidden Markov Model)
# ─────────────────────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    4-state Gaussian HMM trained on daily returns and realised volatility.

    States (unlabelled — learned from data, then heuristically named):
      0 → low-vol bull          (pre-COVID stable)
      1 → high-vol crisis       (COVID crash)
      2 → recovery / momentum
      3 → bear / correction

    Outputs:
      regime_label    : integer 0-3
      regime_prob_*   : posterior probability of each state (4 columns)
    """

    REGIME_NAMES = {0: 'bull_stable', 1: 'crisis', 2: 'recovery', 3: 'bear'}

    def __init__(self, n_states: int = 4, n_iter: int = 200, random_state: int = 42):
        self.n_states    = n_states
        self.n_iter      = n_iter
        self.random_state = random_state
        self.model: Optional[GaussianHMM] = None
        self._obs_scaler = StandardScaler()
        self._is_fitted  = False

    # ── feature construction ──────────────────────────────────────────────────

    @staticmethod
    def _build_obs(df: pd.DataFrame) -> np.ndarray:
        """
        Build observation matrix for HMM:
          col 0 : log daily return
          col 1 : 5-day rolling realised volatility
          col 2 : 20-day rolling realised volatility
        """
        log_ret = np.log(df['close'] / df['close'].shift(1)).fillna(0).values
        vol5    = pd.Series(log_ret).rolling(5).std().fillna(0).values
        vol20   = pd.Series(log_ret).rolling(20).std().fillna(0).values
        return np.column_stack([log_ret, vol5, vol20])

    # ── fit / predict ─────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> 'RegimeDetector':
        """Fit HMM on training data."""
        print(f"  Fitting HMM ({self.n_states} states) on {len(df)} observations …")
        obs = self._build_obs(df)

        # Standardize observations — critical for HMM numerical stability
        self._obs_scaler = StandardScaler()
        obs_sc = self._obs_scaler.fit_transform(obs)

        self.model = GaussianHMM(
            n_components    = self.n_states,
            covariance_type = 'diag',   # more numerically stable than 'full'
            n_iter          = self.n_iter,
            random_state    = self.random_state,
            verbose         = False
        )
        self.model.fit(obs_sc)
        self._is_fitted = True
        print(f"  ✓ HMM fitted  (converged={self.model.monitor_.converged})")
        self._print_regime_stats(df, obs_sc)
        return self

    def _print_regime_stats(self, df: pd.DataFrame, obs_sc: np.ndarray):
        labels = self.model.predict(obs_sc)
        obs    = self._build_obs(df)          # unscaled for readable stats
        log_ret = obs[:, 0]
        for s in range(self.n_states):
            mask = labels == s
            mean_ret = log_ret[mask].mean() * 252
            mean_vol = obs[mask, 2].mean() * np.sqrt(252)
            pct      = mask.mean() * 100
            print(f"    State {s}: {pct:5.1f}% of days | "
                  f"ann.ret={mean_ret:+.2%} | ann.vol={mean_vol:.2%}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime columns to df.
        Must call fit() first (or load a saved model).
        """
        if not self._is_fitted:
            raise RuntimeError("RegimeDetector not fitted. Call fit() first.")

        df         = df.copy()
        obs        = self._build_obs(df)
        obs_sc     = self._obs_scaler.transform(obs)
        labels     = self.model.predict(obs_sc)
        posteriors = self.model.predict_proba(obs_sc)

        df['regime_label'] = labels
        for s in range(self.n_states):
            df[f'regime_prob_{s}'] = posteriors[:, s]

        for s in range(self.n_states):
            df[f'regime_{s}'] = (labels == s).astype(np.float32)

        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"  ✓ Regime labels added — distribution: {dist}")
        return df

    def save(self, path: str = 'regime_hmm.pkl'):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'n_states': self.n_states,
                         'obs_scaler': self._obs_scaler}, f)
        print(f"  ✓ RegimeDetector saved → {path}")

    def load(self, path: str = 'regime_hmm.pkl') -> 'RegimeDetector':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model        = data['model']
        self.n_states     = data['n_states']
        self._obs_scaler  = data['obs_scaler']
        self._is_fitted   = True
        print(f"  ✓ RegimeDetector loaded ← {path}")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# 4.  KERNEL PCA REDUCER
# ─────────────────────────────────────────────────────────────────────────────

class KPCAReducer:
    """
    Kernel PCA with an RBF kernel reducing the feature matrix to exactly
    `n_components` uncorrelated principal components.

    Important: fitted ONLY on training data to avoid look-ahead bias.
    """

    def __init__(self, n_components: int = 120, kernel: str = 'rbf',
                 gamma: Optional[float] = None, random_state: int = 42):
        self.n_components  = n_components
        self.kernel        = kernel
        self.gamma         = gamma
        self.random_state  = random_state
        self._kpca: Optional[KernelPCA] = None
        self._scaler       = StandardScaler()
        self._is_fitted    = False

    def fit(self, X_train: np.ndarray) -> 'KPCAReducer':
        """
        Fit on training features (2-D array: samples × features).
        Applies StandardScaler first (required for RBF KPCA stability).
        """
        print(f"  Fitting KPCA (kernel={self.kernel}, n_components={self.n_components}) "
              f"on {X_train.shape} …")
        X_sc = self._scaler.fit_transform(X_train)

        n_comp = min(self.n_components, X_sc.shape[0] - 1, X_sc.shape[1])
        self._kpca = KernelPCA(
            n_components   = n_comp,
            kernel         = self.kernel,
            gamma          = self.gamma,
            fit_inverse_transform = False,
            random_state   = self.random_state,
            n_jobs         = -1
        )
        self._kpca.fit(X_sc)
        self._is_fitted = True
        print(f"  ✓ KPCA fitted — output dim: {n_comp}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("KPCAReducer not fitted. Call fit() first.")
        X_sc = self._scaler.transform(X)
        return self._kpca.transform(X_sc)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def save(self, path: str = 'kpca_reducer.pkl'):
        with open(path, 'wb') as f:
            pickle.dump({'kpca': self._kpca, 'scaler': self._scaler,
                         'n_components': self.n_components}, f)
        print(f"  ✓ KPCAReducer saved → {path}")

    def load(self, path: str = 'kpca_reducer.pkl') -> 'KPCAReducer':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._kpca       = data['kpca']
        self._scaler     = data['scaler']
        self.n_components = data['n_components']
        self._is_fitted  = True
        print(f"  ✓ KPCAReducer loaded ← {path}")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# 5.  WALK-FORWARD SPLITTER
#     (replaces the simple ratio split for strict no-look-ahead evaluation)
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardSplitter:
    """
    Generates expanding-window train/test folds mimicking live trading.

    For a 10-year dataset (2015–2025) with annual folds:
      Fold 1 : train 2015-2019 | test 2020
      Fold 2 : train 2015-2020 | test 2021
      …
      Fold 5 : train 2015-2024 | test 2025
    """

    def __init__(self, n_folds: int = 5, min_train_frac: float = 0.5):
        self.n_folds        = n_folds
        self.min_train_frac = min_train_frac

    def split(self, X: np.ndarray, y: np.ndarray
              ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Returns a list of (X_train, y_train, X_test, y_test) tuples.
        """
        n      = len(X)
        fold_size = n // (self.n_folds + 1)   # approximate fold size
        folds  = []

        for k in range(1, self.n_folds + 1):
            train_end = k * fold_size
            test_end  = min((k + 1) * fold_size, n)

            if train_end < int(n * self.min_train_frac):
                continue                      # skip folds with insufficient training data

            X_tr, y_tr = X[:train_end],         y[:train_end]
            X_te, y_te = X[train_end:test_end], y[train_end:test_end]
            folds.append((X_tr, y_tr, X_te, y_te))

        print(f"  ✓ WalkForwardSplitter: {len(folds)} folds generated "
              f"(~{fold_size} samples/fold)")
        return folds


# ─────────────────────────────────────────────────────────────────────────────
# 6.  ADVANCED FEATURE PIPELINE  (orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedFeaturePipeline:
    """
    Full Phase 2 pipeline for APEX-ST.

    Steps
    ─────
    1. Compute TA indicators (TechnicalIndicatorBuilder)
    2. Add wavelet decomposition columns (WaveletDecomposer)
    3. Fit & label market regimes (RegimeDetector)
    4. Fit KPCA on training rows & reduce dimensionality (KPCAReducer)
    5. Create 60-step lookback sequences
    6. Split using WalkForwardSplitter

    Output tensor shape:  (samples, 60, 120)  — matches APEX-ST Branch 1 input.
    """

    def __init__(self,
                 lookback: int = 60,
                 n_kpca_components: int = 120,
                 wavelet_cols: List[str] = ('close', 'volume'),
                 n_hmm_states: int = 4,
                 n_wf_folds: int = 5):

        self.lookback      = lookback
        self.ta_builder    = TechnicalIndicatorBuilder()
        self.wavelet       = WaveletDecomposer(wavelet='db4', level=4)
        self.regime        = RegimeDetector(n_states=n_hmm_states)
        self.kpca          = KPCAReducer(n_components=n_kpca_components)
        self.wf_splitter   = WalkForwardSplitter(n_folds=n_wf_folds)
        self.wavelet_cols  = list(wavelet_cols)
        self._feature_cols: List[str] = []

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _create_sequences(features, targets, dates, lookback, horizon=1):  # CHANGED: + dates param
        """Sliding-window sequences; returns X, y_reg, y_cls, decision_dates."""
        X, y_reg, y_cls, decision_dates = [], [], [], []  # CHANGED
        closes = targets
        for i in range(len(features) - lookback - horizon + 1):
            X.append(features[i: i + lookback])
            curr_price = closes[i + lookback - 1]
            future_price = closes[i + lookback + horizon - 1]
            y_reg.append(np.log(future_price / curr_price))  
            y_cls.append(1 if future_price > curr_price else 0)
            decision_dates.append(dates[i + lookback - 1])  # NEW — same index as curr_price
        return (np.array(X, dtype=np.float32),
                np.array(y_reg, dtype=np.float32).reshape(-1, 1),
                np.array(y_cls, dtype=np.int64),
                np.array(decision_dates))  # NEW

    # ── main entry point ──────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame,
                      train_end_idx: Optional[int] = None
                      ) -> Dict:
        """
        Run the full pipeline on df.

        Parameters
        ──────────
        df             : raw OHLCV dataframe with a 'timestamp' column
        train_end_idx  : row index where training data ends (for KPCA + HMM fit).
                         Defaults to 80% of data if not provided.

        Returns
        ───────
        Dict with keys:
          folds       — list of walk-forward (X_tr, y_reg_tr, y_cls_tr,
                                               X_te, y_reg_te, y_cls_te)
          feature_dim — int (should equal n_kpca_components = 120)
          df_enriched — the full dataframe with all extra columns
          pipeline    — dict of fitted objects for inference
        """
        print("\n" + "=" * 68)
        print("  APEX-ST  ADVANCED FEATURE PIPELINE")
        print("=" * 68)

        n = len(df)
        if train_end_idx is None:
            train_end_idx = int(n * 0.80)

        df_train = df.iloc[:train_end_idx].copy()

        # ── Step 1: TA indicators ────────────────────────────────────────────
        print("\n[1/5] Computing technical indicators …")
        df_full = self.ta_builder.compute(df)

        # ── Step 2: Wavelet features ─────────────────────────────────────────
        print("\n[2/5] Wavelet decomposition (Daubechies-4, 4 levels) …")
        df_full = self.wavelet.transform(df_full, columns=self.wavelet_cols)

        # ── Step 3: Regime detection (fit on train, label all) ───────────────
        print("\n[3/5] HMM regime detection …")
        df_full_ta = self.ta_builder.compute(df_train)          # need close on train
        self.regime.fit(df_full_ta)
        df_full = self.regime.transform(df_full)

        # ── Step 4: Select numeric features, drop NaN rows ──────────────────
        print("\n[4/5] KPCA dimensionality reduction …")
        # Collect all numeric columns except timestamp / raw OHLCV used as targets
        exclude = {'timestamp', 'date', 'days_since_last',
                   'symbol', 'interval', 'open', 'high', 'low', 'close', 'volume'}
        self._feature_cols = [c for c in df_full.select_dtypes(include=[np.number]).columns
                               if c not in exclude]

        df_feat = df_full[self._feature_cols].copy()

        # Forward-fill then drop leading NaN (from rolling windows)
        df_feat = df_feat.ffill().dropna()
        # Align df_full to the same index
        df_full = df_full.loc[df_feat.index]

        feature_matrix = df_feat.values                # (N, F)
        train_end_adj  = min(train_end_idx, len(feature_matrix))

        # Fit KPCA on training slice only
        self.kpca.fit(feature_matrix[:train_end_adj])
        features_reduced = self.kpca.transform(feature_matrix)   # (N, 120)

        print(f"  Feature matrix: {feature_matrix.shape} → {features_reduced.shape}")

        # ── Step 5: Create sequences ─────────────────────────────────────────
        print("\n[5/5] Creating lookback sequences …")
        close_prices = df_full['close'].values
        date_values = df_full['timestamp'].values  # NEW
        X, y_reg, y_cls, decision_dates = self._create_sequences(  # CHANGED
            features_reduced, close_prices, date_values, self.lookback  # CHANGED
        )
        print(f"  X shape: {X.shape}   (samples, {self.lookback}, {features_reduced.shape[1]})")
        print(f"  y_reg:   {y_reg.shape}   y_cls: {y_cls.shape}")

        # ── Walk-forward splits ──────────────────────────────────────────────
        folds_raw = self.wf_splitter.split(X, y_reg)
        folds = []
        for X_tr, y_tr, X_te, y_te in folds_raw:
            n_tr = len(X_tr)
            n_te = len(X_te)
            y_cls_tr = y_cls[:n_tr]
            y_cls_te = y_cls[n_tr: n_tr + n_te]
            dates_tr = decision_dates[:n_tr]  # NEW — identical slicing pattern
            dates_te = decision_dates[n_tr: n_tr + n_te]  # NEW
            folds.append({
                'X_train': X_tr, 'y_reg_train': y_tr, 'y_cls_train': y_cls_tr,
                'X_test': X_te, 'y_reg_test': y_te, 'y_cls_test': y_cls_te,
                'dates_train': dates_tr, 'dates_test': dates_te,  # NEW
            })

        print(f"\n  ✅ Pipeline complete — {len(folds)} walk-forward folds ready")
        print("=" * 68 + "\n")

        return {
            'folds':       folds,
            'feature_dim': features_reduced.shape[1],
            'df_enriched': df_full,
            'pipeline': {
                'ta_builder': self.ta_builder,
                'wavelet':    self.wavelet,
                'regime':     self.regime,
                'kpca':       self.kpca,
                'feature_cols': self._feature_cols,
            }
        }

    def transform_inference(self, df_new: pd.DataFrame) -> np.ndarray:
        """
        Apply the fitted pipeline to new data at inference time.
        Returns feature tensor of shape (N', lookback, 120).
        """
        df_new = self.ta_builder.compute(df_new)
        df_new = self.wavelet.transform(df_new, columns=self.wavelet_cols)
        df_new = self.regime.transform(df_new)

        df_feat = df_new[self._feature_cols].ffill().dropna()
        features_reduced = self.kpca.transform(df_feat.values)
        close_prices = df_new.loc[df_feat.index, 'close'].values

        X, _, _ = self._create_sequences(features_reduced, close_prices, self.lookback)
        return X

    def save(self, path: str = 'apex_feature_pipeline.pkl'):
        with open(path, 'wb') as f:
            pickle.dump({
                'regime':       self.regime,
                'kpca':         self.kpca,
                'wavelet':      self.wavelet,
                'feature_cols': self._feature_cols,
                'lookback':     self.lookback,
            }, f)
        print(f"✓ AdvancedFeaturePipeline saved → {path}")

    def load(self, path: str = 'apex_feature_pipeline.pkl') -> 'AdvancedFeaturePipeline':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.regime        = data['regime']
        self.kpca          = data['kpca']
        self.wavelet       = data['wavelet']
        self._feature_cols = data['feature_cols']
        self.lookback      = data['lookback']
        print(f"✓ AdvancedFeaturePipeline loaded ← {path}")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# 7.  SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

def _generate_mock_nse_data(n_days: int = 2500,
                            start: str = '2015-01-01') -> pd.DataFrame:
    """
    Synthetic OHLCV data that mimics an NSE stock.
    Three volatility regimes embedded:
      0-999   : calm bull (pre-COVID)
      1000-1249: crash     (COVID)
      1250+   : recovery
    """
    np.random.seed(0)
    dates = pd.bdate_range(start=start, periods=n_days)

    price = 1000.0
    prices = []
    for i in range(n_days):
        if   i < 1000:   vol = 0.010
        elif i < 1250:   vol = 0.045     # crisis
        else:            vol = 0.018

        ret = np.random.normal(0.0003, vol)
        price *= (1 + ret)
        prices.append(price)

    prices = np.array(prices)
    highs  = prices * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
    lows   = prices * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
    opens  = prices * (1 + np.random.normal(0, 0.003, n_days))
    vols   = np.random.randint(500_000, 5_000_000, n_days).astype(float)

    return pd.DataFrame({
        'timestamp': dates,
        'open':   opens,
        'high':   highs,
        'low':    lows,
        'close':  prices,
        'volume': vols,
    })


def run_smoke_test():
    print("\n" + "▓" * 68)
    print("  APEX-ST Feature Pipeline — Smoke Test")
    print("▓" * 68)

    df = _generate_mock_nse_data(n_days=2500)
    print(f"\nGenerated mock NSE data: {len(df)} trading days "
          f"({df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()})")

    # KernelPCA cannot produce more components than input features (~42 here).
    # In production, apex_synth_runner_v2.py swaps in RawCaptureKPCA and
    # applies NystroemReducer externally (out_dim=min(96, F)), bypassing this
    # limit entirely.  Use 40 here to keep the smoke test self-consistent.
    pipeline = AdvancedFeaturePipeline(
        lookback          = 60,
        n_kpca_components = 40,   # ≤ feature count; 120 used in production via Nystroem
        wavelet_cols      = ['close', 'volume'],
        n_hmm_states      = 4,
        n_wf_folds        = 5,
    )

    result = pipeline.fit_transform(df)

    print("\n─── Results ───────────────────────────────────────────────────────")
    print(f"  Feature dim:       {result['feature_dim']}  "
          f"(smoke-test KPCAReducer path; production uses Nystroem → min(96, F))")
    print(f"  Walk-forward folds: {len(result['folds'])}")
    for i, fold in enumerate(result['folds']):
        Xtr = fold['X_train']
        Xte = fold['X_test']
        print(f"    Fold {i+1}: train={Xtr.shape}  test={Xte.shape}")

    # Save & reload test
    pipeline.save('apex_feature_pipeline_test.pkl')
    p2 = AdvancedFeaturePipeline()
    p2.load('apex_feature_pipeline_test.pkl')
    X_infer = p2.transform_inference(df.tail(200).copy())
    print(f"\n  Inference check: input 200 rows → X shape {X_infer.shape}")
    print("\n  ✅ All checks passed — Sprint 1 smoke test complete")
    print("\n  ℹ  NOTE: This smoke test validates the feature pipeline classes")
    print("         on synthetic data.  To produce real .npy artifacts for")
    print("         Sprints 2–5, run:  python apex_synth_runner_v2.py")
    print("▓" * 68 + "\n")


if __name__ == '__main__':
    run_smoke_test()
