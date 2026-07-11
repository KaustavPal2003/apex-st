"""
stock_screener.py — Relative Strength + Fundamentals Screener (CSV-based)
═══════════════════════════════════════════════════════════════════════════
Implements the two-stage screening plan:

  Stage 1  Relative Strength vs Nifty 50  (replaces TradingView "Bharat Trader" RS)
  Stage 2  Fundamentals filter            (replaces manual Screener.in lookup)

Output: watchlist.json  →  drop-in SYMBOLS list for apex_synth_runner_v2.py

────────────────────────────────────────────────────────────────────────────
INPUT FILES (all CSV, OHLCV format: date,open,high,low,close,volume)
────────────────────────────────────────────────────────────────────────────
  data/NIFTY50.csv          benchmark index
  data/<SYMBOL>.csv         one file per candidate stock (e.g. data/RELIANCE.csv)

  Fundamentals (optional, for Stage 2):
  data/fundamentals.csv     columns: symbol,quarterly_profit_yoy_pct,
                                      annual_profit_growth_pct,
                                      sales_growth_pct,market_cap_cr

If fundamentals.csv is absent, Stage 2 is skipped and the watchlist is based
on relative strength alone (with a note in the output).

────────────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────────────
  python stock_screener.py                       # screen everything in data/
  python stock_screener.py --symbols RELIANCE TCS INFY
  python stock_screener.py --rs-window 63        # ~3-month RS lookback
  python stock_screener.py --min-sales-growth 10 --cap-tier large
  python stock_screener.py --demo                # generate synthetic CSVs first

────────────────────────────────────────────────────────────────────────────
SELECTION RULES  (mirrors the manual checklist)
────────────────────────────────────────────────────────────────────────────
  ✓ RS line trending upward            (RS_now > RS_{window} days ago)
  ✓ RS near 52-week high                (RS_now within --rs-high-pct of 252d max)
  ✓ Quarterly profit growing YoY        (> 0%)
  ✓ Annual profit growth (3-5y)         (> 0%)
  ✓ Sales growth                        (> --min-sales-growth, default 10%)
  ✓ Market cap matches risk tier        (--cap-tier: large / mid / small / any)

A symbol passing ALL active checks is added to watchlist.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path('data')
OUT_FILE = Path('watchlist.json')

CAP_TIERS = {
    'large': (20000, float('inf')),
    'mid':   (5000, 20000),
    'small': (0, 5000),
    'any':   (0, float('inf')),
}


# ─────────────────────────────────────────────────────────────────────────────
# DEMO DATA  — synthetic OHLCV + fundamentals so the script runs standalone
# ─────────────────────────────────────────────────────────────────────────────
def make_demo_data(n_days: int = 600):
    """Generates Nifty50 + 6 sample stocks (mix of outperformers/laggards)
    and a fundamentals.csv, all under data/."""
    DATA_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    dates = pd.bdate_range('2023-01-02', periods=n_days)

    def gbm(start, mu, vol, seed):
        r = np.random.default_rng(seed)
        rets = r.normal(mu, vol, n_days)
        price = start * np.cumprod(1 + rets)
        return price.astype(np.float32)

    def to_ohlcv(close, seed):
        r = np.random.default_rng(seed)
        o = close * (1 + r.normal(0, 0.003, len(close)))
        h = np.maximum(o, close) * (1 + np.abs(r.normal(0, 0.004, len(close))))
        l = np.minimum(o, close) * (1 - np.abs(r.normal(0, 0.004, len(close))))
        v = r.normal(1_000_000, 200_000, len(close)).clip(min=50_000)
        return pd.DataFrame({'date': dates, 'open': o, 'high': h, 'low': l,
                             'close': close, 'volume': v})

    # Benchmark: steady ~15% annualised drift
    nifty = gbm(22000, 0.0006, 0.009, seed=1)
    to_ohlcv(nifty, 100).to_csv(DATA_DIR / 'NIFTY50.csv', index=False)

    # Candidates: mix of strong outperformers, laggards, and a crash story
    profiles = {
        'STRONGCO':  dict(start=1000, mu=0.0014, vol=0.012, seed=11),  # clear outperformer
        'STEADYBNK': dict(start=800,  mu=0.0009, vol=0.010, seed=12),  # mild outperformer
        'LAGCO':     dict(start=1200, mu=0.0002, vol=0.011, seed=13),  # underperformer
        'CRASHCO':   dict(start=2000, mu=-0.0008, vol=0.020, seed=14), # declining RS
        'SMALLCAP1': dict(start=300,  mu=0.0016, vol=0.018, seed=15),  # small-cap riser
        'FLATCO':    dict(start=1500, mu=0.0005, vol=0.009, seed=16),  # tracks index
    }
    for sym, p in profiles.items():
        close = gbm(p['start'], p['mu'], p['vol'], p['seed'])
        to_ohlcv(close, p['seed'] + 100).to_csv(DATA_DIR / f'{sym}.csv', index=False)

    # Fundamentals — designed so STRONGCO/STEADYBNK/SMALLCAP1 pass, others fail
    fund = pd.DataFrame([
        dict(symbol='STRONGCO',  quarterly_profit_yoy_pct=18.5, annual_profit_growth_pct=22.0, sales_growth_pct=16.0, market_cap_cr=85000),
        dict(symbol='STEADYBNK', quarterly_profit_yoy_pct=12.0, annual_profit_growth_pct=14.5, sales_growth_pct=11.0, market_cap_cr=320000),
        dict(symbol='LAGCO',     quarterly_profit_yoy_pct=-3.0, annual_profit_growth_pct=2.0,  sales_growth_pct=4.0,  market_cap_cr=45000),
        dict(symbol='CRASHCO',   quarterly_profit_yoy_pct=-15.0,annual_profit_growth_pct=-8.0, sales_growth_pct=-2.0, market_cap_cr=60000),
        dict(symbol='SMALLCAP1', quarterly_profit_yoy_pct=24.0, annual_profit_growth_pct=28.0, sales_growth_pct=19.0, market_cap_cr=3200),
        dict(symbol='FLATCO',    quarterly_profit_yoy_pct=5.0,  annual_profit_growth_pct=6.0,  sales_growth_pct=8.0,  market_cap_cr=15000),
    ])
    fund.to_csv(DATA_DIR / 'fundamentals.csv', index=False)
    print(f"✓ Demo data written to {DATA_DIR}/  "
          f"(NIFTY50 + {len(profiles)} stocks + fundamentals.csv)")
    print(f"  ⚠  Delete demo CSVs (CRASHCO, FLATCO etc.) before adding real NSE data")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — RELATIVE STRENGTH
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _has_date_column(path: Path) -> bool:
    """Returns True only if the CSV has a 'date' column — filters out
    fundamentals_template.csv and any other non-OHLCV files."""
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        return 'date' in [c.lower() for c in header]
    except Exception:
        return False


def load_close(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    # Normalise column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df.set_index('date')['close']


def relative_strength(stock_close: pd.Series, bench_close: pd.Series) -> pd.Series:
    """RS = stock / benchmark, aligned on common dates."""
    df = pd.concat([stock_close.rename('s'), bench_close.rename('b')], axis=1).dropna()
    return (df['s'] / df['b']).rename('rs')


def rs_signal(rs: pd.Series, window: int, high_window: int, high_pct: float):
    """
    Returns dict with:
      rs_now, rs_then, rs_trending_up, rs_52w_high, rs_near_high
    """
    if len(rs) < max(window, high_window) + 1:
        return None
    rs_now  = float(rs.iloc[-1])
    rs_then = float(rs.iloc[-1 - window])
    rs_52w  = rs.iloc[-high_window:]
    rs_max  = float(rs_52w.max())
    return {
        'rs_now':         rs_now,
        'rs_change_pct':  (rs_now / rs_then - 1) * 100,
        'rs_trending_up': rs_now > rs_then,
        'rs_52w_high':    rs_max,
        'rs_pct_of_high': rs_now / rs_max * 100,
        'rs_near_high':   rs_now >= rs_max * (1 - high_pct / 100),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — FUNDAMENTALS
# ─────────────────────────────────────────────────────────────────────────────
_NUMERIC_FUND_COLS = [
    'quarterly_profit_yoy_pct',
    'annual_profit_growth_pct',
    'sales_growth_pct',
    'market_cap_cr',
]


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Strip non-numeric characters (%, Cr, commas, spaces) and convert to float.
    Safe to run on already-clean numeric columns too."""
    import re

    def parse(v):
        if isinstance(v, str):
            cleaned = re.sub(r'[^\d.\-]', '', v.strip())
            return float(cleaned) if cleaned else float('nan')
        return float(v) if pd.notna(v) else float('nan')
    return series.apply(parse)


def load_fundamentals(path: Path) -> pd.DataFrame:
    if not path.exists():
        return None
    df = pd.read_csv(path).set_index('symbol')
    for col in _NUMERIC_FUND_COLS:
        if col in df.columns:
            df[col] = _clean_numeric(df[col])
    return df


def fundamentals_signal(row: pd.Series, min_sales_growth: float, cap_tier: str):
    lo, hi = CAP_TIERS[cap_tier]
    cap = float(row['market_cap_cr'])
    return {
        'quarterly_profit_growing': row['quarterly_profit_yoy_pct'] > 0,
        'annual_profit_growing':    row['annual_profit_growth_pct'] > 0,
        'sales_growth_ok':          row['sales_growth_pct'] > min_sales_growth,
        'sales_growth_pct':         float(row['sales_growth_pct']),
        'quarterly_profit_yoy_pct': float(row['quarterly_profit_yoy_pct']),
        'annual_profit_growth_pct': float(row['annual_profit_growth_pct']),
        'market_cap_cr':            cap,
        'cap_tier_match':           lo <= cap < hi,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCREENING
# ─────────────────────────────────────────────────────────────────────────────
def screen(symbols, rs_window, rs_high_window, rs_high_pct,
           min_sales_growth, cap_tier, skip_fundamentals=False,
           fundamentals_path=None):

    bench_path = DATA_DIR / 'NIFTY50.csv'
    if not bench_path.exists():
        print(f"❌ Benchmark file missing: {bench_path}")
        print("   Run with --demo to generate sample data first.")
        return None
    bench_close = load_close(bench_path)

    _fund_path = Path(fundamentals_path) if fundamentals_path else DATA_DIR / 'fundamentals.csv'
    fund_df = None if skip_fundamentals else load_fundamentals(_fund_path)
    if fund_df is None and not skip_fundamentals:
        print("ℹ  fundamentals.csv not found — Stage 2 skipped "
              "(watchlist based on Relative Strength only)\n")

    results = []
    for sym in symbols:
        path = DATA_DIR / f'{sym}.csv'
        if not path.exists():
            print(f"  ⚠  {sym:<12} no CSV found, skipping")
            continue

        stock_close = load_close(path)
        rs = relative_strength(stock_close, bench_close)
        rs_res = rs_signal(rs, rs_window, rs_high_window, rs_high_pct)
        if rs_res is None:
            print(f"  ⚠  {sym:<12} insufficient history (<{max(rs_window, rs_high_window)} days)")
            continue

        row = {'symbol': sym, **rs_res}
        stage1_pass = rs_res['rs_trending_up'] and rs_res['rs_near_high']
        row['stage1_pass'] = stage1_pass

        if fund_df is not None and sym in fund_df.index:
            f_res = fundamentals_signal(fund_df.loc[sym], min_sales_growth, cap_tier)
            row.update(f_res)
            stage2_pass = bool(f_res['quarterly_profit_growing'] and
                          f_res['annual_profit_growing'] and
                          f_res['sales_growth_ok'] and
                          f_res['cap_tier_match'])
            row['stage2_pass'] = stage2_pass
        elif fund_df is not None:
            # Symbol passed Stage 1 but has no row in the fundamentals CSV.
            # Treat as pass-through (same as --skip-fundamentals for this
            # symbol) rather than silently disqualifying it.
            row['stage2_pass'] = None  # None = no data, not a failure
        else:
            row['stage2_pass'] = True  # fundamentals stage skipped entirely

        # qualifies: stage1 must pass; stage2 must pass OR have no data
        row['qualifies'] = bool(stage1_pass and (row['stage2_pass'] is not False))
        results.append(row)

    return pd.DataFrame(results)


def print_report(df: pd.DataFrame, rs_window: int, min_sales_growth: float, cap_tier: str):
    print("\n" + "═" * 78)
    print("  STAGE 1 — RELATIVE STRENGTH vs NIFTY 50")
    print("═" * 78)
    print(f"  {'Symbol':<12} {'RS now':>9} {'Δ vs ' + str(rs_window) + 'd':>9} "
          f"{'% of 52w high':>14}  {'Trend':<6} {'Near high':<10}")
    print(f"  {'-'*12} {'-'*9} {'-'*9} {'-'*14}  {'-'*6} {'-'*10}")
    for _, r in df.iterrows():
        trend = '↑' if r['rs_trending_up'] else '↓'
        near  = '✓' if r['rs_near_high'] else '✗'
        print(f"  {r['symbol']:<12} {r['rs_now']:>9.4f} {r['rs_change_pct']:>+8.2f}% "
              f"{r['rs_pct_of_high']:>13.1f}%  {trend:<6} {near:<10}")

    if 'sales_growth_pct' in df.columns:
        print("\n" + "═" * 78)
        print("  STAGE 2 — FUNDAMENTALS  "
              f"(sales growth > {min_sales_growth}%, cap tier = {cap_tier})")
        print("═" * 78)
        print(f"  {'Symbol':<12} {'Qtr YoY':>9} {'Annual':>9} {'Sales gr':>9} "
              f"{'Mkt cap (Cr)':>13}  {'Cap OK':<7} {'Pass':<5}")
        print(f"  {'-'*12} {'-'*9} {'-'*9} {'-'*9} {'-'*13}  {'-'*7} {'-'*5}")
        for _, r in df.iterrows():
            if pd.isna(r.get('sales_growth_pct')):
                qualifies_s = '✓' if r.get('stage1_pass') else '✗'
                print(f"  {r['symbol']:<12}  (no fundamentals data — "
                      f"included if RS passed: {qualifies_s})")
                continue
            cap_ok = '✓' if r['cap_tier_match'] else '✗'
            pas    = '✓' if r['stage2_pass'] else '✗'
            print(f"  {r['symbol']:<12} {r['quarterly_profit_yoy_pct']:>+8.1f}% "
                  f"{r['annual_profit_growth_pct']:>+8.1f}% {r['sales_growth_pct']:>+8.1f}% "
                  f"{r['market_cap_cr']:>13,.0f}  {cap_ok:<7} {pas:<5}")

    print("\n" + "═" * 78)
    print("  FINAL SHORTLIST")
    print("═" * 78)
    qualified = df[df['qualifies']]
    if len(qualified) == 0:
        print("  (none — no symbol passed all active checks)")
    else:
        for _, r in qualified.iterrows():
            print(f"  ✅  {r['symbol']}")
    print("═" * 78)


def main():
    p = argparse.ArgumentParser(description='Relative Strength + Fundamentals screener')
    p.add_argument('--symbols', nargs='*', default=None,
                   help='Symbols to screen (default: all CSVs in data/ except NIFTY50/fundamentals)')
    p.add_argument('--rs-window', type=int, default=63,
                   help='Lookback for RS trend check, in trading days (default: 63 ≈ 3 months)')
    p.add_argument('--rs-high-window', type=int, default=252,
                   help='Window for 52-week RS high (default: 252)')
    p.add_argument('--rs-high-pct', type=float, default=10.0,
                   help='RS counted "near 52w high" if within this %% (default: 10.0)')
    p.add_argument('--min-sales-growth', type=float, default=10.0,
                   help='Minimum sales growth %% (default: 10.0)')
    p.add_argument('--cap-tier', choices=list(CAP_TIERS), default='any',
                   help='Market cap tier filter (default: any)')
    p.add_argument('--fundamentals', default=None, metavar='PATH',
                   help='Path to fundamentals CSV '
                        '(default: data/fundamentals.csv)')
    p.add_argument('--skip-fundamentals', action='store_true',
                   help='Screen on Relative Strength only')
    p.add_argument('--demo', action='store_true',
                   help='Generate synthetic demo CSVs into data/ before screening')
    p.add_argument('--out', default=str(OUT_FILE), help='Output watchlist JSON path')
    args = p.parse_args()

    if args.demo:
        make_demo_data()

    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        if not DATA_DIR.exists():
            print(f"❌ {DATA_DIR}/ not found. Run with --demo to create sample data.")
            sys.exit(1)
        _DEMO_FILES = {'CRASHCO', 'FLATCO', 'LAGCO', 'SMALLCAP1', 'STEADYBNK', 'STRONGCO'}
        _EXCLUDE = {'NIFTY50', 'FUNDAMENTALS', 'FUNDAMENTALS_TEMPLATE'} | _DEMO_FILES
        symbols = sorted(
            f.stem for f in DATA_DIR.glob('*.csv')
            if f.stem.upper() not in _EXCLUDE
            and _has_date_column(f)
        )
        if not symbols:
            print(f"❌ No candidate CSVs found in {DATA_DIR}/.")
            sys.exit(1)

    print("█" * 78)
    print("  STOCK SCREENER — Relative Strength + Fundamentals")
    print(f"  Candidates : {', '.join(symbols)}")
    print(f"  RS window  : {args.rs_window}d  |  52w high window: {args.rs_high_window}d  "
          f"|  near-high tolerance: {args.rs_high_pct}%")
    if not args.skip_fundamentals:
        print(f"  Fund. min sales growth: {args.min_sales_growth}%  |  cap tier: {args.cap_tier}")
    print("█" * 78)

    df = screen(symbols, args.rs_window, args.rs_high_window, args.rs_high_pct,
                args.min_sales_growth, args.cap_tier, args.skip_fundamentals,
                fundamentals_path=args.fundamentals)
    if df is None or len(df) == 0:
        print("\n❌ No results — check your data/ directory.")
        sys.exit(1)

    print_report(df, args.rs_window, args.min_sales_growth, args.cap_tier)

    watchlist = sorted(df[df['qualifies']]['symbol'].tolist())
    out = {
        'watchlist': watchlist,
        'generated_from': symbols,
        'params': {
            'rs_window': args.rs_window,
            'rs_high_window': args.rs_high_window,
            'rs_high_pct': args.rs_high_pct,
            'min_sales_growth': args.min_sales_growth,
            'cap_tier': args.cap_tier,
            'fundamentals_used': not args.skip_fundamentals,
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n✓ Watchlist written → {args.out}")
    print(f"  {len(watchlist)} symbol(s): {watchlist}")
    print(f"\n  Use in apex_synth_runner_v2.py:")
    print(f"    import json")
    print(f"    SYMBOLS = json.load(open('{args.out}'))['watchlist']")


if __name__ == '__main__':
    main()
