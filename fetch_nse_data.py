"""
fetch_nse_data.py — NSE OHLCV Data Fetcher (via yfinance)
═══════════════════════════════════════════════════════════
Pulls real NSE price history for any list of stocks + the Nifty 50 index,
and writes them into the data/ folder in the exact CSV format that
stock_screener.py and apex_synth_runner_v2.py expect.

INSTALL (once):
  pip install yfinance pandas

USAGE
─────
  # Fetch default watchlist  (Nifty 50 + top-20 Nifty stocks)
  python fetch_nse_data.py

  # Fetch specific symbols
  python fetch_nse_data.py --symbols RELIANCE TCS HDFCBANK INFY ICICIBANK

  # Control history length (default 5 years)
  python fetch_nse_data.py --period 3y
  python fetch_nse_data.py --start 2019-01-01 --end 2024-12-31

  # Fetch + immediately run the screener
  python fetch_nse_data.py --screen

  # Full pipeline: fetch → screen → train APEX-ST on shortlist
  python fetch_nse_data.py --screen --apex

HOW YAHOO FINANCE MAPS NSE SYMBOLS
───────────────────────────────────
  Stock  →  <SYMBOL>.NS   (e.g. RELIANCE.NS, TCS.NS)
  Index  →  ^NSEI          (Nifty 50)

OUTPUTS
───────
  data/NIFTY50.csv        benchmark index
  data/<SYMBOL>.csv       one file per stock, columns: date,open,high,low,close,volume
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

DATA_DIR = Path('data')

# ─────────────────────────────────────────────────────────────────────────────
# BROWSER-IMPERSONATION SESSION (curl_cffi)
# ─────────────────────────────────────────────────────────────────────────────
# yfinance talks to Yahoo's undocumented chart API directly. Yahoo fingerprints
# the TLS/HTTP2 handshake of that traffic and rate-limits anything that looks
# like a bare Python `requests`/`urllib3` client — which is what bare yfinance
# uses under the hood. curl_cffi reproduces a real Chrome TLS fingerprint, so
# requests routed through it are far less likely to get silently throttled
# (the failure mode is a 200 response with an empty/invalid body, which is
# what surfaces as "Expecting value: line 1 column 1 (char 0)").
#
# One session is created and reused across every ticker in a run — creating a
# fresh impersonated session per request adds overhead and gives Yahoo more
# distinct connection fingerprints to flag, working against the goal.
def build_session():
    """
    Returns a curl_cffi Session configured to impersonate Chrome, or None if
    curl_cffi isn't installed. Callers must handle the None case by falling
    back to yfinance's default (un-impersonated) transport.
    """
    try:
        from curl_cffi import requests as cffi_requests
    except ImportError:
        print("  ⚠  curl_cffi not installed — falling back to yfinance's "
              "default session (more likely to be rate-limited).")
        print("      Install with: pip install curl_cffi")
        return None

    try:
        session = cffi_requests.Session(impersonate="chrome")
        return session
    except Exception as e:
        print(f"  ⚠  curl_cffi session creation failed ({e}) — falling back "
              f"to yfinance's default session.")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT SYMBOL LISTS
# ─────────────────────────────────────────────────────────────────────────────

# Top-20 Nifty 50 constituents by weight (as of mid-2025)
NIFTY_TOP20 = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'BHARTIARTL', 'ICICIBANK',
    'INFY',     'SBIN', 'HINDUNILVR', 'ITC',       'LT',
    'KOTAKBANK','AXISBANK','BAJFINANCE','ASIANPAINT','MARUTI',
    'HCLTECH',  'SUNPHARMA','TITAN',    'ULTRACEMCO','WIPRO',
]

# Broader Nifty 100 coverage — useful for a wider screener
NIFTY_100_EXTRA = [
    'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'BAJAJFINSV', 'BAJAJ-AUTO',
    'BPCL',     'BRITANNIA',  'CIPLA',      'COALINDIA',  'DIVISLAB',
    'DRREDDY',  'EICHERMOT',  'GRASIM',     'HEROMOTOCO', 'HINDALCO',
    'INDUSINDBK','JSWSTEEL',  'M&M',        'NESTLEIND',  'NTPC',
    'ONGC',     'POWERGRID',  'SBILIFE',    'SHRIRAMFIN', 'TATAMOTORS',
    'TATASTEEL','TECHM',      'TRENT',      'TATACONSUM', 'VEDL',
]

# Maps our clean symbol names to Yahoo Finance tickers
def to_yf_ticker(symbol: str) -> str:
    """Convert NSE symbol → Yahoo Finance ticker."""
    overrides = {
        'M&M':     'M%26M.NS',      # & needs encoding in some contexts
        'BAJAJ-AUTO': 'BAJAJ-AUTO.NS',
    }
    return overrides.get(symbol, f'{symbol}.NS')


# ─────────────────────────────────────────────────────────────────────────────
# FETCH + SAVE
# ─────────────────────────────────────────────────────────────────────────────
def fetch_symbol(yf_ticker: str, symbol_name: str,
                 period: str = None, start: str = None, end: str = None,
                 min_rows: int = 100, max_retries: int = 3,
                 session=None) -> bool:
    """
    Download OHLCV for one ticker and save to data/<symbol_name>.csv.
    Returns True on success.

    Retries with exponential backoff on failure, since Yahoo Finance
    rate-limits aggressively — a burst of requests (e.g. 50+ symbols
    back-to-back) commonly trips a temporary block where EVERY request,
    including ones for valid tickers, returns an empty/invalid response
    (manifests as "Expecting value: line 1 column 1 (char 0)").

    `session`, if provided, should be a curl_cffi Session impersonating a
    real browser's TLS fingerprint (see build_session()) — this is passed
    straight through to yf.Ticker() and substantially reduces how often
    that rate-limit trips in the first place. If None, yfinance falls back
    to its own default transport.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  ❌ yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            ticker = yf.Ticker(yf_ticker, session=session) if session is not None \
                     else yf.Ticker(yf_ticker)
            if start:
                hist = ticker.history(start=start, end=end, auto_adjust=True)
            else:
                hist = ticker.history(period=period or '5y', auto_adjust=True)

            if hist is None or len(hist) < min_rows:
                if attempt < max_retries:
                    wait = 5 * attempt
                    print(f"  ⏳  {symbol_name:<16} attempt {attempt}/{max_retries} "
                          f"got {len(hist) if hist is not None else 0} rows — "
                          f"retrying in {wait}s")
                    time.sleep(wait)
                    continue
                print(f"  ⚠  {symbol_name:<16} only "
                      f"{len(hist) if hist is not None else 0} rows after "
                      f"{max_retries} attempts — skipped")
                return False

            # Normalise column names + index
            df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = 'date'
            df = df.reset_index()
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            df = df.sort_values('date').reset_index(drop=True)
            df = df.dropna(subset=['close'])

            out = DATA_DIR / f'{symbol_name}.csv'
            df.to_csv(out, index=False)
            print(f"  ✅  {symbol_name:<16} {len(df):>5} rows  "
                  f"{df['date'].iloc[0]} → {df['date'].iloc[-1]}")
            return True

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = 5 * attempt
                print(f"  ⏳  {symbol_name:<16} attempt {attempt}/{max_retries} "
                      f"error: {str(e)[:60]} — retrying in {wait}s")
                time.sleep(wait)
            else:
                print(f"  ❌  {symbol_name:<16} error after {max_retries} "
                      f"attempts: {e}")
                return False
    return False


def fetch_all(symbols: list, period: str = None,
              start: str = None, end: str = None,
              delay: float = 2.0) -> dict:
    """Fetch benchmark + all candidate symbols. Returns {symbol: ok}."""
    DATA_DIR.mkdir(exist_ok=True)
    results = {}

    session = build_session()
    if session is not None:
        print("  🛡  Using curl_cffi (Chrome-impersonated) session for all requests")

    print(f"\n[1/{len(symbols)+1}] Fetching Nifty 50 benchmark (^NSEI)…")
    results['NIFTY50'] = fetch_symbol('^NSEI', 'NIFTY50', period, start, end,
                                       session=session)

    consecutive_failures = 0 if results['NIFTY50'] else 1

    for i, sym in enumerate(symbols, start=2):
        # Early-abort: if the benchmark AND the first 3 symbols all fail,
        # this is almost certainly Yahoo rate-limiting the whole session,
        # not a problem with individual tickers. Stop burning requests.
        if consecutive_failures >= 4:
            remaining = symbols[i-2:]
            print(f"\n  🛑  {consecutive_failures} consecutive failures — "
                  f"this looks like a Yahoo Finance rate limit, not a "
                  f"per-ticker problem.")
            print(f"      Stopping early to avoid making it worse. "
                  f"{len(remaining)} symbols not attempted: "
                  f"{', '.join(remaining[:5])}{'...' if len(remaining) > 5 else ''}")
            print(f"      Wait 10-15 minutes, then retry with a longer "
                  f"--delay (e.g. --delay 3) and fewer symbols at once.")
            for sym_remaining in remaining:
                results[sym_remaining] = False
            break

        print(f"[{i}/{len(symbols)+1}] {sym}…", end='  ')
        yf_sym = to_yf_ticker(sym)
        ok = fetch_symbol(yf_sym, sym, period, start, end, session=session)
        results[sym] = ok
        consecutive_failures = 0 if ok else consecutive_failures + 1
        time.sleep(delay)   # polite pacing — avoids rate-limit blocks

    return results


# ─────────────────────────────────────────────────────────────────────────────
# FUNDAMENTALS TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
def write_fundamentals_template(symbols: list):
    """
    Writes a fundamentals_template.csv pre-filled with the symbol names
    and blank values. User fills it from Screener.in and renames it to
    fundamentals.csv.
    """
    path = DATA_DIR / 'fundamentals_template.csv'
    rows = [
        {
            'symbol':                   sym,
            'quarterly_profit_yoy_pct': '',   # from Screener.in → Quarterly Results
            'annual_profit_growth_pct': '',   # from Screener.in → Annual Results
            'sales_growth_pct':         '',   # from Screener.in → Profit & Loss
            'market_cap_cr':            '',   # top of Screener.in company page
        }
        for sym in symbols
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\n✓ Fundamentals template → {path}")
    print("  Fill values from Screener.in, then rename to fundamentals.csv")
    print("  Columns:")
    print("    quarterly_profit_yoy_pct  — latest quarter profit YoY % change")
    print("    annual_profit_growth_pct  — 3-year CAGR of net profit")
    print("    sales_growth_pct          — 3-year CAGR of revenue")
    print("    market_cap_cr             — market cap in ₹ Crore")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='NSE OHLCV fetcher for stock_screener.py')
    p.add_argument('--symbols', nargs='*',
                   help='NSE symbols to fetch (default: Nifty top-20)')
    p.add_argument('--nifty100', action='store_true',
                   help='Fetch all ~50 Nifty 100 stocks (top-20 + extra)')
    p.add_argument('--period', default='5y',
                   help='History length: 1y 2y 3y 5y 10y max (default: 5y)')
    p.add_argument('--start',  default=None,
                   help='Start date YYYY-MM-DD (overrides --period)')
    p.add_argument('--end',    default=None,
                   help='End date YYYY-MM-DD (default: today)')
    p.add_argument('--delay',  type=float, default=2.0,
                   help='Seconds between requests (default: 2.0 — Yahoo '
                        'rate-limits aggressively above ~20 symbols; '
                        'increase if you still hit failures)')
    p.add_argument('--screen', action='store_true',
                   help='Run stock_screener.py after fetching')
    p.add_argument('--rs-high-pct', type=float, default=10.0,
                   help='Passed to screener --rs-high-pct (default: 10.0)')
    p.add_argument('--min-sales-growth', type=float, default=10.0,
                   help='Passed to screener --min-sales-growth (default: 10.0)')
    p.add_argument('--cap-tier', default='any',
                   choices=['large','mid','small','any'],
                   help='Passed to screener --cap-tier (default: any)')
    p.add_argument('--apex', action='store_true',
                   help='Run apex_synth_runner_v2.py on shortlist after screening')
    p.add_argument('--skip-fundamentals', action='store_true',
                   help='Pass --skip-fundamentals to screener (RS only)')
    return p.parse_args()


def main():
    args = parse_args()

    # Build symbol list
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    elif args.nifty100:
        symbols = NIFTY_TOP20 + NIFTY_100_EXTRA
    else:
        symbols = NIFTY_TOP20

    print("█" * 70)
    print("  NSE DATA FETCHER")
    print(f"  Symbols  : {len(symbols)} stocks + Nifty 50 benchmark")
    print(f"  History  : {args.start or args.period}"
          + (f" → {args.end}" if args.end else ""))
    print(f"  Output   : {DATA_DIR}/")
    print("█" * 70)

    results = fetch_all(symbols, period=args.period,
                        start=args.start, end=args.end, delay=args.delay)

    ok     = [s for s, v in results.items() if v and s != 'NIFTY50']
    failed = [s for s, v in results.items() if not v and s != 'NIFTY50']

    print(f"\n{'═'*70}")
    print(f"  FETCH SUMMARY")
    print(f"{'═'*70}")
    print(f"  ✅ Success : {len(ok)} symbols")
    if failed:
        print(f"  ❌ Failed  : {', '.join(failed)}")

    write_fundamentals_template(ok)

    if not results.get('NIFTY50'):
        print("\n  ⚠  NIFTY50 benchmark failed — screener cannot run without it.")
        sys.exit(1)

    if args.screen:
        import subprocess, shlex
        screen_cmd = [
            sys.executable, 'stock_screener.py',
            '--rs-high-pct',       str(args.rs_high_pct),
            '--min-sales-growth',  str(args.min_sales_growth),
            '--cap-tier',          args.cap_tier,
        ]
        if args.skip_fundamentals:
            screen_cmd.append('--skip-fundamentals')

        print(f"\n{'═'*70}")
        print(f"  RUNNING SCREENER")
        print(f"  {' '.join(screen_cmd)}")
        print(f"{'═'*70}\n")
        result = subprocess.run(screen_cmd)

        if args.apex and result.returncode == 0:
            import json
            wl_path = Path('watchlist.json')
            if not wl_path.exists():
                print("  ⚠  watchlist.json not found — skipping APEX-ST")
            else:
                shortlist = json.loads(wl_path.read_text()).get('watchlist', [])
                if not shortlist:
                    print("  ⚠  Empty shortlist — nothing to train on")
                else:
                    apex_cmd = [
                        sys.executable, 'apex_synth_runner_v2.py',
                    ] + shortlist
                    print(f"\n{'═'*70}")
                    print(f"  RUNNING APEX-ST on shortlist: {shortlist}")
                    print(f"{'═'*70}\n")
                    subprocess.run(apex_cmd)

    print(f"\n{'═'*70}")
    print("  NEXT STEPS")
    print(f"{'═'*70}")
    if not args.screen:
        print(f"  1. Fill fundamentals data:")
        print(f"       Edit data/fundamentals_template.csv from Screener.in")
        print(f"       Rename to data/fundamentals.csv")
        print(f"  2. Run screener:")
        print(f"       python stock_screener.py --rs-high-pct 10")
        print(f"     Or with RS only (skip fundamentals):")
        print(f"       python stock_screener.py --skip-fundamentals")
    print(f"  3. Run APEX-ST on qualified symbols:")
    print(f"       python apex_synth_runner_v2.py  (reads watchlist.json)")
    print(f"{'═'*70}")


if __name__ == '__main__':
    main()
