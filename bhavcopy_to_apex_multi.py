"""
bhavcopy_to_apex_multi.py — like bhavcopy_to_apex.py, but merges MULTIPLE
yearly bhavcopy CSVs (e.g. NSE_OHLCV_2010.csv ... NSE_OHLCV_2025.csv)
into one continuous per-symbol history at data/<SYMBOL>.csv, which is
what apex_synth_runner_v2.py actually expects (one file per symbol
covering its full available history, not one file per year).

Usage:
    # all NSE_OHLCV_*.csv files in the current folder:
    python bhavcopy_to_apex_multi.py --glob "NSE_OHLCV_*.csv" --out-dir data

    # explicit files, restricted to your watchlist:
    python bhavcopy_to_apex_multi.py NSE_OHLCV_2010.csv NSE_OHLCV_2011.csv \
        --symbols RELIANCE TCS INFY HDFCBANK ICICIBANK --out-dir data

    # a symbol that was renamed on NSE over the years: use CURRENT:HISTORICAL
    # to search the CSV under its old name but write output under the new one.
    # e.g. Adani Ports traded as MUNDRAPORT before its 2011-12 rename:
    python bhavcopy_to_apex_multi.py --glob "NSE_OHLCV_*.csv" \
        --symbols ADANIENT ADANIPORTS:MUNDRAPORT APOLLOHOSP --out-dir data
    # -> writes data/ADANIPORTS.csv, populated from MUNDRAPORT rows in early
    #    years and ADANIPORTS rows in later years, once the rename happened.
"""
import argparse
import glob
import sys
from pathlib import Path
import pandas as pd


def load_and_tag(path, series):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    required = {"SYMBOL", "DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"}
    missing = required - set(df.columns)
    if missing:
        print(f"  ! skipping {path}: missing columns {missing}")
        return None
    if series and "SERIES" in df.columns:
        df = df[df["SERIES"] == series]
    return df


def parse_aliases(symbol_args):
    """
    Splits '--symbols ADANIPORTS:MUNDRAPORT TCS' into:
      - lookup_names: ALL source-CSV names to search for, both the current
        name and any historical alias (e.g. both 'ADANIPORTS' and
        'MUNDRAPORT') -- a ticker rename means the symbol appears under
        the old name in earlier files and the new name in later ones, so
        both must be searched or rows after the rename get silently
        dropped (this was a real bug: an earlier version only searched
        the historical name and truncated ADANIPORTS at its 2012 rename
        date instead of continuing under the new name).
      - rename_map: {source_symbol_in_csv: desired_output_symbol}, e.g.
        {'MUNDRAPORT': 'ADANIPORTS', 'ADANIPORTS': 'ADANIPORTS'}
    Plain entries with no ':' map to themselves.
    """
    lookup_names = []
    rename_map = {}
    for entry in symbol_args:
        if ":" in entry:
            current, historical = entry.split(":", 1)
            lookup_names.append(current)
            lookup_names.append(historical)
            rename_map[current] = current
            rename_map[historical] = current
        else:
            lookup_names.append(entry)
            rename_map[entry] = entry
    return lookup_names, rename_map


def convert_many(paths, out_dir="data", symbols=None, series="EQ"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    frames = []
    for p in paths:
        df = load_and_tag(p, series)
        if df is not None:
            frames.append(df)
            print(f"  loaded {p}: {len(df)} rows")

    if not frames:
        print("ERROR: no valid files loaded"); sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined across {len(frames)} files: {len(combined)} rows total")

    rename_map = {}
    if symbols:
        lookup_names, rename_map = parse_aliases(symbols)
        combined = combined[combined["SYMBOL"].isin(lookup_names)]
        combined = combined.assign(SYMBOL=combined["SYMBOL"].map(rename_map))

    written = []
    for sym, g in combined.groupby("SYMBOL"):
        g = g.sort_values("DATE")
        out = pd.DataFrame({
            "date": g["DATE"],
            "open": g["OPEN"],
            "high": g["HIGH"],
            "low": g["LOW"],
            "close": g["CLOSE"],
            "volume": g["VOLUME"],
        })
        before = len(out)
        out = out.drop_duplicates(subset="date", keep="last")
        dupes = before - len(out)

        dest = out_path / f"{sym}.csv"
        out.to_csv(dest, index=False)
        written.append((sym, len(out), out["date"].min(), out["date"].max(), dupes))

    return written


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*", help="Explicit CSV paths (omit if using --glob)")
    p.add_argument("--glob", default=None, help='e.g. "NSE_OHLCV_*.csv"')
    p.add_argument("--out-dir", default="data")
    p.add_argument("--symbols", nargs="*", default=None,
                    help="Restrict to these symbols (default: all symbols found). "
                         "Use CURRENT:HISTORICAL for renamed tickers, e.g. "
                         "ADANIPORTS:MUNDRAPORT")
    p.add_argument("--series", default="EQ", help="NSE series to keep; '' to keep all")
    args = p.parse_args()

    paths = args.files
    if args.glob:
        paths = sorted(glob.glob(args.glob))
    if not paths:
        print("ERROR: no input files. Use positional args or --glob."); sys.exit(1)

    print(f"Found {len(paths)} source file(s):")
    for pth in paths:
        print(f"  {pth}")
    print()

    result = convert_many(paths, args.out_dir, args.symbols, args.series or None)
    print(f"\nWrote {len(result)} per-symbol files to '{args.out_dir}/':")
    for sym, n, dmin, dmax, dupes in sorted(result, key=lambda r: -r[1])[:20]:
        dupe_note = f"  ({dupes} dup dates dropped)" if dupes else ""
        print(f"  {sym:<15} {n:>5} rows  {dmin} -> {dmax}{dupe_note}")
    if len(result) > 20:
        print(f"  ... and {len(result)-20} more")
