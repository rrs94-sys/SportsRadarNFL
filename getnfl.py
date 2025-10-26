#!/usr/bin/env python3
"""
fetch_nfl_pbp_2025_nflreadpy.py

Fetch NFL play-by-play data for 2025 using nflreadpy and export to CSV/Parquet.

Features:
- Uses Polars (via nflreadpy) for performance
- CLI flags: season, weeks, output path, overwrite, show-cols
- Week-filtering applied after loading whole season
- Handles missing data / caching via nflreadpy
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import polars as pl

try:
    import nflreadpy as nfl
except ImportError:
    print(
        "Missing dependency: nflreadpy\n"
        "Install with:  pip install nflreadpy\n",
        file=sys.stderr,
    )
    sys.exit(1)


def parse_weeks(weeks_str: Optional[str]) -> Optional[List[int]]:
    """
    Parse a weeks filter string (e.g., '1,2,3' or '1:8') into a list of ints.
    """
    if not weeks_str:
        return None
    weeks_str = weeks_str.strip()
    if ":" in weeks_str:
        a, b = weeks_str.split(":")
        start, end = int(a), int(b)
        if start > end:
            start, end = end, start
        return list(range(start, end + 1))
    parts = [int(x.strip()) for x in weeks_str.split(",") if x.strip()]
    return sorted(set(parts))


def fetch_pbp(season: int) -> pl.DataFrame:
    """
    Fetch play-by-play data for a given season using nflreadpy.
    """
    print(f"[info] Fetching play-by-play for season {season} via nflreadpy...")
    df: pl.DataFrame = nfl.load_pbp(season)  # defaults to full season if single int
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch 2025 NFL play-by-play using nflreadpy.")
    parser.add_argument("--season", type=int, default=2025,
                        help="Season year to fetch (default: 2025).")
    parser.add_argument("--weeks", type=str, default=None,
                        help="Weeks filter, e.g. '1,2,3' or '1:8'. Applied AFTER download.")
    parser.add_argument("--outfile", type=str, default="pbp_2025.parquet",
                        help="Output file path (.parquet or .csv). Default: pbp_2025.parquet")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwrite of existing outfile.")
    parser.add_argument("--show-cols", action="store_true",
                        help="Print a sample of important columns and exit.")
    args = parser.parse_args()

    out_path = Path(args.outfile)
    if out_path.exists() and not args.overwrite:
        print(f"[error] Output file exists: {out_path}. Use --overwrite to replace.", file=sys.stderr)
        sys.exit(2)

    # Fetch the data
    df = fetch_pbp(args.season)

    # Convert to pandas if you prefer (optional)
    # df_pd = df.to_pandas()

    # Filter by weeks if requested
    weeks_list = parse_weeks(args.weeks)
    if weeks_list:
        before_count = df.height
        df = df.filter(pl.col("week").is_in(weeks_list))
        after_count = df.height
        print(f"[info] Filtered to weeks {weeks_list} → {after_count:,} rows (from {before_count:,}).")

    if args.show_cols:
        sample_cols = ["season", "season_type", "week", "game_id",
                       "home_team", "away_team", "posteam", "defteam",
                       "qtr", "down", "ydstogo", "yardline_100",
                       "play_type", "pass_length", "rush_attempt", "pass_attempt",
                       "epa", "yards_gained", "score_differential"]
        present = [c for c in sample_cols if c in df.columns]
        print(f"[info] Present columns ({len(present)}): {present}")
        # Continue to save anyway

    # Write output
    out_ext = out_path.suffix.lower()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_ext == ".parquet":
        df.write_parquet(out_path)
    elif out_ext == ".csv":
        df.write_csv(out_path)
    else:
        print(f"[warn] Unknown extension '{out_ext}', defaulting to Parquet (.parquet).")
        out_path = out_path.with_suffix(".parquet")
        df.write_parquet(out_path)

    print(f"[done] Saved {df.height:,} rows → {out_path.resolve()}")


if __name__ == "__main__":
    main()
