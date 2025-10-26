#!/usr/bin/env python3
"""
tank01_injuries_snapshot.py
Fetch ALL current NFL injuries from Tank01 by walking team rosters and
aggregating injury-related fields into a single CSV.

What it does
- /getNFLTeams → list team abbreviations
- /getNFLTeamRoster?teamAbv=XXX → pull current roster (injury info lives here)
- Extracts flexible injury keys (Tank01 fields vary), normalizes to tidy columns
- Outputs a timestamped CSV (default: nfl_injuries_YYYYMMDD_HHMM.csv)

Requirements
  pip install requests pandas python-dateutil

Environment (recommended)
  export RAPIDAPI_KEY="your_rapidapi_key"
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional

import requests
import pandas as pd
from dateutil import tz

RAPID_HOST = "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
GET_TEAMS = f"https://{RAPID_HOST}/getNFLTeams"
GET_ROSTER = f"https://{RAPID_HOST}/getNFLTeamRoster"

DEFAULT_TIMEOUT = 20
RETRIES = 3
BACKOFF = 1.6

# Keys we commonly see for injury data across Tank01 payloads.
# We check generously to future-proof field names.
INJURY_HINT_KEYS = {
    "injury", "injuryStatus", "injury_status", "status",
    "practiceStatus", "gameStatus", "game_status",
    "injuryBodyPart", "bodyPart", "body_part",
    "estReturn", "est_return", "estReturnDate", "returnDate",
    "out", "doubtful", "questionable", "probable", "active",
    "onIR", "ir", "pup", "nfi", "dnp", "limited", "full"
}

# Player identity / context columns we always try to keep
CORE_KEEP = [
    "playerID", "gsisID", "esbID", "yahoo_id", "espn_id",
    "firstName", "lastName", "playerName", "displayName",
    "teamID", "teamAbv", "team", "position", "pos", "depthChartOrder"
]

def _headers(api_key: str) -> Dict[str, str]:
    return {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": RAPID_HOST,
    }

def _req_json(url: str, headers: Dict[str, str], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    last_err = None
    for i in range(RETRIES + 1):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=DEFAULT_TIMEOUT)
            # Tank01 often returns 200 even on logical errors; check JSON structure below
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            # Both endpoints typically wrap payload in {statusCode, body}
            if not isinstance(data, dict) or "body" not in data:
                # Some calls may return a list; fall back to raw
                return data
            return data
        except Exception as e:
            last_err = e
            if i < RETRIES:
                sleep_s = BACKOFF ** i
                print(f"[warn] GET {url} failed: {e} | retry {i+1}/{RETRIES} in {sleep_s:.1f}s")
                time.sleep(sleep_s)
            else:
                raise RuntimeError(f"Failed GET {url}: {last_err}") from last_err

def fetch_teams(api_key: str) -> List[Dict[str, Any]]:
    res = _req_json(GET_TEAMS, _headers(api_key))
    # Expect {statusCode, body=[{teamAbv,...}, ...]}
    body = res.get("body") if isinstance(res, dict) else res
    if not isinstance(body, list):
        raise RuntimeError("Unexpected teams response shape (no list body).")
    return body

def fetch_roster_for_team(api_key: str, team_abv: str) -> List[Dict[str, Any]]:
    params = {"teamAbv": team_abv}
    res = _req_json(GET_ROSTER, _headers(api_key), params=params)
    body = res.get("body") if isinstance(res, dict) else res
    if not isinstance(body, dict):
        return []
    roster = body.get("roster", [])
    if not isinstance(roster, list):
        return []
    # Each element is a player map
    return roster

def extract_injury_slice(player: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull out any injury-related fields (best-effort) and return a flat dict.
    We also keep core identity fields for joinability.
    """
    out: Dict[str, Any] = {}

    # Identity / context
    for k in CORE_KEEP:
        if k in player:
            out[k] = player.get(k)

    # Best-effort name fallback
    if "playerName" not in out:
        first = str(player.get("firstName", "")).strip()
        last = str(player.get("lastName", "")).strip()
        if first or last:
            out["playerName"] = (first + " " + last).strip() or None

    # Grab any keys that look like injury signals
    for k, v in player.items():
        kl = str(k).strip()
        if kl in INJURY_HINT_KEYS or any(tok in kl.lower() for tok in [
            "injur", "status", "return", "practice", "ir", "pup", "dnp", "limited", "full"
        ]):
            out[kl] = v

    return out

def normalize_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Heuristic: consider a row "injury-relevant" if any injury-ish columns contain true-ish or non-empty values
    injury_cols = [c for c in df.columns if (
        c in INJURY_HINT_KEYS or
        any(tok in c.lower() for tok in ["injur", "status", "return", "practice", "ir", "pup"])
    )]

    if injury_cols:
        # keep rows where at least one injury column is non-null and not trivially false
        mask = False
        for c in injury_cols:
            # Treat values like '', 'Active', None as possibly not injuries—still keep if practice status present
            mask = mask | df[c].notna() & (df[c].astype(str).str.len() > 0)
        df = df.loc[mask].copy()

    # Deduplicate by playerID+team if present
    subset_cols = [c for c in ["playerID", "teamAbv"] if c in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols)

    # Sort for readability
    sort_cols = [c for c in ["teamAbv", "position", "playerName"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    return df.reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser(description="Fetch current NFL injuries from Tank01 (aggregated from team rosters).")
    ap.add_argument("--rapidapi-key", default=os.getenv("RAPIDAPI_KEY"), help="RapidAPI key (env RAPIDAPI_KEY if omitted)")
    ap.add_argument("--outfile", default=None, help="CSV filepath (default: nfl_injuries_YYYYMMDD_HHMM.csv)")
    ap.add_argument("--include-healthy", action="store_true", help="Include players with no obvious injury flags.")
    args = ap.parse_args()

    if not args.rapidapi_key:
        print("Missing RapidAPI key. Provide --rapidapi-key or set RAPIDAPI_KEY.", file=sys.stderr)
        sys.exit(2)

    teams = fetch_teams(args.rapidapi_key)
    team_abvs = [t.get("teamAbv") for t in teams if t.get("teamAbv")]
    if not team_abvs:
        raise RuntimeError("No teams returned from /getNFLTeams")

    rows: List[Dict[str, Any]] = []
    for abv in team_abvs:
        try:
            roster = fetch_roster_for_team(args.rapidapi_key, abv)
            for p in roster:
                rec = extract_injury_slice(p)
                if args.include_healthy:
                    # keep all; downstream can filter
                    rec["_sourceTeam"] = abv
                    rows.append(rec)
                else:
                    # only keep if we see at least one injury-ish field with content
                    if any(k for k in rec.keys() if k not in CORE_KEEP):
                        rec["_sourceTeam"] = abv
                        rows.append(rec)
        except Exception as e:
            print(f"[warn] roster fetch failed for {abv}: {e}", file=sys.stderr)

    df = normalize_rows(rows)

    # Timestamped default name in local timezone
    now_local = datetime.now(tz=tz.tzlocal())
    stamp = now_local.strftime("%Y%m%d_%H%M")
    outpath = args.outfile or f"nfl_injuries_{stamp}.csv"

    # If user asked for healthy included and nothing looks injury-related, still output whatever we got
    if df.empty and args.include_healthy and rows:
        df = pd.DataFrame(rows)

    df.to_csv(outpath, index=False)
    print(f"[done] wrote {len(df):,} rows → {outpath}")

if __name__ == "__main__":
    main()
