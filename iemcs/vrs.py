"""Parse the Valve Regional Standings repository into clean, CS2-only tables.

The repo (https://github.com/ValveSoftware/counter-strike_regional_standings) ships
monthly standings snapshots. Each snapshot has one markdown "details" file per team
that lists every match the roster played in the trailing 6-month window, with date,
opponent and win/loss. Across all snapshots (2024-02 .. 2026-05) and all teams this
reconstructs the full CS2 professional match history.

Outputs:
  * parse_matches()    -> deduped [date, winner, loser]
  * parse_standings()  -> [snapshot, date, rank, points, team, roster] (top roster/org)
  * build_region_map() -> {team -> region}
"""
from __future__ import annotations

import glob
import os
import re
import subprocess

import pandas as pd

from . import config
from .teams import normalize_name

# A details-table match row, e.g.
# |  44 |  4 | 2026-05-03 | Natus Vincere | W | 1.000 | ... |
_ROW_RE = re.compile(
    r"^\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(.+?)\s*\|\s*([WL])\s*\|"
)
_TEAM_RE = re.compile(r"Team Name:\s*(.+?)<br */>")
_REGION_RE = re.compile(r"Region:\s*\[(\w+)\]")

# A global standings row: | rank | points | name | roster | ... |
_STANDING_RE = re.compile(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|")


# --------------------------------------------------------------------------- #
# Repo handling
# --------------------------------------------------------------------------- #
def ensure_repo(update: bool = False) -> str:
    """Clone the VRS repo into config.VRS_REPO_DIR if absent. Returns its path."""
    path = config.VRS_REPO_DIR
    if os.path.isdir(os.path.join(path, "live")):
        if update:
            subprocess.run(["git", "-C", path, "pull", "--depth", "1"], check=False)
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Cloning VRS repo into {path} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", config.VRS_REPO_URL, path], check=True
    )
    return path


def _detail_paths() -> list[tuple[str, str]]:
    """Return (snapshot_date, filepath) for every team-details markdown file."""
    pattern = os.path.join(config.VRS_REPO_DIR, "live", "*", "details", "*", "*.md")
    out = []
    for fp in glob.glob(pattern):
        snapshot = os.path.basename(os.path.dirname(fp))  # e.g. 2026_05_04
        out.append((snapshot, fp))
    return sorted(out)


def _standings_paths() -> list[str]:
    pattern = os.path.join(
        config.VRS_REPO_DIR, "live", "*", "standings_global_*.md"
    )
    return sorted(glob.glob(pattern))


# --------------------------------------------------------------------------- #
# Matches
# --------------------------------------------------------------------------- #
def parse_matches(min_date: str = config.CS2_CUTOFF_DATE) -> pd.DataFrame:
    """Deduplicated CS2 match results.

    Each physical match appears in both teams' files and across multiple snapshots;
    we collapse on (date, {teamA, teamB}). Returns columns: date, winner, loser.
    """
    seen: dict[tuple[str, frozenset], str] = {}
    for _snapshot, fp in _detail_paths():
        with open(fp, encoding="utf-8") as f:
            txt = f.read()
        mt = _TEAM_RE.search(txt)
        if not mt:
            continue
        team = normalize_name(mt.group(1))
        for line in txt.splitlines():
            m = _ROW_RE.match(line)
            if not m:
                continue
            date, opp, wl = m.groups()
            opp = normalize_name(opp)
            if team == opp:
                continue
            key = (date, frozenset((team, opp)))
            if key in seen:
                continue
            seen[key] = team if wl == "W" else opp

    rows = []
    for (date, pair), winner in seen.items():
        a, b = tuple(pair)
        loser = b if winner == a else a
        rows.append((date, winner, loser))
    df = pd.DataFrame(rows, columns=["date", "winner", "loser"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.Timestamp(min_date)]
    df = df.sort_values("date").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Standings (ratings + rosters)
# --------------------------------------------------------------------------- #
def parse_standings() -> pd.DataFrame:
    """Monthly global VRS standings.

    VRS ranks *rosters*, so an org can appear multiple times; we keep the highest
    ranked (lowest rank number) entry per org per snapshot.
    Columns: snapshot, date, rank, points, team, roster.
    """
    records = []
    for fp in _standings_paths():
        snapshot = re.search(r"standings_global_(\d{4}_\d{2}_\d{2})", fp).group(1)
        date = pd.Timestamp(snapshot.replace("_", "-"))
        best: dict[str, tuple] = {}
        for line in open(fp, encoding="utf-8").read().splitlines():
            m = _STANDING_RE.match(line)
            if not m or not m.group(1).isdigit():
                continue
            rank, pts, name, roster = m.groups()
            name = normalize_name(name)
            rank = int(rank)
            if name not in best or rank < best[name][0]:
                best[name] = (rank, int(pts), roster.strip())
        for name, (rank, pts, roster) in best.items():
            records.append((snapshot, date, rank, pts, name, roster))
    df = pd.DataFrame(
        records, columns=["snapshot", "date", "rank", "points", "team", "roster"]
    )
    return df.sort_values(["date", "rank"]).reset_index(drop=True)


def build_region_map() -> dict[str, str]:
    """team -> region, taken from the most recent details file mentioning the team."""
    region: dict[str, str] = {}
    for snapshot, fp in _detail_paths():  # sorted ascending -> later snapshots win
        txt = open(fp, encoding="utf-8").read()
        mt = _TEAM_RE.search(txt)
        mr = _REGION_RE.search(txt)
        if mt and mr:
            region[normalize_name(mt.group(1))] = mr.group(1)
    return region


# --------------------------------------------------------------------------- #
# Convenience: current snapshot view of the 32-team field
# --------------------------------------------------------------------------- #
def current_standings(snapshot: str = config.CURRENT_SNAPSHOT) -> pd.DataFrame:
    st = parse_standings()
    cur = st[st["snapshot"] == snapshot].copy()
    if cur.empty:  # fall back to the latest available snapshot
        latest = st["snapshot"].max()
        cur = st[st["snapshot"] == latest].copy()
    return cur
