import json
import sqlite3
import sys

sys.path.insert(0, ".")
from o1_config import ALL_TEAMS, API_BASE, STAGE1_TEAMS, STAGE2_TEAMS, STAGE3_TEAMS, TEAM_IDS

conn = sqlite3.connect("data/hltv_cache.db")

# Stage label lookup
_stage = {}
for t in STAGE3_TEAMS: _stage[t] = "S3"
for t in STAGE2_TEAMS: _stage[t] = "S2"
for t in STAGE1_TEAMS: _stage[t] = "S1"

print(f"\n{'Team':<24} {'Stg':>3}  {'Results':>7}  {'Pages':>5}  {'Maps':>5}")
print("-" * 56)

total_results = total_pages = total_maps = 0
pending = []

for team in ALL_TEAMS:
    tid = TEAM_IDS.get(team)
    if not tid:
        continue

    # Check if team results API has been cached
    api_url = f"{API_BASE}/teams/{tid}/results?limit=100"
    row = conn.execute(
        "SELECT content FROM page_cache WHERE url = ?", (api_url,)
    ).fetchone()

    if not row:
        pending.append(team)
        print(f"{team:<24} {_stage[team]:>3}  {'pending':>7}")
        continue

    try:
        data = json.loads(row[0])
        results = data if isinstance(data, list) else data.get("results", [])
    except Exception:
        print(f"{team:<24} {_stage[team]:>3}  {'err':>7}")
        continue

    n_results = len(results)

    # Count how many of those match pages are cached (= were scraped for mapstats)
    match_urls = [
        r.get("matchUrl") or r.get("match_url", "")
        for r in results
    ]
    match_urls = [u for u in match_urls if u]

    if match_urls:
        placeholders = ",".join("?" * len(match_urls))
        n_pages = conn.execute(
            f"SELECT COUNT(*) FROM page_cache WHERE url IN ({placeholders})",
            match_urls,
        ).fetchone()[0]
    else:
        n_pages = 0

    # Count mapstats pages cached for those match pages
    # Mapstats are only scrape-able once the match page is cached;
    # use heuristic: mapstats pages fetched since the team was first seen
    # Instead just count total mapstats cached globally (shown in footer)
    n_maps = 0  # placeholder — global total shown below

    total_results += n_results
    total_pages   += n_pages

    bar_len = 12
    filled  = round(bar_len * n_pages / n_results) if n_results else 0
    bar     = "█" * filled + "░" * (bar_len - filled)

    print(
        f"{team:<24} {_stage[team]:>3}  {n_pages:>3}/{n_results:<3}    [{bar}]"
    )

# Global cache totals
total_cached = conn.execute("SELECT COUNT(*) FROM page_cache").fetchone()[0]
total_maps   = conn.execute(
    "SELECT COUNT(*) FROM page_cache WHERE url LIKE '%mapstatsid%'"
).fetchone()[0]
conn.close()

print("-" * 56)
print(f"{'Match pages scraped':<24}  {total_pages:>4} / {total_results}")
print(f"{'Mapstats pages cached':<24}  {total_maps:>4}")
print(f"{'Total cache entries':<24}  {total_cached:>4}")
if pending:
    print(f"\nNot yet fetched ({len(pending)}): {', '.join(pending)}")
print()
