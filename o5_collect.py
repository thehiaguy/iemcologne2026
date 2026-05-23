"""
[5/7] collect.py
----------------
Data collection for IEM Cologne 2026 CS2 match prediction.

Fetches match history for all SEED_TEAM_IDS using N_BROWSERS parallel
Chrome instances and caches everything in SQLite. On completion writes:

  data/raw_matches.parquet   — deduplicated match records
  data/team_profiles.pkl     — team profile metadata (rankings, age, lineup)

Run this before train.py. Re-running is safe: cached pages are reused
within the 72-hour TTL so only new/expired pages are re-fetched.

Usage
-----
  python o5_collect.py
"""

import logging
import math
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from o1_config import DATA_DIR, MAPSTATS_START_DATE, N_BROWSERS, SEED_TEAM_IDS
from o2_scraper import HLTVScraper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

def _result_to_record(
    res: dict, player_rows: Optional[List[dict]] = None
) -> Optional[dict]:
    try:
        t1_won = res["matchWon"]
        record = {
            "match_id":   res["matchId"],
            "match_date": res["matchDate"],
            "team1_name": res["team1Name"],
            "team2_name": res["team2Name"],
            "team1_maps": res["team1Score"],
            "team2_maps": res["team2Score"],
            "match_type": res["matchType"],
            "label":      1 if t1_won else 0,
            "total_maps": res["team1Score"] + res["team2Score"],
        }

        for slot in (1, 2):
            prefix   = f"team{slot}"
            slot_idx = slot - 1
            if player_rows:
                pf = pd.DataFrame([p for p in player_rows if p["team_slot"] == slot_idx])
            else:
                pf = pd.DataFrame()

            stat_cols = (
                "avg_rating", "avg_adr", "avg_kast", "avg_opening_kd",
                "avg_opening_kills", "avg_multi_kills", "avg_clutches",
                "avg_flash_assists", "avg_hs_kills",
            )
            src_map = {
                "avg_rating":        "rating",
                "avg_adr":           "adr",
                "avg_kast":          "kast_pct",
                "avg_opening_kd":    "opening_kd",
                "avg_opening_kills": "opening_kills",
                "avg_multi_kills":   "multi_kills",
                "avg_clutches":      "clutches",
                "avg_flash_assists": "flash_assists",
                "avg_hs_kills":      "hs_kills",
            }

            if pf.empty:
                for col in stat_cols:
                    record[f"{prefix}_{col}"] = np.nan
            else:
                for col, src in src_map.items():
                    val = float(np.nanmean(pf[src])) if src in pf.columns else np.nan
                    record[f"{prefix}_{col}"] = val

        return record
    except Exception as exc:
        log.debug("_result_to_record error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Collection workers
# ---------------------------------------------------------------------------

def _collect_team_batch(team_ids: List[int]) -> Tuple[List[dict], Dict[int, dict]]:
    """Worker: one Chrome instance handles its entire batch of teams."""
    records:   List[dict]      = []
    profiles:  Dict[int, dict] = {}
    seen_mids: set              = set()

    with HLTVScraper() as scraper:
        for team_id in team_ids:
            log.info("[browser] team %d — fetching results", team_id)
            results = scraper.get_team_results(team_id, limit=100)
            total   = len(results)
            log.info("[browser] team %d — %d results", team_id, total)

            new_matches  = 0
            scraped_full = 0
            scraped_skip = 0

            for i, res in enumerate(results, 1):
                mid = res["matchId"]
                if not mid or mid in seen_mids:
                    scraped_skip += 1
                    continue
                seen_mids.add(mid)
                new_matches += 1

                match_url = res["matchUrl"]
                if not match_url:
                    scraped_skip += 1
                    continue

                match_date     = pd.to_datetime(res.get("matchDate", ""), errors="coerce")
                scrape_players = pd.notna(match_date) and match_date >= MAPSTATS_START_DATE

                if scrape_players:
                    log.info(
                        "[browser] team %d — match %d/%d  (full scrape, %d done)",
                        team_id, i, total, scraped_full,
                    )
                    ms_ids          = scraper.get_mapstats_ids(match_url)
                    all_player_rows: List[dict] = []
                    for ms_id, slug in ms_ids:
                        all_player_rows.extend(scraper.get_map_stats(ms_id, slug))
                    rec = _result_to_record(res, all_player_rows or None)
                    scraped_full += 1
                else:
                    rec = _result_to_record(res)
                    scraped_skip += 1

                if rec:
                    records.append(rec)

            log.info(
                "[browser] team %d — done  new=%d  full=%d  skipped=%d",
                team_id, new_matches, scraped_full, scraped_skip,
            )
            profiles[team_id] = scraper.get_team_profile(team_id)

    return records, profiles


def collect_match_data() -> Tuple[pd.DataFrame, Dict[int, dict]]:
    """
    Collect raw match data using N_BROWSERS parallel Chrome instances.
    Returns (raw DataFrame, profiles dict).
    """
    batch_size = math.ceil(len(SEED_TEAM_IDS) / N_BROWSERS)
    batches = [
        SEED_TEAM_IDS[i : i + batch_size]
        for i in range(0, len(SEED_TEAM_IDS), batch_size)
    ]
    log.info(
        "Launching %d Chrome browser(s) for %d teams "
        "(mapstats only for matches >= %s)",
        len(batches), len(SEED_TEAM_IDS), MAPSTATS_START_DATE.date(),
    )

    all_records:  List[dict]      = []
    all_profiles: Dict[int, dict] = {}
    seen_mids:    set              = set()

    with ThreadPoolExecutor(max_workers=N_BROWSERS) as pool:
        futures = {pool.submit(_collect_team_batch, batch): batch for batch in batches}
        for future in as_completed(futures):
            try:
                records, profiles = future.result()
                for rec in records:
                    mid = rec.get("match_id")
                    if mid and mid not in seen_mids:
                        seen_mids.add(mid)
                        all_records.append(rec)
                all_profiles.update(profiles)
            except Exception as exc:
                log.error("Browser batch failed: %s", exc)

    log.info("Collected %d unique match records", len(all_records))
    if not all_records:
        raise RuntimeError("No data collected. Check SEED_TEAM_IDS and API/Selenium.")

    df = pd.DataFrame(all_records)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.dropna(subset=["match_date"]).sort_values("match_date").reset_index(drop=True)
    return df, all_profiles


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DATA_DIR.mkdir(exist_ok=True)
    raw_df, profiles = collect_match_data()

    out_parquet  = DATA_DIR / "raw_matches.parquet"
    out_profiles = DATA_DIR / "team_profiles.pkl"

    raw_df.to_parquet(out_parquet, index=False)
    with open(out_profiles, "wb") as fh:
        pickle.dump(profiles, fh)

    log.info("Saved %d matches  → %s", len(raw_df), out_parquet)
    log.info("Saved profiles for %d teams → %s", len(profiles), out_profiles)


if __name__ == "__main__":
    main()
