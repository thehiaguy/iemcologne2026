"""
smoke_test.py
-------------
Probes both data sources and prints exactly what fields each returns.

SOURCE A: hltv-async-api  (async HTTP — blocked by Cloudflare without proxy)
SOURCE B: hltv-scraper     (Selenium + undetected-chromedriver — bypasses Cloudflare)

Usage:
  python smoke_test.py                # both sources
  python smoke_test.py --async-only   # only hltv-async-api
  python smoke_test.py --scraper-only # only Selenium scraper (opens Chrome)

Test subjects:
  Team  : Natus Vincere (HLTV ID 4608)
  Player: Aleksib       (HLTV ID 9816)
"""

import argparse
import asyncio
import time

TEST_TEAM_ID   = 4608
TEST_PLAYER_ID = 9816
TEST_TEAM_SLUG = "natus-vincere"
TEST_PLAYER_SLUG = "aleksib"

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"
WARN = "[WARN]"


def sep(title: str):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print("=" * 65)


def show(obj, _depth: int = 0, max_depth: int = 4):
    pad = "  " * (_depth + 1)
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and _depth < max_depth:
                print(f"{pad}{k}: {type(v).__name__}")
                show(v, _depth + 1, max_depth)
            else:
                snippet = repr(v)
                print(f"{pad}{k}: {type(v).__name__} = {snippet[:80]}")
    elif isinstance(obj, list):
        print(f"{pad}[list of {len(obj)} items]")
        if obj and _depth < max_depth:
            print(f"{pad}First item:")
            show(obj[0], _depth + 1, max_depth)
    else:
        print(f"{pad}{repr(obj)[:80]}")


# ---------------------------------------------------------------------------
# SOURCE A: hltv-async-api
# ---------------------------------------------------------------------------

async def test_async_api():
    sep("SOURCE A: hltv-async-api")

    try:
        from hltv_async_api import Hltv
        print(f"  {PASS} Import OK (version 0.8.3)")
    except ImportError as e:
        print(f"  {FAIL} Import failed: {e}")
        return

    print(f"\n  {INFO} Available methods on Hltv object:")
    methods = [m for m in dir(Hltv) if not m.startswith("_") and callable(getattr(Hltv, m))]
    for m in methods:
        print(f"      {m}")

    print(f"\n  {WARN} NOTE: hltv-async-api uses direct HTTP requests.")
    print(f"  {WARN} HLTV blocks these with Cloudflare (HTTP 403).")
    print(f"  {WARN} A proxy is required for production use.")
    print(f"  {INFO} Attempting anyway (expect 403 retries)...")

    async with Hltv(max_delay=3, min_delay=1, timeout=10) as hltv:

        # --- get_results: recent match results ---
        print(f"\n  -- get_results() --")
        try:
            results = await hltv.get_results(days=1, max_=5)
            print(f"  {PASS} Type: {type(results).__name__}, len={len(results) if results else 0}")
            show(results)
        except Exception as e:
            print(f"  {FAIL} {e}")

        await asyncio.sleep(2)

        # --- get_team_info ---
        print(f"\n  -- get_team_info({TEST_TEAM_ID}, '{TEST_TEAM_SLUG}') --")
        try:
            team = await hltv.get_team_info(TEST_TEAM_ID, TEST_TEAM_SLUG)
            print(f"  {PASS} Type: {type(team).__name__}")
            show(team)
        except Exception as e:
            print(f"  {FAIL} {e}")

        await asyncio.sleep(2)

        # --- get_player_info ---
        print(f"\n  -- get_player_info({TEST_PLAYER_ID}, '{TEST_PLAYER_SLUG}') --")
        try:
            player = await hltv.get_player_info(TEST_PLAYER_ID, TEST_PLAYER_SLUG)
            print(f"  {PASS} Type: {type(player).__name__}")
            show(player)
        except Exception as e:
            print(f"  {FAIL} {e}")

        await asyncio.sleep(2)

        # --- get_match_info: try with a known recent match ID ---
        # This match ID is from NAVI's recent results (IEM Atlanta 2026)
        test_match_id = 2394178
        print(f"\n  -- get_match_info({test_match_id}) --")
        try:
            match = await hltv.get_match_info(
                test_match_id,
                "gamerlegion-vs-natus-vincere-iem-atlanta-2026"
            )
            print(f"  {PASS} Type: {type(match).__name__}")
            show(match)
        except Exception as e:
            print(f"  {FAIL} {e}")

        await asyncio.sleep(2)

        # --- get_top_players ---
        print(f"\n  -- get_top_players(top=5) --")
        try:
            top = await hltv.get_top_players(top=5)
            print(f"  {PASS} Type: {type(top).__name__}, len={len(top) if top else 0}")
            show(top)
        except Exception as e:
            print(f"  {FAIL} {e}")

        await asyncio.sleep(2)

        # --- get_event_results: grab results from a recent event ---
        print(f"\n  -- get_event_results() --")
        try:
            # IEM Atlanta 2026 event ID (approximate)
            event_results = await hltv.get_event_results(7883, max_=5)
            print(f"  {PASS} Type: {type(event_results).__name__}")
            show(event_results)
        except Exception as e:
            print(f"  {FAIL} {e}")


# ---------------------------------------------------------------------------
# SOURCE B: hltv-scraper (Selenium / undetected-chromedriver)
# ---------------------------------------------------------------------------

def test_scraper():
    sep("SOURCE B: hltv-scraper  (Selenium + undetected-chromedriver)")

    try:
        import undetected_chromedriver as uc
        from bs4 import BeautifulSoup
        print(f"  {PASS} Imports OK")
    except ImportError as e:
        print(f"  {FAIL} Import failed: {e}")
        return

    # Launch Chrome — NOT headless, Cloudflare detects headless mode
    try:
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--start-maximized")
        print(f"  {INFO} Launching Chrome 148 (visible window, bypasses Cloudflare)...")
        driver = uc.Chrome(options=options, version_main=148)
        print(f"  {PASS} Chrome launched OK")
    except Exception as e:
        print(f"  {FAIL} Chrome failed to launch: {e}")
        print(f"  {INFO} Ensure Google Chrome is installed.")
        return

    try:
        # --- Test 1: Individual player stats page ---
        url = f"https://www.hltv.org/stats/players/{TEST_PLAYER_ID}/{TEST_PLAYER_SLUG}"
        print(f"\n  -- Player stats page --")
        print(f"  {INFO} {url}")
        driver.get(url)
        time.sleep(8)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        print(f"  {INFO} Page title: {soup.title.string if soup.title else 'N/A'}")

        _scrape_stats_rows(soup, "player stats rows")
        _scrape_table_headers(soup, "player stats table headers")

        # ---------------------------------------------------------------
        # Test 2: Players ranking table (aggregate stats for all players)
        # ---------------------------------------------------------------
        time.sleep(2)
        url = "https://www.hltv.org/stats/players?startDate=2025-01-01&endDate=2025-12-31&rankingFilter=Top30"
        print(f"\n  -- Players ranking table (2025, Top 30 teams) --")
        print(f"  {INFO} {url}")
        driver.get(url)
        time.sleep(9)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        print(f"  {INFO} Page title: {soup.title.string if soup.title else 'N/A'}")

        headers = _scrape_table_headers(soup, "ranking table columns")

        if headers:
            rows = soup.select("table tbody tr")
            print(f"\n  {INFO} Sample data rows ({min(3, len(rows))} of {len(rows)}):")
            for row in rows[:3]:
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                print(f"      {cells}")

        # ---------------------------------------------------------------
        # Test 3: Match stats page
        # ---------------------------------------------------------------
        time.sleep(2)
        url = "https://www.hltv.org/stats/matches/mapstatsid/199398/natus-vincere-vs-gamerlegion"
        print(f"\n  -- Match map stats page --")
        print(f"  {INFO} {url}")
        driver.get(url)
        time.sleep(8)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        print(f"  {INFO} Page title: {soup.title.string if soup.title else 'N/A'}")
        _scrape_table_headers(soup, "match stats table columns")

        rows = soup.select("table tbody tr")
        print(f"\n  {INFO} Sample rows ({min(3, len(rows))} of {len(rows)}):")
        for row in rows[:3]:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            print(f"      {cells}")

    finally:
        driver.quit()
        print(f"\n  {INFO} Chrome closed")


def _scrape_stats_rows(soup, label: str):
    rows = soup.select(".stats-row")
    if rows:
        print(f"\n  {PASS} {label} ({len(rows)} rows):")
        for row in rows:
            spans = row.find_all("span")
            if len(spans) >= 2:
                k = spans[0].get_text(strip=True)
                v = spans[1].get_text(strip=True)
                print(f"      {k:35s} = {v}")
    else:
        print(f"  {WARN} No .stats-row elements found for: {label}")


def _scrape_table_headers(soup, label: str):
    headers = [th.get_text(strip=True) for th in soup.select("table thead th")]
    if headers:
        print(f"\n  {PASS} {label}:")
        for h in headers:
            print(f"      {h}")
    else:
        print(f"  {WARN} No table headers found for: {label}")
    return headers


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary():
    sep("Combined strategy for model_training.py")
    print("""
  SOURCE A: hltv-async-api
    Methods available : get_results, get_match_info, get_team_info,
                        get_player_info, get_top_players, get_event_results
    Cloudflare status : BLOCKED without proxy (HTTP 403)
    Fix needed        : Add proxy list to Hltv(proxies=[...]) or use
                        the same undetected-chromedriver approach

  SOURCE B: hltv-scraper (Selenium)
    Cloudflare status : BYPASSED via undetected-chromedriver
    Data type         : Aggregate stats per date range (not per-match)
    Gives             : Rating 2.1, KAST%, opening kill ratio, K/D, ADR,
                        headshot %, impact score, weapon breakdowns

  RECOMMENDED APPROACH:
    Use Selenium (Source B) for EVERYTHING since it bypasses Cloudflare.
    Drive these HLTV pages:
      /stats/players?startDate=X&endDate=Y   -> aggregate player benchmarks
      /stats/matches/matchid/XXX/...         -> per-match player scorecards
      /results?team=ID                       -> team match history
    This gives both rolling per-match features AND aggregate stat features.
""")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--async-only",   action="store_true")
    parser.add_argument("--scraper-only", action="store_true")
    args = parser.parse_args()

    run_async   = not args.scraper_only
    run_scraper = not args.async_only

    if run_async:
        asyncio.run(test_async_api())

    if run_scraper:
        test_scraper()

    print_summary()
