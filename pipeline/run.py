# -*- coding: utf-8 -*-
"""
Update every league: download new data, then make predictions.

    python -m pipeline.run                # all leagues
    python -m pipeline.run la-liga        # just one
    python -m pipeline.run --no-fetch     # predictions only, from saved data

Each league uses only its own data (data/<league>/). If one league fails,
the others still run, and the failure is reported at the end.
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import load_env  # noqa: E402,F401  (reads API_FOOTBALL_KEY from a local .env)

from .config import LEAGUES  # noqa: E402
from . import fetch, models  # noqa: E402


def main(argv):
    no_fetch = "--no-fetch" in argv
    chosen = [a for a in argv if not a.startswith("--")] or list(LEAGUES)
    key = os.environ.get("API_FOOTBALL_KEY", "")
    if not no_fetch and not key:
        sys.exit("Set API_FOOTBALL_KEY (environment variable, .env file or GitHub secret).")

    if not no_fetch:
        try:
            st = fetch.api_get("status", {}, key).get("response") or {}
            sub, req = st.get("subscription") or {}, st.get("requests") or {}
            plan, limit, used = sub.get("plan", "?"), req.get("limit_day"), req.get("current")
            print(f"API-Football plan: {plan} (used {used} of {limit} requests today)")
            if str(plan).lower() == "free":
                fetch.set_interval(6.5)
                print("WARNING: free plan (100 requests/day, 10/minute) - not enough for all leagues. "
                      "Put the key from your paid account in the API_FOOTBALL_KEY secret.")
                chosen = chosen[:1]
        except Exception as e:
            print(f"Could not read API status: {e}")

    failed = []
    for league in chosen:
        name, league_id, country, _slug = LEAGUES[league]
        print(f"\n=== {name} ===", flush=True)
        try:
            if not no_fetch:
                n_stats, n_players = fetch.update_league(league, league_id, key)
                print(f"  fetched: {n_stats} matches needing stats, {n_players} players "
                      f"({fetch.CALLS[0]} API calls so far)")
            out = models.predict_league(league, name, country)
            if out:
                print(f"  predicted {len(out['fixtures'])} fixtures from {out['matches_used']} past matches")
            else:
                print("  no upcoming fixtures found")
        except Exception as e:  # keep going with the other leagues
            traceback.print_exc()
            failed.append(f"{name}: {e}")

    print(f"\nAPI calls this run: {fetch.CALLS[0]}")
    if failed:
        print("FAILED:\n  " + "\n  ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
