"""
batch_sim.py — Batch at-bat simulator for realism analysis.

Calls the local Flask API at http://localhost:5000/api/predict
to simulate ~200 at-bats and computes MLB-comparable statistics.

At-bat state machine
---------------------
Each pitch outcome is determined by:
  1. The model outputs swing_prob + contact_outcome.
  2. We stochastically decide SWING vs TAKE based on swing_prob.
  3. If SWING:
       - contact_outcome determines what happens
       - contact classes expected: 'foul', 'strike' (swing+miss), 'fair_ball' (in-play)
  4. If TAKE:
       - is_strike (based on location bounds) determines ball vs called_strike

At-bat ends when:
  - strikes == 3  → strikeout
  - balls == 4    → walk
  - fair_ball hit → in play (single/double/out etc. — we don't simulate fielding here)
"""

import requests
import random
import json
import sys
from collections import defaultdict

API_BASE = "http://localhost:5000"

# -------------------------------------------------------------------
# Strike-zone geometry (matches inference.py)
# -------------------------------------------------------------------
STRIKE_ZONE_X = 0.83   # |plate_x| <= this is in zone
STRIKE_ZONE_Z_LO = 1.5
STRIKE_ZONE_Z_HI = 3.5

def is_in_strike_zone(plate_x: float, plate_z: float) -> bool:
    return abs(plate_x) <= STRIKE_ZONE_X and STRIKE_ZONE_Z_LO <= plate_z <= STRIKE_ZONE_Z_HI


def classify_contact(contact_outcome: str) -> str:
    """Map model contact_outcome label to one of: foul | miss | fair"""
    co = contact_outcome.lower()
    if "foul" in co:
        return "foul"
    if "fair" in co or "in_play" in co or "hit" in co or "ball_in_play" in co:
        return "fair"
    # swing-and-miss
    return "miss"


def get_players():
    r = requests.get(f"{API_BASE}/api/players", timeout=10)
    r.raise_for_status()
    data = r.json()
    return data["pitchers"], data["batters"]


def call_predict(pitcher_id: int, batter_id: int, user_context: dict) -> dict:
    payload = {
        "pitcher_mlbam": pitcher_id,
        "batter_mlbam": batter_id,
        "user_context": user_context,
    }
    r = requests.post(f"{API_BASE}/api/predict", json=payload, timeout=15)
    r.raise_for_status()
    return r.json()


def simulate_atbat(pitcher: dict, batter: dict) -> dict:
    """
    Simulate a single at-bat; return a result dict.
    """
    pitcher_id = pitcher["id"]
    batter_id = batter["id"]

    balls = 0
    strikes = 0
    pitches = []
    counts_seen = set()
    first_pitch_strike = None

    while True:
        counts_seen.add((balls, strikes))
        user_context = {
            "balls": balls,
            "strikes": strikes,
            "outs_when_up": 1,
            "on_1b": 0,
            "on_2b": 0,
            "on_3b": 0,
            "inning": 5,
            "score_diff": 0,
        }
        if pitches:
            last = pitches[-1]
            user_context["last_pitch_type"] = last["pitch_type"]

        result = call_predict(pitcher_id, batter_id, user_context)

        pitch_type = result["pitch_type"]
        plate_x = result["location"]["plate_x"]
        plate_z = result["location"]["plate_z"]
        swing_prob = result["swing_prob"]
        contact_outcome = result.get("contact_outcome", "miss")

        # Stochastic swing decision
        swung = random.random() < swing_prob

        in_zone = is_in_strike_zone(plate_x, plate_z)

        if swung:
            contact = classify_contact(contact_outcome)
            if contact == "fair":
                pitch_record = {
                    "pitch_type": pitch_type,
                    "plate_x": plate_x,
                    "plate_z": plate_z,
                    "swung": True,
                    "in_zone": in_zone,
                    "outcome": "fair_ball",
                    "contact": "fair",
                }
                pitches.append(pitch_record)
                result_label = "in_play"
                break
            elif contact == "foul":
                if strikes < 2:
                    strikes += 1
                pitch_record = {
                    "pitch_type": pitch_type,
                    "plate_x": plate_x,
                    "plate_z": plate_z,
                    "swung": True,
                    "in_zone": in_zone,
                    "outcome": "foul",
                    "contact": "foul",
                }
            else:  # miss (swinging strike)
                strikes += 1
                pitch_record = {
                    "pitch_type": pitch_type,
                    "plate_x": plate_x,
                    "plate_z": plate_z,
                    "swung": True,
                    "in_zone": in_zone,
                    "outcome": "swinging_strike",
                    "contact": "miss",
                }
        else:
            # Take
            if in_zone:
                strikes += 1
                pitch_record = {
                    "pitch_type": pitch_type,
                    "plate_x": plate_x,
                    "plate_z": plate_z,
                    "swung": False,
                    "in_zone": in_zone,
                    "outcome": "called_strike",
                    "contact": None,
                }
            else:
                balls += 1
                pitch_record = {
                    "pitch_type": pitch_type,
                    "plate_x": plate_x,
                    "plate_z": plate_z,
                    "swung": False,
                    "in_zone": in_zone,
                    "outcome": "ball",
                    "contact": None,
                }

        pitches.append(pitch_record)

        # Record first-pitch result
        if first_pitch_strike is None:
            first_pitch_strike = (pitch_record["outcome"] in
                                  {"called_strike", "swinging_strike", "foul", "fair_ball"})

        # Check terminal conditions
        if strikes >= 3:
            result_label = "strikeout"
            break
        if balls >= 4:
            result_label = "walk"
            break

    if first_pitch_strike is None and pitches:
        first_pitch_strike = (pitches[0]["outcome"] in
                              {"called_strike", "swinging_strike", "foul", "fair_ball"})

    return {
        "pitcher_id": pitcher_id,
        "batter_id": batter_id,
        "result": result_label,
        "pitches": pitches,
        "pitch_count": len(pitches),
        "final_balls": balls,
        "final_strikes": strikes,
        "first_pitch_strike": first_pitch_strike,
        "counts_seen": counts_seen,
    }


def main():
    N_ATBATS = 200
    print(f"Fetching player list from {API_BASE}...")
    pitchers, batters = get_players()
    print(f"  {len(pitchers)} pitchers, {len(batters)} batters available")

    # Use a fixed random seed for reproducibility
    random.seed(42)

    all_ab_results = []
    print(f"\nSimulating {N_ATBATS} at-bats...")

    for i in range(N_ATBATS):
        pitcher = random.choice(pitchers)
        batter = random.choice(batters)
        try:
            ab = simulate_atbat(pitcher, batter)
            all_ab_results.append(ab)
            if (i + 1) % 20 == 0:
                print(f"  Completed {i+1}/{N_ATBATS} at-bats...")
        except Exception as e:
            print(f"  [WARN] AT-BAT {i+1} FAILED ({pitcher['label']} vs {batter['label']}): {e}", file=sys.stderr)

    total_ab = len(all_ab_results)
    print(f"\nSuccessfully simulated {total_ab} at-bats.\n")

    # ---------------------------------------------------------------
    # Aggregate statistics
    # ---------------------------------------------------------------

    # At-bat outcomes
    outcomes = defaultdict(int)
    for ab in all_ab_results:
        outcomes[ab["result"]] += 1

    strikeout_rate = outcomes["strikeout"] / total_ab
    walk_rate = outcomes["walk"] / total_ab
    inplay_rate = outcomes["in_play"] / total_ab
    other_rate = 1.0 - strikeout_rate - walk_rate - inplay_rate

    # Pitch counts
    all_pitch_counts = [ab["pitch_count"] for ab in all_ab_results]
    avg_pitches = sum(all_pitch_counts) / total_ab

    # First pitch strike
    fp_strikes = sum(1 for ab in all_ab_results if ab["first_pitch_strike"])
    fp_strike_rate = fp_strikes / total_ab

    # Pitch type distribution
    pitch_type_counts = defaultdict(int)
    total_pitches = 0
    for ab in all_ab_results:
        for p in ab["pitches"]:
            pitch_type_counts[p["pitch_type"]] += 1
            total_pitches += 1

    # Swing stats
    total_swings = 0
    total_takes = 0
    swings_in_zone = 0
    swings_out_zone = 0
    contact_swings = 0  # fair + foul
    miss_swings = 0
    foul_swings = 0
    fair_swings = 0

    for ab in all_ab_results:
        for p in ab["pitches"]:
            if p["swung"]:
                total_swings += 1
                if p["in_zone"]:
                    swings_in_zone += 1
                else:
                    swings_out_zone += 1
                c = p["contact"]
                if c == "miss":
                    miss_swings += 1
                elif c == "foul":
                    foul_swings += 1
                    contact_swings += 1
                elif c == "fair":
                    fair_swings += 1
                    contact_swings += 1
            else:
                total_takes += 1

    swing_rate = total_swings / total_pitches if total_pitches else 0
    contact_rate = contact_swings / total_swings if total_swings else 0
    whiff_rate = miss_swings / total_swings if total_swings else 0

    # Count distribution
    all_counts = defaultdict(int)
    for ab in all_ab_results:
        for c in ab["counts_seen"]:
            all_counts[c] += 1

    # Zone distribution
    total_in_zone = sum(1 for ab in all_ab_results for p in ab["pitches"] if p["in_zone"])
    zone_pct = total_in_zone / total_pitches if total_pitches else 0

    # ---------------------------------------------------------------
    # Classify pitch families
    # ---------------------------------------------------------------
    FASTBALL_CODES  = {"FF", "FT", "SI", "FC"}
    BREAKING_CODES  = {"SL", "CU", "KC", "ST", "CS", "SV"}
    OFFSPEED_CODES  = {"CH", "FS", "FO", "SF", "EP"}

    fb_count = sum(v for k, v in pitch_type_counts.items() if k in FASTBALL_CODES)
    br_count = sum(v for k, v in pitch_type_counts.items() if k in BREAKING_CODES)
    os_count = sum(v for k, v in pitch_type_counts.items() if k in OFFSPEED_CODES)
    other_pt = total_pitches - fb_count - br_count - os_count

    # ---------------------------------------------------------------
    # Print report
    # ---------------------------------------------------------------
    sep = "=" * 60

    print(sep)
    print("  BATCH SIMULATION RESULTS  (N =", total_ab, "at-bats)")
    print(sep)

    print("\n--- AT-BAT OUTCOMES ---")
    print(f"  Strikeouts   : {outcomes['strikeout']:4d}  ({strikeout_rate*100:5.1f}%)   [MLB ~23%]")
    print(f"  Walks        : {outcomes['walk']:4d}  ({walk_rate*100:5.1f}%)   [MLB ~8-9%]")
    print(f"  Ball in play : {outcomes['in_play']:4d}  ({inplay_rate*100:5.1f}%)   [MLB ~70%]")
    for k, v in sorted(outcomes.items()):
        if k not in ("strikeout", "walk", "in_play"):
            print(f"  {k:<13}: {v:4d}  ({v/total_ab*100:5.1f}%)")

    print("\n--- PITCH VOLUME ---")
    print(f"  Total pitches simulated : {total_pitches}")
    print(f"  Avg pitches per AB      : {avg_pitches:.2f}   [MLB ~3.9]")
    print(f"  Pitch count distribution:")
    from collections import Counter
    pc_dist = Counter(all_pitch_counts)
    for k in sorted(pc_dist):
        print(f"    {k} pitches: {pc_dist[k]} ABS ({pc_dist[k]/total_ab*100:.1f}%)")

    print("\n--- FIRST PITCH STRIKE ---")
    print(f"  First pitch strike rate : {fp_strike_rate*100:.1f}%   [MLB ~60%]")

    print("\n--- PITCH TYPE DISTRIBUTION ---")
    print(f"  Fastballs (FF/FT/SI/FC) : {fb_count:5d}  ({fb_count/total_pitches*100:5.1f}%)")
    print(f"  Breaking balls (SL/CU..) : {br_count:5d}  ({br_count/total_pitches*100:5.1f}%)")
    print(f"  Offspeed (CH/FS..)       : {os_count:5d}  ({os_count/total_pitches*100:5.1f}%)")
    if other_pt:
        print(f"  Other/Unknown            : {other_pt:5d}  ({other_pt/total_pitches*100:5.1f}%)")
    print(f"\n  By code:")
    for code, cnt in sorted(pitch_type_counts.items(), key=lambda x: -x[1]):
        print(f"    {code:<5}: {cnt:5d}  ({cnt/total_pitches*100:5.1f}%)")

    print("\n--- SWING / CONTACT ---")
    print(f"  Overall swing rate      : {swing_rate*100:.1f}%")
    print(f"  Contact rate (on swings): {contact_rate*100:.1f}%   (fair+foul / total swings)")
    print(f"  Whiff rate (on swings)  : {whiff_rate*100:.1f}%")
    print(f"  Foul rate (on swings)   : {foul_swings/total_swings*100:.1f}%" if total_swings else "")
    print(f"  Fair ball rate (on swings): {fair_swings/total_swings*100:.1f}%" if total_swings else "")
    print(f"  Zone %% (pitches in zone): {zone_pct*100:.1f}%")

    print("\n--- COUNT DISTRIBUTION (% of ATs that reached each count) ---")
    print("  Count   Reached   %")
    interesting = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2),(3,0),(3,1),(3,2),(3,3),(4,0)]
    for b, s in interesting:
        if b <= 3 and s <= 2:
            cnt = all_counts.get((b, s), 0)
            print(f"    {b}-{s}     {cnt:5d}    {cnt/total_ab*100:5.1f}%")

    print("\n" + sep)
    print("  REALISM ASSESSMENT")
    print(sep)

    issues = []
    ok = []

    if 0.18 <= strikeout_rate <= 0.28:
        ok.append(f"Strikeout rate {strikeout_rate*100:.1f}% is within MLB range (18-28%)")
    else:
        issues.append(f"Strikeout rate {strikeout_rate*100:.1f}% is OUTSIDE MLB range (~23%)")

    if 0.06 <= walk_rate <= 0.12:
        ok.append(f"Walk rate {walk_rate*100:.1f}% is within MLB range (6-12%)")
    else:
        issues.append(f"Walk rate {walk_rate*100:.1f}% is OUTSIDE MLB range (~8-9%)")

    if 0.60 <= inplay_rate <= 0.78:
        ok.append(f"Ball-in-play rate {inplay_rate*100:.1f}% is within MLB range (60-78%)")
    else:
        issues.append(f"Ball-in-play rate {inplay_rate*100:.1f}% is OUTSIDE MLB range (~70%)")

    if 3.4 <= avg_pitches <= 4.6:
        ok.append(f"Avg pitches/PA {avg_pitches:.2f} is within MLB range (3.4-4.6)")
    else:
        issues.append(f"Avg pitches/PA {avg_pitches:.2f} is OUTSIDE MLB range (~3.9)")

    if 0.52 <= fp_strike_rate <= 0.68:
        ok.append(f"First-pitch strike rate {fp_strike_rate*100:.1f}% is within MLB range (52-68%)")
    else:
        issues.append(f"First-pitch strike rate {fp_strike_rate*100:.1f}% is OUTSIDE MLB range (~60%)")

    if ok:
        print("\n  REALISTIC:")
        for msg in ok:
            print(f"    [OK] {msg}")

    if issues:
        print("\n  CONCERNS:")
        for msg in issues:
            print(f"    [!!] {msg}")

    if not issues:
        print("\n  All key metrics are within realistic MLB bounds.")

    print()


if __name__ == "__main__":
    main()
