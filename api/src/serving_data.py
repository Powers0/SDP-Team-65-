import pandas as pd

# Columns that (when present) define chronological order for sequence windows
SEQ_ORDER_COLS = ["pitcher", "game_date", "game_pk", "at_bat_number", "pitch_number"]


def load_serving_table(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if all(c in df.columns for c in SEQ_ORDER_COLS):
        df = df.sort_values(SEQ_ORDER_COLS)
    return df


def _sort_serving(df: pd.DataFrame) -> pd.DataFrame:
    """Sort a serving table/window in chronological order when possible."""
    cols = [c for c in SEQ_ORDER_COLS if c in df.columns]
    if cols:
        return df.sort_values(cols)
    return df


def _take_tail(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0:
        return df.iloc[0:0]
    if len(df) <= n:
        return df
    return df.tail(n)


def get_latest_window(df, pitcher_mlbam: int, batter_mlbam: int, seq_len: int):
    """Backward-compatible: strict matchup-only window."""
    sub = df[(df["pitcher"] == pitcher_mlbam) & (df["batter"] == batter_mlbam)]
    if len(sub) < seq_len:
        raise ValueError("Not enough pitches in this matchup to build a context window.")
    sub = _sort_serving(sub).tail(seq_len)
    return sub


def get_window_with_fallback(df: pd.DataFrame, pitcher_mlbam: int, batter_mlbam: int, seq_len: int):
    """
    Always returns an EXACTLY `seq_len` window by falling back when the matchup is sparse.

    Priority:
      1) matchup (pitcher+batter)
      2) pitcher-only
      3) batter-only
      4) global

    Returns:
      window_df (seq_len rows),
      label (e.g., "matchup", "matchup+pitcher", "pitcher+global").
    """

    df = _sort_serving(df)

    matchup = df[(df["pitcher"] == pitcher_mlbam) & (df["batter"] == batter_mlbam)]
    pit = df[df["pitcher"] == pitcher_mlbam]
    bat = df[df["batter"] == batter_mlbam]
    glob = df

    sources = [
        ("matchup", matchup),
        ("pitcher", pit),
        ("batter", bat),
        ("global", glob),
    ]

    parts = []
    used = {"matchup": 0, "pitcher": 0, "batter": 0, "global": 0}
    remaining = seq_len

    # Best-effort dedupe key (prevents pulling the same pitch twice across sources)
    unique_cols = [c for c in ["game_pk", "at_bat_number", "pitch_number"] if c in df.columns]

    taken = pd.DataFrame()

    for name, sub in sources:
        if remaining <= 0:
            break
        if sub is None or len(sub) == 0:
            continue

        sub = _sort_serving(sub)

        # Remove already-taken rows if we have a dedupe key
        if not taken.empty and unique_cols:
            sub = sub.merge(
                taken[unique_cols].drop_duplicates(),
                on=unique_cols,
                how="left",
                indicator=True,
            )
            sub = sub[sub["_merge"] == "left_only"].drop(columns=["_merge"])

        chunk = _take_tail(sub, remaining)
        if len(chunk) > 0:
            parts.append(chunk)
            taken = pd.concat([taken, chunk], ignore_index=True)
            used[name] += len(chunk)
            remaining -= len(chunk)

    if remaining > 0:
        raise ValueError(f"Serving table has fewer than seq_len={seq_len} total usable pitches.")

    window = pd.concat(parts, ignore_index=True)
    window = _take_tail(window, seq_len)

    label = "+".join([k for k in ["matchup", "pitcher", "batter", "global"] if used.get(k, 0) > 0])

    return window, label