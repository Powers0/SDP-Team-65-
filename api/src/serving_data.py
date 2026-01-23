import pandas as pd

SEQ_ORDER_COLS = ["pitcher", "game_date", "game_pk", "at_bat_number", "pitch_number"]

def load_serving_table(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Ensure sorting (parquet should already be sorted)
    if all(c in df.columns for c in SEQ_ORDER_COLS):
        df = df.sort_values(SEQ_ORDER_COLS)
    return df

def get_latest_window(df, pitcher_mlbam: int, batter_mlbam: int, seq_len: int):
    sub = df[(df["pitcher"] == pitcher_mlbam) & (df["batter"] == batter_mlbam)]
    if len(sub) < seq_len:
        raise ValueError("Not enough pitches in this matchup to build a context window.")
    sub = sub.tail(seq_len)
    return sub