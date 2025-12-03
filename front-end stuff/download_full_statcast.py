import pandas as pd
import datetime as dt
from pybaseball import statcast
from time import sleep


def download_season(year):
    print(f"\n==== DOWNLOADING FULL MLB SEASON {year} ====")

    start = dt.date(year, 3, 15)   # captures early ST / edge cases
    end   = dt.date(year, 11, 15)  # captures postseason

    all_days = []
    current = start

    while current <= end:
        next_day = current + dt.timedelta(days=1)
        start_str = current.strftime("%Y-%m-%d")
        end_str = next_day.strftime("%Y-%m-%d")

        print(f"Fetching {start_str} ... ", end="", flush=True)

        try:
            df_day = statcast(start_str, end_str)
            print(f"{len(df_day)} rows.")
            if len(df_day) > 0:
                all_days.append(df_day)
        except Exception as e:
            print(f"ERROR: {e}")
            sleep(2)
            continue

        sleep(0.5)  # avoid rate limits
        current = next_day

    df = pd.concat(all_days, ignore_index=True)
    df.drop_duplicates(inplace=True)

    outname = f"statcast_full_{year}.csv"
    df.to_csv(outname, index=False)

    print(f"SAVED {outname}: {df.shape[0]:,} rows")


if __name__ == "__main__":
    for y in [2021, 2022, 2023, 2024]:
        download_season(y)