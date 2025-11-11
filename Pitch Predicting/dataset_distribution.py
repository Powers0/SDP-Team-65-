import pandas as pd
from pybaseball import statcast

# load your same dataset
df = statcast(start_dt='2025-09-01', end_dt='2025-09-30')  # or whatever range you used

# count pitch types
pitch_counts = df['pitch_type'].value_counts(dropna=False)

# calculate percentages
pitch_percentages = (pitch_counts / len(df)) * 100

print("Pitch Type Counts:\n", pitch_counts)
print("\nPitch Type Percentages:\n", pitch_percentages)