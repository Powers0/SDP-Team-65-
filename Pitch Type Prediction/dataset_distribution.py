import pandas as pd

# Load CSVs
df_2021 = pd.read_csv("statcast_2021.csv")
df_2022 = pd.read_csv("statcast_2022.csv")
df_2023 = pd.read_csv("statcast_2023.csv")
df_2024 = pd.read_csv("statcast_2024.csv")

# Combine into a single DataFrame
df = pd.concat([df_2021, df_2022, df_2023, df_2024], ignore_index=True)
print(f"Combined data shape: {df.shape}")

# Count pitch types
pitch_counts = df['pitch_type'].value_counts(dropna=False)

# Calculate percentages
pitch_percentages = (pitch_counts / len(df)) * 100

# Print results
print("Pitch Type Counts:\n", pitch_counts)
print("\nPitch Type Percentages:\n", pitch_percentages)