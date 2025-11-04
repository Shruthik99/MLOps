import pandas as pd

# Load original data
df = pd.read_csv('data/wine_quality_raw.csv')

print(f"Original dataset shape: {df.shape}")

# Filter: Keep only wines with alcohol > 10%
df_filtered = df[df['alcohol'] > 10].copy()

print(f"Filtered dataset shape: {df_filtered.shape}")
print(f"Removed {len(df) - len(df_filtered)} samples")

# Save updated dataset
df_filtered.to_csv('data/wine_quality_raw.csv', index=False)

print("✓ Dataset updated!")