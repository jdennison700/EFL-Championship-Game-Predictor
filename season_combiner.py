import pandas as pd
import glob
import re

# Find all CSV files that follow the pattern
csv_files = glob.glob('championship-*-GMTStandardTime.csv')

all_dfs = []

for file in csv_files:
    # Extract the year from the filename using regex (e.g., 2019 from championship-2019-...)
    match = re.search(r'championship-(\d{4})', file)
    if match:
        year = match.group(1)
        # Often seasons are referred to as 2019/20, but for now I'll use the start year
        season_label = f"{year}"
    else:
        season_label = "Unknown"
    
    temp_df = pd.read_csv(file)
    temp_df['Season'] = season_label
    all_dfs.append(temp_df)

if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv('combined_championship_seasons_2019-2025.csv', index=False)
    print("Combined CSV created successfully.")
    print(combined_df.head())
    print(f"Total rows: {len(combined_df)}")
else:
    print("No matching CSV files found.")