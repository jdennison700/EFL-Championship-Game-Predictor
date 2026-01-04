"""
Combine multiple CSV files representing different seasons of a championship into a single CSV file.
"""

import glob
import re
import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def get_latest_2025_data():
    """
    Downloads latest 2025 data to datasets folder
    """
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    target_folder = os.path.join(repo_dir, 'Datasets')
    file_name = "championship-2025-GMTStandardTime.csv"
    target_file_path = os.path.join(target_folder, file_name)

    if os.path.exists(target_file_path):
        print(f"Existing file found. Deleting {file_name} to replace it...")
        os.remove(target_file_path)

    chrome_options = Options()
    chrome_options.add_argument("--headless")

    prefs = {
        "download.default_directory": target_folder,
        "download.prompt_for_download": False,
        "directory_upgrade": True
    }
    chrome_options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=chrome_options)
    url = "https://fixturedownload.com/download/csv/championship-2025"

    print("Opening browser and waiting for download...")
    driver.get(url)

    time.sleep(5)
    driver.quit()
    print("Downloaded latest 2025 championship data to {target_folder} complete")

def make_combined_csv():
    """
    Combines all seasons into one csv
    """

    # Find all CSV files that follow the pattern
    csv_files = glob.glob('Datasets/championship-*-GMTStandardTime.csv')

    all_dfs = []

    for file in csv_files:
        # Extract the year from the filename using regex (e.g., 2019 from championship-2019-...)
        match = re.search(r'championship-(\d{4})', file)
        if match:
            year = match.group(1)
            # Often seasons are referred to as 2019/20, but for now I'll use the start year
            SEASON_LABEL = f"{year}"
        else:
            SEASON_LABEL = "Unknown"

        temp_df = pd.read_csv(file)
        temp_df['Season'] = SEASON_LABEL
        all_dfs.append(temp_df)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv('Datasets/combined_championship_seasons_2019-2025.csv', index = False)
        print("Combined CSV created successfully.")
        print(combined_df.head())
        print(f"Total rows: {len(combined_df)}")
    else:
        print("No matching CSV files found.")


