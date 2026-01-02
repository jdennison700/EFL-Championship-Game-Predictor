"""EFL Championship Game Predictor"""

import pandas as pd
import numpy as np

def load_data(scores=None):
    """Load and process championship data."""

    scores = pd.read_csv('combined_championship_seasons_2019-2025.csv') 

    scores['Date'] = pd.to_datetime(scores['Date'], dayfirst=True)
    scores[['HomeScore', 'AwayScore']] = (
        scores['Result'].str.extract(r'(\d+)\s*-\s*(\d+)').astype(float)
    )
    played = (
        scores['HomeScore'].notna()
        & scores['AwayScore'].notna()
    )
    scores["y_home_win"] = np.where( # 1 if home win, 0 otherwise
    played & (scores['HomeScore'] > scores['AwayScore']).astype(int),  1, 0
    )
    scores = scores.sort_values(by=['Date']).reset_index(drop=True)
    scores["game_id"] = (np.arange(len(scores), dtype =int))

    return scores


GAMES = None
GAMES = load_data(GAMES)
