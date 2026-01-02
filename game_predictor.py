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
    scores["y_home_win"] = np.where( # 1 if home win, 0 if away win, 0.5 if draw
    played & (scores['HomeScore'] > scores['AwayScore']).astype(float),  1.0, 0.0
    )
    scores.loc[
        played & (scores['HomeScore'] == scores['AwayScore']),
        'y_home_win'
    ] = 0.5
    scores = scores.sort_values(by=['Date']).reset_index(drop=True)
    scores["game_id"] = (np.arange(len(scores), dtype =int))

    return scores

def add_elo_pregame(games, k=20, hfa=55.0, regress=0.75):
    g = games.sort_values(by=["Date", "game_id"]).copy()
    g['home_elo_pre'] = np.nan
    g['away_elo_pre'] = np.nan

    elo = {}
    last_season = None

    for idx, row in g.iterrows():
        season = int(row['Season'])
        if last_season is None:
            last_season = season
        if season != last_season:
            for team in elo:
                elo[team] = regress * 1500.0 + (1 - regress) * elo[team]   
            last_season = season

        home = row['HomeTeam']
        away = row['AwayTeam']
        eh = elo.get(home, 1500.0)
        ea = elo.get(away, 1500.0)

        g.at[idx, 'home_elo_pre'] = eh
        g.at[idx, 'away_elo_pre'] = ea

        if pd.notna(row['y_home_win']):
            expected_home = 1 / (1 + 10 ** ((ea - (eh + hfa)) / 400))
            actual_home = row['y_home_win']
            elo[home] = eh + k * (actual_home - expected_home)
            elo[away] = ea + k * ((1 - actual_home) - (1 - expected_home))
           
    return g

GAMES = None
GAMES = load_data(GAMES)
print(GAMES.head())
