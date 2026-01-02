"""EFL Championship Game Predictor"""

import pandas as pd
import numpy as np

def load_data(scores=None):
    """Load and process championship data."""

    scores = pd.read_csv('Datasets/combined_championship_seasons_2019-2025.csv') 

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
    """Add pre-game Elo ratings to each game."""
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

        home = row['Home Team']
        away = row['Away Team']
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
#Features to compute rolling statistics for
FORM_FEATURES = ["win", "goals_for", "goals_against", "goal_difference"]

def build_team_history(games_elo):
    """Build team history from game data with Elo ratings."""
    played = games_elo[games_elo['y_home_win'].notna()].copy()

    home =pd.DataFrame({
        "game_id": played['game_id'].values,
        "date": played['Date'].values,
        "season": played['Season'].values,
        "team": played['Home Team'].values,
        "is_home": 1,
        "goals_for": played['HomeScore'].values,
        "goals_against": played['AwayScore'].values,
    })
    away = pd.DataFrame({
        "game_id": played['game_id'].values,
        "date": played['Date'].values,
        "season": played['Season'].values,
        "team": played['Away Team'].values,
        "is_home": 0,
        "goals_for": played['AwayScore'].values,
        "goals_against": played['HomeScore'].values,
    })

    hist = pd.concat([home, away], ignore_index=True)
    hist['win'] = (hist['goals_for'] > hist['goals_against']).astype(float)
    hist['goal_difference'] = hist['goals_for'] - hist['goals_against']

    hist = hist.sort_values(by=['team', 'date', 'game_id']).reset_index(drop=True)
    hist['rest_days'] = hist.groupby("team")['date'].diff().dt.days
    hist['rest_days_capped'] = hist['rest_days'].clip(lower=0, upper=21)
    return hist

def add_rollups(hist, window=5):
    """Add rolling statistics to team history."""
    d = hist.sort_values(["season", "team", "date", "game_id"]).copy()
    g = d.groupby(["season", "team"], sort=False)

    for feat in FORM_FEATURES:
        d[f"{feat}_s2d_pre"] = g[feat].apply(lambda s: s.expanding().mean().shift(1)).reset_index(level=[0, 1], drop=True)
        d[f"{feat}_l{window}_pre"] = g[feat].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean()).reset_index(level=[0, 1], drop=True)

    return d
GAMES = None
GAMES = load_data(GAMES)
GAMES_ELO = add_elo_pregame(GAMES)
TEAM_HIST = build_team_history(GAMES_ELO)   
TEAM_HIST = add_rollups(TEAM_HIST, window=5)
print(TEAM_HIST.head())
