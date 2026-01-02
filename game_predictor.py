"""EFL Championship Game Predictor"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

def load_data():
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
    scores["y_home_win"] = np.where(  # 1 if home win, 0 if away win, 0.5 if draw
        played & (scores['HomeScore'] > scores['AwayScore']),
        1.0,
        np.nan,
    )
    scores.loc[
        played & (scores['HomeScore'] < scores['AwayScore']),
        'y_home_win'
    ] = 0.0
    scores.loc[
        played & (scores['HomeScore'] == scores['AwayScore']),
        'y_home_win'
    ] = 0.5
    scores["y_home_win_binary"] = np.where(
    played & (scores["HomeScore"] > scores["AwayScore"]),
    1,
    0
)
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

def build_training_matrix(games_elo, hist_roll, window = 5):

    """Build training matrix from games with Elo and team history with rollups."""
    played = games_elo[games_elo['y_home_win'].notna()].copy().sort_values(by=['Date', 'game_id'])
    home_rows = hist_roll[hist_roll['is_home'] == 1].set_index('game_id')
    away_rows = hist_roll[hist_roll['is_home'] == 0].set_index('game_id')

    played.reset_index(drop=True, inplace=True)
    X = pd.DataFrame(index=played.index)

    for feat in FORM_FEATURES:
        for suffix in ['s2d_pre', f'l{window}_pre']:
            col = f"{feat}_{suffix}"
            X[f"{col}_diff"] = (
                home_rows.loc[played['game_id'], col].values
                - away_rows.loc[played['game_id'], col].values
            )
    X['rest_days_capped_diff'] = (
        home_rows.loc[played['game_id'], 'rest_days_capped'].values
        - away_rows.loc[played['game_id'], 'rest_days_capped'].values
        )
    X['elo_diff'] = (
        played['home_elo_pre'].values
        - played['away_elo_pre'].values
    )
    X['Season'] = played['Season'].values

    y = played['y_home_win_binary'].astype(float).values

    meta = played[['game_id', 'Date', 'Home Team', 'Away Team', 'y_home_win']].copy()
    return X, y, meta

def train_logistic_model(X, y, meta):

  # Remove rows with any NaN values
    valid_mask = X.notna().all(axis=1)
    
    X = X[valid_mask]
    y = y[valid_mask]
    meta = meta[valid_mask]

    train_mask = X['Season'] <= 2023
    X_train = X[train_mask].drop(columns=['Season'])
    y_train = y[train_mask]

    test_mask = X['Season'] >= 2024
    X_test = X[test_mask].drop(columns=['Season'])
    y_test = y[test_mask]

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)[:, 1]

    importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Weight': model.coef_[0]
})
    return model, y_probs, y_test, meta, importance

def main():
    games = load_data()
    games_elo = add_elo_pregame(games)
    team_hist = build_team_history(games_elo)
    team_hist = add_rollups(team_hist, window=5)

    X, y, meta = build_training_matrix(games_elo, team_hist, window=5)

    model, probs, y_true, meta, feature_importance = train_logistic_model(X, y, meta)

    test_meta = meta[X['Season'] >= 2024].copy()
    test_meta['win_prob'] = probs

    print(test_meta.sort_values('win_prob', ascending=False).head(50))
    print(f"Log Loss: {log_loss(y_true, probs):.4f}")
    print(feature_importance.sort_values('Weight', ascending=False))


if __name__ == "__main__":
    main()