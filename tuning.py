"""EFL Championship Game Predictor"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import season_combiner

def load_data():
    """Load and process championship data."""

    season_combiner.get_latest_2025_data()
    season_combiner.make_combined_csv()

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
    return g, elo
#Features to compute rolling statistics for
FORM_FEATURES = ["win", "goals_for", "goals_against", "goal_difference"]

def build_team_history(games_elo):
    """Build team history from game data with Elo ratings."""
    played = games_elo.copy()

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
    hist['win'] = np.where(
        hist['goals_for'].notna(),
        (hist['goals_for'] > hist['goals_against']).astype(float),
        np.nan
    )
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
    played = games_elo.copy().sort_values(by=['Date', 'game_id'])
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

    train_mask = (X['Season'] <= 2023) & meta['y_home_win'].notna()
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

def get_latest_stats(team, team_hist, window=5):
    """Get the latest rolling stats for a team."""
    # Filter for games that have been played (have a result)
    df = team_hist[(team_hist['team'] == team) & (team_hist['goals_for'].notna())].sort_values('date')
    if df.empty:
        return None
    
    last_row = df.iloc[-1]
    last_season = last_row['season']
    last_date = last_row['date']
    
    # Use only current season data for stats
    season_df = df[df['season'] == last_season]
    
    stats = {'last_date': last_date}
    for feat in FORM_FEATURES:
        stats[f"{feat}_s2d"] = season_df[feat].mean()
        stats[f"{feat}_l{window}"] = season_df[feat].tail(window).mean()
    
    return stats

def predict_interactive(model, current_elo, team_hist, upcoming_games, window=5):
    """Interactive loop for user predictions."""
    print("\n--- Interactive Match Predictor ---")
    print("Type 'exit' to quit.")
    
    while True:
        home = input("\nEnter Home Team (or 'today'): ").strip()
        if home.lower() == 'exit': break
        
        if home.lower() == 'today':
            today = pd.Timestamp.now().normalize()
            todays_games = upcoming_games[upcoming_games['Date'].dt.normalize() == today]
            
            if todays_games.empty:
                print(f"No games found for today ({today.date()}) in the dataset.")
            else:
                print(f"\n--- Games for {today.date()} ---")
                for _, row in todays_games.iterrows():
                    print(f"{row['Home Team']} vs {row['Away Team']}: Home Win Prob {row['win_prob']:.1%}")
            continue

        away = input("Enter Away Team: ").strip()
        if away.lower() == 'exit': break
        
        if home not in current_elo or away not in current_elo:
            print("Team not found in history.")
            continue
            
        h_stats = get_latest_stats(home, team_hist, window)
        a_stats = get_latest_stats(away, team_hist, window)
        
        if not h_stats or not a_stats:
            print("Insufficient history for one of the teams.")
            continue
            
        # Build feature vector
        features = {}
        for feat in FORM_FEATURES:
            features[f"{feat}_s2d_pre_diff"] = h_stats[f"{feat}_s2d"] - a_stats[f"{feat}_s2d"]
            features[f"{feat}_l{window}_pre_diff"] = h_stats[f"{feat}_l{window}"] - a_stats[f"{feat}_l{window}"]
            
        try:
            date_str = input("Enter Game Date (YYYY-MM-DD) [Default: Today]: ").strip()
            if date_str:
                game_date = pd.to_datetime(date_str)
            else:
                game_date = pd.Timestamp.now().normalize()
        except (ValueError, TypeError):
            print("Invalid date.")
            continue
            
        h_rest = (game_date - h_stats['last_date']).days
        a_rest = (game_date - a_stats['last_date']).days
        features['rest_days_capped_diff'] = np.clip(h_rest, 0, 21) - np.clip(a_rest, 0, 21)
        
        features['elo_diff'] = current_elo[home] - current_elo[away]
        
        # Create DataFrame with correct column order matching training
        feature_order = []
        for feat in FORM_FEATURES:
            feature_order.append(f"{feat}_s2d_pre_diff")
            feature_order.append(f"{feat}_l{window}_pre_diff")
        feature_order.append('rest_days_capped_diff')
        feature_order.append('elo_diff')
        
        X_input = pd.DataFrame([features])[feature_order]
        
        prob = model.predict_proba(X_input)[0][1]
        print(f"\nPrediction for {home} vs {away}:")
        print(f"Home Win Probability: {prob:.1%}")
        print(f"Home Elo: {current_elo[home]:.0f}, Away Elo: {current_elo[away]:.0f}")

def tune_hyperparameters(games, team_hist_roll):
    """Grid search for best Elo parameters."""
    import itertools
    
    # Define parameter grid
    k_values = [15, 20, 25, 30]
    hfa_values = [40, 55, 70]
    regress_values = [0.6, 0.75, 0.9]
    
    best_loss = float('inf')
    best_params = (20, 55.0, 0.75)
    
    print(f"\n--- Starting Hyperparameter Tuning ---")
    
    for k, hfa, regress in itertools.product(k_values, hfa_values, regress_values):
        # Recalculate Elo
        games_elo, _ = add_elo_pregame(games, k=k, hfa=hfa, regress=regress)
        
        # Rebuild training matrix
        X, y, meta = build_training_matrix(games_elo, team_hist_roll, window=5)
        
        # Train model
        model, probs, y_test_actual, meta_valid, _ = train_logistic_model(X, y, meta)
        
        # Evaluate on completed test games
        test_meta = meta_valid[X['Season'] >= 2024]
        completed_mask = test_meta['y_home_win'].notna()
        
        if completed_mask.sum() > 0:
            loss = log_loss(y_test_actual[completed_mask], probs[completed_mask])
            if loss < best_loss:
                best_loss = loss
                best_params = (k, hfa, regress)
                print(f"New Best: k={k}, hfa={hfa}, regress={regress} -> Log Loss: {loss:.5f}")

    print(f"\nBest Parameters: k={best_params[0]}, hfa={best_params[1]}, regress={best_params[2]}")
    return best_params

def main():
    games = load_data()
    games_elo, current_elo = add_elo_pregame(games)
    team_hist = build_team_history(games_elo)
    team_hist = add_rollups(team_hist, window=5)

    best_k, best_hfa, best_regress = tune_hyperparameters(games, team_hist)
    games_elo, current_elo = add_elo_pregame(games, k=best_k, hfa=best_hfa, regress=best_regress)

    X, y, meta = build_training_matrix(games_elo, team_hist, window=5)

    model, probs, y_true, meta, feature_importance = train_logistic_model(X, y, meta)

    test_meta = meta[X['Season'] >= 2024].copy()
    test_meta['win_prob'] = probs

    # Separate upcoming games from played games
    # upcoming_mask = test_meta['y_home_win'].isna()

    # if not test_meta[~upcoming_mask].empty:
    #     print(f"\nLog Loss (Completed Games): {log_loss(y_true[~upcoming_mask], probs[~upcoming_mask]):.4f}")

    # Start interactive mode
    # predict_interactive(model, current_elo, team_hist, test_meta[upcoming_mask], window=5)

if __name__ == "__main__":
    main()