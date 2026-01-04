# EFL Championship Game Predictor

A machine learning project designed to predict the outcomes of English Football League (EFL) Championship matches. It utilizes historical match data, custom Elo ratings, and rolling form statistics to estimate the probability of a home win.

## Overview

This project uses a **Logistic Regression** model trained on historical data (2019-2023) to predict outcomes for the 2024 season and beyond. The model considers:

*   **Elo Ratings**: Dynamic strength ratings updated after every match (including regression to the mean between seasons).
*   **Team Form**: Rolling averages for wins, goals scored, and goals conceded (both Season-to-Date and Last 5 Games).
*   **Rest Days**: Impact of fatigue based on days since the last match.
*   **Home Advantage**: Built into the Elo calculation and feature engineering.

## Files

*   `game_predictor.py`: The main application. It handles data loading, feature engineering, model training, evaluation, and the interactive prediction loop.
*   `season_combiner.py`: A utility script to merge individual season CSV files (e.g., `championship-2019-GMTStandardTime.csv`) into a single master dataset.
*   `Datasets/`: Directory containing the raw CSV data files.

## Prerequisites

*   Python 3.14.0
*   `pandas`
*   `numpy`
*   `scikit-learn`

## Usage

### 1. Data Preparation
Ensure the raw season CSV files are in the `Datasets/` folder. Run the combiner to generate the master dataset:

```bash
python season_combiner.py
```
This generates `Datasets/combined_championship_seasons_2019-2025.csv`.

### 2. Training and Prediction
Run the main script to train the model and view evaluation metrics:

```bash
python game_predictor.py
```

The script will output:
*   Log Loss score for completed games.
*   Feature importance weights.

### 3. Interactive Mode
After the model trains, the script enters an interactive loop:

*   **Specific Matchup**: Enter a `Home Team` and `Away Team` to get a win probability and current Elo ratings.
*   **Today's Games**: Type `today` when prompted for the Home Team to automatically list predictions for all games scheduled for the current date.
*   **Exit**: Type `exit` to close the program.

*Note: For "Today's Games" to work, the master CSV must contain rows for the upcoming matches with the correct Date.*
