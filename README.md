# MLB Sports Prediction AI

A machine learning project that predicts MLB game outcomes using historical team performance data.

## Project Overview

The goal of this project is to build a machine learning pipeline capable of estimating the probability of an MLB team winning a game.

The project currently uses historical MLB game data from the 2024 season and a Logistic Regression classification model.

## Machine Learning Pipeline

Historical MLB Data  
↓  
Data Collection  
↓  
Data Cleaning  
↓  
Game Reconstruction  
↓  
Feature Engineering  
↓  
Train/Test Split  
↓  
Logistic Regression  
↓  
Game Outcome Prediction

## Features

The current model uses pre-game information including:

- Team historical win percentage
- Average runs scored
- Average runs allowed
- Recent win percentage
- Home-team performance
- Away-team performance

The features are calculated using information available before each game to reduce data leakage.

## Dataset

The current dataset contains:

- Season: 2024
- Games: 2,413
- Teams: 30
- Complete games reconstructed from team-level results

The raw and processed datasets are excluded from Git using `.gitignore`.

## Model

The first version uses Logistic Regression from Scikit-learn.

### Training

- Training games: 1,930
- Testing games: 483

### Results

| Metric | Result |
|---|---:|
| Accuracy | 57.97% |
| Log Loss | 0.6849 |

The results represent the initial baseline model. Future versions will attempt to improve predictive performance using additional seasons and more detailed player and team statistics.

## Project Structure

sports-ai/
│
├── README.md
├── .gitignore
│
├── src/
│   ├── collect_data.py
│   ├── clean_data.py
│   ├── features.py
│   ├── train_model.py
│   ├── inspect_data.py
│   ├── check_dates.py
│   ├── teams.py
│   ├── test_dates.py
│   └── test_pandas.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
└── predictions/

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- pybaseball
- Git
- GitHub

## How to Run

Clone the repository:

    git clone https://github.com/RaminLouis4353/sports-ai.git
    cd sports-ai

Create a virtual environment:

    python -m venv .venv

Activate it on Windows PowerShell:

    .\.venv\Scripts\Activate.ps1

Install the required packages:

    pip install pandas numpy scikit-learn matplotlib pybaseball

Run the data pipeline:

    python .\src\collect_data.py
    python .\src\clean_data.py
    python .\src\features.py

Train the model:

    python .\src\train_model.py

## Future Improvements

Planned improvements include:

- Add multiple MLB seasons
- Add starting pitcher statistics
- Add player-level performance
- Add offensive statistics
- Add bullpen statistics
- Add home/away splits
- Add recent team form
- Compare Logistic Regression with Random Forest
- Test Gradient Boosting models
- Improve probability calibration
- Build a prediction interface
- Track predictions against actual results
- Create visualizations and dashboards

## Disclaimer

This project is intended for educational and machine learning research purposes. Sports outcomes are inherently uncertain, and model predictions are not guaranteed to be correct.