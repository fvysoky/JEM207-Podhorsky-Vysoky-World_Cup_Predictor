# 2026 FIFA World Cup Predictive Model & Simulator

An end-to-end machine learning pipeline and deterministic tournament simulator for the 2026 FIFA World Cup. This project scrapes financial and ranking data, merges it with historical match results, trains a calibrated predictive model on international match outcomes (1X2 market), and simulates the expanded 48-team tournament bracket.

## Project Architecture

The repository is structured into three distinct phases: Data Engineering, Predictive Modeling, and Tournament Simulation.

### 1. Data Engineering & Scraping
The foundation of the model relies on merging point-in-time financial and ranking metrics with historical match results. 
* **Raw Match Data:** The base match history (`results.csv`) is sourced from Kaggle, containing official international football matches from the 1930s to the present. **Note:** To maintain relevance to modern tactical eras and current squad compositions, the operational training window is strictly filtered to matches from **2022 onward** (last 2022 FIFA World Cup).
* **FIFA Ranking Scrapers (`FIFA_202x_v5.ipynb`):** Automated web scrapers utilizing Playwright and BeautifulSoup to extract official FIFA Men's World Rankings from 2022 to the present.
* **Transfermarkt Scraper (`Transfermarkt_scraper_v6.ipynb`):** Extracts national team squad market values (in Euros) and average squad ages.
* **Dataset Builder (`df_fin_v1.ipynb`):** Merges the disparate data sources. Standardizes country naming conventions across platforms, removes non-standard friendly matches, and calculates rolling 5-match team form. 

### 2. Predictive Modeling (`wc_predictor_v3_classes.py`)
The model evaluates the classic 1X2 betting market (Home Win, Draw, Away Win) using an eXtreme Gradient Boosting classifier.

**Methodology:**
* **Feature Engineering:** Uses differential features (`rank_difference`, `value_difference`, `form_difference`, `age_difference`) and a binary `home_advantage` flag (resolving neutral venue bias). 
* **Algorithm:** `XGBClassifier` constrained by `max_depth=4` and `learning_rate=0.05` to prevent overfitting on historical noise. 
* **Validation:** Implements a strict **chronological 80/20 train/test split**. Random splitting is explicitly avoided to prevent temporal data leakage (predicting past matches using future knowledge).
* **Probability Calibration:** Tree-based models inherently distort minority class distributions (e.g., Draws). The base XGBoost model is wrapped in scikit-learn’s `CalibratedClassifierCV` using Sigmoid (Platt) scaling. This forces the softmax outputs to mirror real-world historical frequencies, ensuring the probabilities are mathematically viable for Expected Value (+EV) calculations in the Bet Builder UI.

### 3. Tournament Simulation
The simulation engine translates the calibrated probabilities into a progressing 48-team bracket based on the official 2026 format.

* **Group Stage:** 12 groups of 4 teams. Match outcomes are predicted deterministically. 
* **Tiebreaker Logic:** The simulator utilizes a tiered tiebreaker: Points ➔ Two-way Head-to-Head ➔ FIFA Rank fallback. (See *Scope Constraints* below regarding Goal Difference).
* **Third-Place Advancement:** Safely aggregates the standings across all groups and assigns the 8 best third-place teams to their official FIFA downstream match slots.
* **Knockout Stage:** Forces binary progression. If the model's highest probability for a knockout match is a Draw at 90 minutes, the simulator forces advancement based on the team with the higher baseline win probability (`prob_home_win` vs `prob_away_win`).

## Scope Constraints & Future Work
* **Target Variable Isolation (No Goal Vectors):** This project intentionally bounds its scope to evaluating the classic 1X2 probability market. Because the model predicts discrete outcomes rather than exact goal vectors, traditional Goal Difference is unavailable for group stage tiebreakers. Reintroducing Goal Difference would require a secondary Poisson distribution model for Expected Goals (xG), which introduces complex variance outside the scope of evaluating the primary XGBoost classification engine. 
* **Static Features:** Currently, `form_difference` and `value_difference` remain static as simulated teams advance deep into the tournament. Future iterations must dynamically update team form post-group stage.

## Setup & Execution

**Requirements:**
```bash
pip install pandas numpy scikit-learn xgboost playwright beautifulsoup4
playwright install chromium
