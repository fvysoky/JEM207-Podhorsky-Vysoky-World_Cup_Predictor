# PROJECT: Predicting and Simulating the 2026 FIFA World Cup

A machine learning project using historical international football results, financial metrics, and official rankings to predict match outcomes and simulate the expanded 48-team 2026 World Cup bracket.

## Motivation
The purpose of a predictive sports model is to identify profitable inefficiencies in the betting market and forecast tournament progression. With the 2026 FIFA World Cup expanding to 48 teams, the complexity of the tournament—including the introduction of a Round of 32 and best third-place advancements—makes manual forecasting nearly impossible. This project aims to combine historical match data with point-in-time financial and team-form metrics to accurately predict match probabilities, identify positive Expected Value (+EV) bets, and deterministically simulate the entire tournament bracket.

## Sources
The analysis and predictive model are based on three primary open-data sources:
- **Kaggle (`results.csv`)**: Historical international football matches from the 1930s to the present. **Note:** To maintain relevance to modern tactical eras, we strictly filtered the training data to matches from 2022 onward.
- **FIFA Official Website**: Point-in-time Men's World Rankings scraped for the years 2022–2026.
- **Transfermarkt**: Scraped squad market values (in Euros) and average squad ages.

## How to run
Clone this repository onto your computer using the following command:
```text
git clone https://github.com/fvysoky/JEM207-Podhorsky-Vysoky-World_Cup_Predictor.git
```
Open a terminal instance in the directory where the repository is cloned and run the following commands to install the necessary libraries:
```text
pip install -r requirements.txt
playwright install chromium
```
To run the data pipeline and train the model, open your Jupyter Notebook client and execute the files in this order:
1. **Scrapers:** Run `FIFA_202x_v5.ipynb` and `Transfermarkt_scraper_v6.ipynb` to gather the latest metrics.
2. **Data Builder:** Run `df_fin_v1.ipynb` to clean, merge, and output the final feature matrix.
3. **Model & Simulation:** Run `wc_predictor_v3.ipynb` and click 'Run All' to train the XGBoost model, evaluate the chronological holdout, and simulate the 2026 World Cup. 

To launch the interactive Bet Builder and Simulation Dashboard, run the following command in your terminal:
```text
streamlit run app.py
```

## Methodology
For finding the optimal predictions, we decided on five core features on which we based our machine learning model:
- **Financial dominance:** The difference in total squad market value between the two teams.
- **Team strength:** The difference in official FIFA rankings.
- **Momentum:** The difference in recent form (calculated from the points earned in the previous 5 matches).
- **Experience:** The difference in average squad age.
- **Home advantage:** A binary indicator of whether a team is playing in their home country versus a neutral venue.

Since a tournament requires both individual match predictions and structural progression, we settled on dividing the methodology into two parts: Predictive Modeling and Tournament Simulation.

### 1. Predictive Modeling (1X2 Market)
We utilized an eXtreme Gradient Boosting (`XGBClassifier`) algorithm. To prevent the model from memorizing historical noise, we restricted tree depth (`max_depth=4`) and learning rate (`learning_rate=0.05`). Crucially, we evaluated the model using a strict chronological 80/20 train/test split. We decided to exclude random splitting, reasoning that predicting past matches using future knowledge (temporal data leakage) would invalidate the model's accuracy.

Furthermore, tree-based models naturally distort probabilities for rare events (like Draws). To make the model usable for a Bet Builder, we wrapped it in a Sigmoid Calibrator (`CalibratedClassifierCV`). The tool calculates Expected Value (+EV) using the following equation:

**Expected Value = (P(Outcome) * Decimal Odds) - 1**

If the resulting EV is greater than 0, the bet is mathematically profitable in the long term.

### 2. Tournament Simulation & Scope Constraints
For this part, we used the predicted probabilities to simulate the official 2026 format. The Group Stage assigns 3 points for a predicted Win, 1 for a Draw, and 0 for a Loss. 

**Scope Constraint:** Because our project intentionally bounds its scope to evaluating the classic 1X2 probability market, we lacked the exact goal vectors required for traditional Goal Difference. Reintroducing Goal Difference would require a secondary Poisson distribution model for Expected Goals (xG), which introduces complex variance outside the scope of our primary classification engine. Therefore, we used the following tiered logic to compute the final group standings:

**Standings = 1st (Points) -> 2nd (Head-to-Head) -> 3rd (FIFA Rank)**

For the knockout stages, we routed the Top 2 teams and the 8 best 3rd-place teams into their official FIFA Round-of-32 slots. If the model predicted a knockout match would end in a Draw at 90 minutes, we deterministically advanced the team with the higher baseline win probability.

## Key findings:
### 1. Model Performance
- Our calibrated model achieved a strictly out-of-sample accuracy of **~63%** on the 3-way (1X2) international market. 
- The baseline accuracy for blindly picking the favorite in our test set was approximately 48%, proving the model successfully extracts significant predictive signal from financial and ranking differences.

### 2. Best Teams (Simulation Results)
When running the 2026 Group Stage simulation, the model cleanly identified the heavy tier-one favorites. 
- **Flawless Group Stages (9 Points):** Mexico, Canada, Brazil, USA, Netherlands, Belgium, Spain, France, Argentina, and England.
- **Tightest Groups:** Group E (Germany and Ecuador tied on 6 points, resolved by tiebreaker) and Group K (Portugal and Colombia tied on 7 points, resolved by tiebreaker).

### 3. Knockout Simulation and Predicted Champion
When evaluating the bracket pathway from the Round of 32 down to the final:
- The simulation behaves highly deterministically, heavily rewarding elite squad market values and punishing vast rank differentials. 
- **The Final:** The simulation dynamically generated a benchmark Final between **France and Brazil**. 
- **The Champion:** Driven by superior differential features, the model predicts **France** to win the 2026 FIFA World Cup, with **Spain** securing the 3rd-place play-off spot.

When removing the artificial "draw hallucinations" found in uncalibrated models, the simulation behaves highly deterministically, heavily rewarding elite squad market values and punishing vast rank differentials. 

## Authors
This project was created as part of a university course Data Processing in Python JEM207.

Authors: Lukas Daniel Podhorsky, Filip Vysoky
