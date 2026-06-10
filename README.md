# PROJECT: World Cup Outcome Predictor

A machine learning project using historical football match data, FIFA rankings, and squad valuations to predict World Cup match outcomes.

## Motivation
Predicting football match outcomes is a classic challenge in sports analytics. While many factors influence a match result (form, quality, home advantage, squad composition), this project aims to identify which variables have the strongest predictive power. By building a logistic regression model, we can quantify the impact of different team characteristics and potentially predict future World Cup results.

## Data Sources
The analysis combines three primary data sources:

- **Match results**: Historical football matches from 1930 to 2025, including World Cups, continental championships, and competitive qualifiers (friendlies excluded)
  - Source: Kaggle/Football dataset
  - Fields: date, home/away teams, scores, tournament type, venue

- **FIFA World Rankings**: Monthly rankings for each country (2022-2026)
  - Scraped from official FIFA website
  - Ensures ranking data is relevant to match date

- **Squad data**: Average age and market value for national teams
  - Scraped from Transfermarkt
  - Current squads used as proxy (limitation: historical squad data unavailable)

## Dataset Preparation

### Filtering
- Only matches from 2022 onwards (post last World Cup)
- Excluding friendly matches (low competitive intensity)
- Keeping only teams with complete FIFA ranking data

### Feature Engineering
For each match, we calculate four difference-based features:

| Feature | Calculation | Interpretation |
|---------|-------------|----------------|
| **Rank difference** | Home rank - Away rank | Negative = home team better ranked |
| **Age difference** | Home avg age - Away avg age | Positive = home team has older squad |
| **Value difference** | Home market value - Away market value | Positive = home squad more valuable |
| **Form difference** | Home form points - Away form points | Positive = home team in better recent form |

**Form calculation**: Rolling sum of points from last 5 competitive matches (3 for win, 1 for draw, 0 for loss)

**Target variable**: 
- 1 = Home win
- 0 = Draw  
- 2 = Away win

## How to Run

Clone this repository:
```bash
git clone https://github.com/modrazz/world-cup-predictor
cd world-cup-predictor
