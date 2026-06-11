"""Model training and 2026 World Cup tournament simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover
    raise ImportError("xgboost is required. Install it with: python -m pip install xgboost") from exc


# Teams used for the 2026 World Cup group-stage simulation.
OFFICIAL_GROUPS_2026: Dict[str, List[str]] = {
    "A": ["Mexico", "South Africa", "Korea Republic", "Czechia"],
    "B": ["Canada", "Qatar", "Switzerland", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Côte d'Ivoire", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "IR Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Official Round-of-32 slots and eligible third-place groups.
ROUND_OF_32_SLOTS: Tuple[Dict[str, object], ...] = (
    {"match_number": 73, "home": ("R", "A"), "away": ("R", "B")},
    {"match_number": 74, "home": ("W", "E"), "away": ("3", ("A", "B", "C", "D", "F"))},
    {"match_number": 75, "home": ("W", "F"), "away": ("R", "C")},
    {"match_number": 76, "home": ("W", "C"), "away": ("R", "F")},
    {"match_number": 77, "home": ("W", "I"), "away": ("3", ("C", "D", "F", "G", "H"))},
    {"match_number": 78, "home": ("R", "E"), "away": ("R", "I")},
    {"match_number": 79, "home": ("W", "A"), "away": ("3", ("C", "E", "F", "H", "I"))},
    {"match_number": 80, "home": ("W", "L"), "away": ("3", ("E", "H", "I", "J", "K"))},
    {"match_number": 81, "home": ("W", "D"), "away": ("3", ("B", "E", "F", "I", "J"))},
    {"match_number": 82, "home": ("W", "G"), "away": ("3", ("A", "E", "H", "I", "J"))},
    {"match_number": 83, "home": ("R", "K"), "away": ("R", "L")},
    {"match_number": 84, "home": ("W", "H"), "away": ("R", "J")},
    {"match_number": 85, "home": ("W", "B"), "away": ("3", ("E", "F", "G", "I", "J"))},
    {"match_number": 86, "home": ("W", "J"), "away": ("R", "H")},
    {"match_number": 87, "home": ("W", "K"), "away": ("3", ("D", "E", "I", "J", "L"))},
    {"match_number": 88, "home": ("R", "D"), "away": ("R", "G")},
)

# Fixed bracket path after the Round of 32.
KNOCKOUT_PATHWAY: Dict[str, Tuple[Dict[str, object], ...]] = {
    "Round of 16": (
        {"match_number": 89, "home_source": 73, "away_source": 75},
        {"match_number": 90, "home_source": 74, "away_source": 77},
        {"match_number": 91, "home_source": 76, "away_source": 78},
        {"match_number": 92, "home_source": 79, "away_source": 80},
        {"match_number": 93, "home_source": 83, "away_source": 84},
        {"match_number": 94, "home_source": 81, "away_source": 82},
        {"match_number": 95, "home_source": 86, "away_source": 88},
        {"match_number": 96, "home_source": 85, "away_source": 87},
    ),
    "Quarter-final": (
        {"match_number": 97, "home_source": 89, "away_source": 90},
        {"match_number": 98, "home_source": 93, "away_source": 94},
        {"match_number": 99, "home_source": 91, "away_source": 92},
        {"match_number": 100, "home_source": 95, "away_source": 96},
    ),
    "Semi-final": (
        {"match_number": 101, "home_source": 97, "away_source": 98},
        {"match_number": 102, "home_source": 99, "away_source": 100},
    ),
    "Third-place play-off": (
        {"match_number": 103, "home_source_loser": 101, "away_source_loser": 102},
    ),
    "Final": (
        {"match_number": 104, "home_source": 101, "away_source": 102},
    ),
}


@dataclass(frozen=True)
class ProjectConfig:
    """Configuration for model training and simulation."""

    data_path: Path = Path("df_fin_v1.csv")
    output_dir: Path = Path("final_outputs_v2")
    target: str = "result"
    test_size: float = 0.20
    random_state: int = 42
    calibration_method: str = "sigmoid"
    official_groups: Dict[str, List[str]] = field(default_factory=lambda: OFFICIAL_GROUPS_2026.copy())

    # Features used as model inputs.
    base_features: Tuple[str, ...] = (
        "home_advantage",
        "rank_difference",
        "value_difference_log",
        "form_difference",
        "age_difference",
        "rank_gap",
        "value_gap_log",
        "form_gap",
        "age_gap",
    )


class FootballDataPreprocessor:
    """Loads data and creates model features."""

    # Dataset columns required for training, prediction and simulation.
    REQUIRED_COLUMNS = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "country",
        "neutral",
        "home_rank",
        "away_rank",
        "rank_difference",
        "home_avg_age",
        "away_avg_age",
        "home_market_value",
        "away_market_value",
        "age_difference",
        "value_difference",
        "home_form",
        "away_form",
        "form_difference",
        "result",
    }

    def __init__(self, config: ProjectConfig):
        self.config = config

    def load(self) -> pd.DataFrame:
        # Read the raw dataset and apply the common preparation steps.
        df = pd.read_csv(self.config.data_path)
        return self.prepare(df)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        # Clean dates, sort chronologically and create derived model features.
        self._validate_columns(df)
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

        # Convert existing team differences into model-friendly variables.
        df["home_advantage"] = (~df["neutral"].astype(bool)).astype(int)
        df["value_difference_log"] = self._signed_log1p(df["value_difference"])
        df["rank_gap"] = df["rank_difference"].abs()
        df["value_gap_log"] = df["value_difference_log"].abs()
        df["form_gap"] = df["form_difference"].abs()
        df["age_gap"] = df["age_difference"].abs()
        return df

    def historical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Keep only completed matches where the target result is known.
        features = list(self.config.base_features)
        history = df[df[self.config.target].notna()].copy()
        history = history.dropna(subset=features + [self.config.target, "date"])
        history[self.config.target] = history[self.config.target].astype(int)
        return history.sort_values("date").reset_index(drop=True)

    def future_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Keep only future fixtures where the target result is not known yet.
        features = list(self.config.base_features)
        future = df[df[self.config.target].isna()].copy()
        return future.dropna(subset=features + ["date"]).sort_values("date").reset_index(drop=True)

    def temporal_split(self, history: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Split by time, so older matches train the model and newer matches test it.
        split_idx = int(len(history) * (1 - self.config.test_size))
        return history.iloc[:split_idx].copy(), history.iloc[split_idx:].copy()

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = sorted(self.REQUIRED_COLUMNS - set(df.columns))
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

    @staticmethod
    def _signed_log1p(series: pd.Series) -> pd.Series:
        return np.sign(series) * np.log1p(np.abs(series))


class FootballOutcomeModel:
    """Calibrated XGBoost classifier for home win, draw and away win."""

    CLASS_ORDER = [0, 1, 2]
    CLASS_COLUMNS = {0: "prob_draw", 1: "prob_home_win", 2: "prob_away_win"}

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.model: Optional[CalibratedClassifierCV] = None
        self.feature_names = list(config.base_features)

    def build_estimator(self) -> Pipeline:
        # Scale numeric features and pass them into the XGBoost classifier.
        preprocessor = ColumnTransformer(
            transformers=[("scale", StandardScaler(), self.feature_names)],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        # Multi-class model predicts draw, home win and away win.
        xgb_model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            max_depth=3,
            learning_rate=0.05,
            n_estimators=80,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.00,
            eval_metric="mlogloss",
            random_state=self.config.random_state,
            n_jobs=1,
        )

        return Pipeline(steps=[("preprocess", preprocessor), ("model", xgb_model)])

    def fit(self, train_df: pd.DataFrame) -> "FootballOutcomeModel":
        # Fit the model and calibrate probabilities using time-series folds.
        X_train = train_df[self.feature_names]
        y_train = train_df[self.config.target].astype(int)
        calibrated = CalibratedClassifierCV(
            estimator=self.build_estimator(),
            method=self.config.calibration_method,
            cv=TimeSeriesSplit(n_splits=3),
        )
        calibrated.fit(X_train, y_train)
        self.model = calibrated
        return self

    def predict_proba(self, match_df: pd.DataFrame) -> np.ndarray:
        # Return probabilities for all three match outcomes.
        self._check_fitted()
        return self.model.predict_proba(match_df[self.feature_names])

    def predict(self, match_df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(match_df[self.feature_names])

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, object]:
        # Evaluate model quality on the held-out chronological test set.
        y_true = test_df[self.config.target].astype(int)
        y_pred = self.predict(test_df)
        y_proba = self.predict_proba(test_df)
        return {
            "test_matches": int(len(test_df)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "log_loss": float(log_loss(y_true, y_proba, labels=self.CLASS_ORDER)),
            "classification_report": classification_report(
                y_true,
                y_pred,
                labels=self.CLASS_ORDER,
                target_names=["draw", "home_win", "away_win"],
                output_dict=True,
                zero_division=0,
            ),
        }

    def add_predictions(self, match_df: pd.DataFrame) -> pd.DataFrame:
        # Add predicted labels and probabilities to a match table.
        result = match_df.copy()
        probabilities = self.predict_proba(result)
        predictions = self.predict(result)
        result["pred_result"] = predictions
        for class_index, column_name in self.CLASS_COLUMNS.items():
            probability_index = list(self.model.classes_).index(class_index)
            result[column_name] = probabilities[:, probability_index]
        result["pred_label"] = result["pred_result"].map({0: "Draw", 1: "Home win", 2: "Away win"})
        return result

    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")


class TeamFeatureStore:
    """Creates neutral-match feature rows for arbitrary team pairings."""

    def __init__(self, prepared_df: pd.DataFrame):
        self.team_table = self._build_team_table(prepared_df)

    def create_match(self, home_team: str, away_team: str, neutral: bool = True) -> pd.DataFrame:
        # Build one synthetic match row from the latest available team features.
        home = self._get_team(home_team)
        away = self._get_team(away_team)
        row = {
            "home_team": home_team,
            "away_team": away_team,
            "neutral": neutral,
            "home_advantage": int(not neutral),
            "home_rank": home["rank"],
            "away_rank": away["rank"],
            "rank_difference": home["rank"] - away["rank"],
            "home_avg_age": home["avg_age"],
            "away_avg_age": away["avg_age"],
            "age_difference": home["avg_age"] - away["avg_age"],
            "home_market_value": home["market_value"],
            "away_market_value": away["market_value"],
            "value_difference": home["market_value"] - away["market_value"],
            "home_form": home["form"],
            "away_form": away["form"],
            "form_difference": home["form"] - away["form"],
        }
        row["value_difference_log"] = np.sign(row["value_difference"]) * np.log1p(abs(row["value_difference"]))
        row["rank_gap"] = abs(row["rank_difference"])
        row["value_gap_log"] = abs(row["value_difference_log"])
        row["form_gap"] = abs(row["form_difference"])
        row["age_gap"] = abs(row["age_difference"])
        return pd.DataFrame([row])

    def _get_team(self, team: str) -> pd.Series:
        if team not in self.team_table.index:
            raise KeyError(f"No features available for team: {team}")
        return self.team_table.loc[team]

    @staticmethod
    def _build_team_table(df: pd.DataFrame) -> pd.DataFrame:
        # Combine home and away rows into one latest-feature table per team.
        home = df[["home_team", "home_rank", "home_avg_age", "home_market_value", "home_form"]].rename(
            columns={
                "home_team": "team",
                "home_rank": "rank",
                "home_avg_age": "avg_age",
                "home_market_value": "market_value",
                "home_form": "form",
            }
        )
        away = df[["away_team", "away_rank", "away_avg_age", "away_market_value", "away_form"]].rename(
            columns={
                "away_team": "team",
                "away_rank": "rank",
                "away_avg_age": "avg_age",
                "away_market_value": "market_value",
                "away_form": "form",
            }
        )
        teams = pd.concat([home, away], ignore_index=True).dropna()
        return teams.groupby("team", as_index=True).last()


class ExpectedValueCalculator:
    """Bet Builder helper for decimal odds and calibrated model probabilities."""

    @staticmethod
    def expected_value(probability: float, decimal_odds: float) -> float:
        # Positive value means the model probability is higher than the odds imply.
        if decimal_odds <= 1:
            raise ValueError("Decimal odds must be greater than 1.")
        return probability * decimal_odds - 1

    @staticmethod
    def break_even_probability(decimal_odds: float) -> float:
        if decimal_odds <= 1:
            raise ValueError("Decimal odds must be greater than 1.")
        return 1 / decimal_odds

    @classmethod
    def evaluate_selection(cls, probability: float, decimal_odds: float) -> Dict[str, float | bool]:
        implied = cls.break_even_probability(decimal_odds)
        ev = cls.expected_value(probability, decimal_odds)
        return {
            "model_probability": probability,
            "bookmaker_implied_probability": implied,
            "expected_value": ev,
            "is_positive_ev": ev > 0,
        }


class WorldCupSimulator:
    """Deterministic tournament simulator using official group and bracket structure."""

    def __init__(
        self,
        model: FootballOutcomeModel,
        team_store: TeamFeatureStore,
        official_groups: Optional[Dict[str, List[str]]] = None,
    ):
        self.model = model
        self.team_store = team_store
        self.official_groups = official_groups or OFFICIAL_GROUPS_2026

    def validate_future_matches(self, future_matches: pd.DataFrame) -> None:
        """Fail loudly if future fixture names do not match the hardcoded official groups."""
        fixture_teams = set(future_matches["home_team"]).union(set(future_matches["away_team"]))
        official_teams = {team for teams in self.official_groups.values() for team in teams}
        missing_from_fixtures = sorted(official_teams - fixture_teams)
        unknown_in_fixtures = sorted(fixture_teams - official_teams)
        if missing_from_fixtures or unknown_in_fixtures:
            raise ValueError(
                "Future fixtures do not match OFFICIAL_GROUPS_2026. "
                f"Missing from fixtures: {missing_from_fixtures}. "
                f"Unknown in fixtures: {unknown_in_fixtures}."
            )

    def simulate_group_stage(self, future_predictions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Convert predicted group matches into points and final group standings.
        self.validate_future_matches(future_predictions)
        all_tables: List[pd.DataFrame] = []
        group_match_rows: List[Dict[str, object]] = []

        # Simulate every group separately.
        for group_letter, teams in self.official_groups.items():
            group_matches = future_predictions[
                future_predictions["home_team"].isin(teams) & future_predictions["away_team"].isin(teams)
            ].copy()
            if len(group_matches) != 6:
                raise ValueError(
                    f"Group {group_letter} should have 6 matches, found {len(group_matches)}. "
                    "Check team names and fixture data."
                )

            table = self._empty_group_table(group_letter, teams)
            head_to_head: Dict[frozenset, str] = {}

            # Award points based on the predicted 1X2 result.
            for _, match in group_matches.iterrows():
                home = match["home_team"]
                away = match["away_team"]
                outcome = int(match["pred_result"])

                if outcome == 1:
                    self._award_result(table, home, away, home_points=3, away_points=0)
                    winner = home
                elif outcome == 2:
                    self._award_result(table, home, away, home_points=0, away_points=3)
                    winner = away
                else:
                    self._award_result(table, home, away, home_points=1, away_points=1)
                    winner = "Draw"

                table.loc[home, "expected_points"] += 3 * match["prob_home_win"] + match["prob_draw"]
                table.loc[away, "expected_points"] += 3 * match["prob_away_win"] + match["prob_draw"]
                head_to_head[frozenset([home, away])] = winner

                group_match_rows.append(
                    {
                        "group": group_letter,
                        "date": match.get("date"),
                        "home_team": home,
                        "away_team": away,
                        "pred_label": match["pred_label"],
                        "prob_home_win": match["prob_home_win"],
                        "prob_draw": match["prob_draw"],
                        "prob_away_win": match["prob_away_win"],
                    }
                )

            ranked = self._rank_group_table(table, head_to_head).reset_index().rename(columns={"index": "team"})
            all_tables.append(ranked)

        return pd.concat(all_tables, ignore_index=True), pd.DataFrame(group_match_rows)

    def simulate_knockout(self, group_table: pd.DataFrame) -> pd.DataFrame:
        # Build the knockout bracket from group positions and simulate each round.
        rows: List[Dict[str, object]] = []
        winners: Dict[int, str] = {}
        losers: Dict[int, str] = {}

        # First resolve the Round-of-32 slots from group winners, runners-up and third-place teams.
        for slot in ROUND_OF_32_SLOTS:
            home = self._resolve_round32_slot(slot["home"], group_table)
            away = self._resolve_round32_slot(slot["away"], group_table)
            row = self._simulate_single_knockout_match("Round of 32", int(slot["match_number"]), home, away)
            rows.append(row)
            winners[int(slot["match_number"])] = str(row["winner"])
            losers[int(slot["match_number"])] = str(row["loser"])

        # Later rounds use winners from earlier match numbers.
        for round_name in ["Round of 16", "Quarter-final", "Semi-final"]:
            for match_def in KNOCKOUT_PATHWAY[round_name]:
                home = winners[int(match_def["home_source"])]
                away = winners[int(match_def["away_source"])]
                row = self._simulate_single_knockout_match(round_name, int(match_def["match_number"]), home, away)
                rows.append(row)
                winners[int(match_def["match_number"])] = str(row["winner"])
                losers[int(match_def["match_number"])] = str(row["loser"])

        third_def = KNOCKOUT_PATHWAY["Third-place play-off"][0]
        third_place_row = self._simulate_single_knockout_match(
            "Third-place play-off",
            int(third_def["match_number"]),
            losers[int(third_def["home_source_loser"])],
            losers[int(third_def["away_source_loser"])],
        )
        rows.append(third_place_row)
        winners[int(third_def["match_number"])] = str(third_place_row["winner"])
        losers[int(third_def["match_number"])] = str(third_place_row["loser"])

        final_def = KNOCKOUT_PATHWAY["Final"][0]
        final_row = self._simulate_single_knockout_match(
            "Final",
            int(final_def["match_number"]),
            winners[int(final_def["home_source"])],
            winners[int(final_def["away_source"])],
        )
        rows.append(final_row)
        winners[int(final_def["match_number"])] = str(final_row["winner"])
        losers[int(final_def["match_number"])] = str(final_row["loser"])

        return pd.DataFrame(rows)

    def predict_neutral_match(self, home_team: str, away_team: str) -> Dict[str, float | str | int]:
        # Predict one neutral-site match, mainly used for knockout fixtures.
        row = self.team_store.create_match(home_team, away_team, neutral=True)
        predicted = self.model.add_predictions(row).iloc[0]
        return {
            "home_team": home_team,
            "away_team": away_team,
            "pred_result": int(predicted["pred_result"]),
            "pred_label": predicted["pred_label"],
            "prob_home_win": float(predicted["prob_home_win"]),
            "prob_draw": float(predicted["prob_draw"]),
            "prob_away_win": float(predicted["prob_away_win"]),
        }

    def _simulate_single_knockout_match(self, round_name: str, match_number: int, home: str, away: str) -> Dict[str, object]:
        # Store one knockout match with its probabilities, winner and loser.
        prediction = self.predict_neutral_match(home, away)
        winner = self._choose_knockout_winner(prediction)
        loser = away if winner == home else home
        return {
            "round": round_name,
            "match_number": match_number,
            "home_team": home,
            "away_team": away,
            "prob_home_win_90": prediction["prob_home_win"],
            "prob_draw_90": prediction["prob_draw"],
            "prob_away_win_90": prediction["prob_away_win"],
            "winner": winner,
            "loser": loser,
            "advancement_rule": "90min winner; if draw, higher baseline win probability advances",
        }

    def _resolve_round32_slot(self, slot_spec: Tuple[str, object], group_table: pd.DataFrame) -> str:
        # Translate bracket slot labels such as WA, RB or eligible third place into team names.
        kind, group_spec = slot_spec
        if kind in {"W", "R"}:
            position = 1 if kind == "W" else 2
            group = str(group_spec)
            row = group_table[(group_table["group"] == group) & (group_table["group_position"] == position)]
            if row.empty:
                raise ValueError(f"Cannot resolve {kind}{group} from group table.")
            return str(row.iloc[0]["team"])
        if kind == "3":
            assignment = self._assign_third_place_slots(group_table)
            eligible_groups = tuple(group_spec)
            match_number = self._match_number_for_third_slot(eligible_groups)
            third_group = assignment[match_number]
            row = group_table[(group_table["group"] == third_group) & (group_table["group_position"] == 3)]
            if row.empty:
                raise ValueError(f"Cannot resolve third-place team from Group {third_group}.")
            return str(row.iloc[0]["team"])
        raise ValueError(f"Unknown slot type: {kind}")

    def _assign_third_place_slots(self, group_table: pd.DataFrame) -> Dict[int, str]:
        """
        Select the eight best third-place groups and place them into official
        third-place-eligible Round-of-32 slots using deterministic backtracking.

        This does not reseed teams by strength. It only fills the fixed FIFA slots
        whose labels allow that group letter.
        """
        best_thirds = self._best_third_place_groups(group_table)
        third_groups = tuple(best_thirds["group"].tolist())
        third_slots = [slot for slot in ROUND_OF_32_SLOTS if slot["away"][0] == "3"]
        candidates: Dict[int, Tuple[str, ...]] = {
            int(slot["match_number"]): tuple(g for g in slot["away"][1] if g in third_groups)
            for slot in third_slots
        }
        ordered_slots = sorted(candidates, key=lambda m: (len(candidates[m]), m))
        assignment = self._backtrack_third_assignment(ordered_slots, candidates, set(third_groups), {})
        if assignment is None:
            raise ValueError(
                "Could not assign best third-place teams to official eligible Round-of-32 slots. "
                f"Qualified third-place groups: {third_groups}."
            )
        return dict(sorted(assignment.items()))

    @staticmethod
    def _backtrack_third_assignment(
        remaining_slots: Sequence[int],
        candidates: Dict[int, Tuple[str, ...]],
        remaining_groups: set,
        current: Dict[int, str],
    ) -> Optional[Dict[int, str]]:
        # Stop once each third-place slot has one eligible group.
        if not remaining_slots:
            return current.copy()
        slot = remaining_slots[0]
        for group in candidates[slot]:
            if group not in remaining_groups:
                continue
            current[slot] = group
            remaining_groups.remove(group)
            solved = WorldCupSimulator._backtrack_third_assignment(
                remaining_slots[1:], candidates, remaining_groups, current
            )
            if solved is not None:
                return solved
            remaining_groups.add(group)
            current.pop(slot, None)
        return None

    @staticmethod
    def _match_number_for_third_slot(eligible_groups: Tuple[str, ...]) -> int:
        # First resolve the Round-of-32 slots from group winners, runners-up and third-place teams.
        for slot in ROUND_OF_32_SLOTS:
            if slot["away"][0] == "3" and tuple(slot["away"][1]) == eligible_groups:
                return int(slot["match_number"])
        raise ValueError(f"No third-place slot found for eligibility {eligible_groups}.")

    @staticmethod
    def _empty_group_table(group_letter: str, teams: List[str]) -> pd.DataFrame:
        # Start every team in a group with zero points and zero results.
        table = pd.DataFrame(index=teams)
        table["group"] = group_letter
        table["points"] = 0
        table["wins"] = 0
        table["draws"] = 0
        table["losses"] = 0
        table["expected_points"] = 0.0
        return table

    @staticmethod
    def _award_result(table: pd.DataFrame, home: str, away: str, home_points: int, away_points: int) -> None:
        # Update points and win/draw/loss counts after one simulated group match.
        table.loc[home, "points"] += home_points
        table.loc[away, "points"] += away_points
        if home_points == 3:
            table.loc[home, "wins"] += 1
            table.loc[away, "losses"] += 1
        elif away_points == 3:
            table.loc[away, "wins"] += 1
            table.loc[home, "losses"] += 1
        else:
            table.loc[home, "draws"] += 1
            table.loc[away, "draws"] += 1

    def _rank_group_table(self, table: pd.DataFrame, head_to_head: Dict[frozenset, str]) -> pd.DataFrame:
        # Rank teams by points, simple head-to-head bonus, FIFA rank and expected points.
        ranked = table.copy()
        ranked["fifa_rank"] = [self.team_store.team_table.loc[team, "rank"] for team in ranked.index]
        ranked["h2h_bonus"] = 0.0

        for _, tied in ranked.groupby("points"):
            tied_teams = list(tied.index)
            if len(tied_teams) == 2:
                winner = head_to_head.get(frozenset(tied_teams))
                if winner in tied_teams:
                    ranked.loc[winner, "h2h_bonus"] = 0.5

        ranked = ranked.sort_values(
            by=["points", "h2h_bonus", "fifa_rank", "expected_points"],
            ascending=[False, False, True, False],
        )
        ranked["group_position"] = range(1, len(ranked) + 1)
        return ranked

    @staticmethod
    def _best_third_place_groups(group_table: pd.DataFrame) -> pd.DataFrame:
        # Select the eight strongest third-place teams for the Round of 32.
        thirds = group_table[group_table["group_position"].eq(3)].copy()
        return thirds.sort_values(
            by=["points", "fifa_rank", "expected_points"],
            ascending=[False, True, False],
        ).head(8)

    @staticmethod
    def _choose_knockout_winner(prediction: Dict[str, float | str | int]) -> str:
        # Knockout games cannot end in a draw, so draws are resolved by higher win probability.
        if prediction["pred_result"] == 1:
            return str(prediction["home_team"])
        if prediction["pred_result"] == 2:
            return str(prediction["away_team"])
        return str(
            prediction["home_team"]
            if prediction["prob_home_win"] >= prediction["prob_away_win"]
            else prediction["away_team"]
        )


class WorldCupPredictorProject:
    """Runs the full modelling and simulation workflow."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.preprocessor = FootballDataPreprocessor(config)
        self.model = FootballOutcomeModel(config)
        self.df: Optional[pd.DataFrame] = None
        self.history: Optional[pd.DataFrame] = None
        self.future: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.evaluation: Optional[Dict[str, object]] = None

    def run(self) -> Dict[str, pd.DataFrame | Dict[str, object]]:
        # Full workflow: prepare data, train model, predict fixtures, simulate tournament and save outputs.
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = self.preprocessor.load()
        self.history = self.preprocessor.historical_data(self.df)
        self.future = self.preprocessor.future_data(self.df)

        # Train on older historical matches and test on newer historical matches.
        train_df, test_df = self.preprocessor.temporal_split(self.history)
        self.test_df = test_df
        self.model.fit(train_df)
        self.evaluation = self.model.evaluate(test_df)

        # Generate predictions for both the test set and future World Cup fixtures.
        test_predictions = self.model.add_predictions(test_df)
        future_predictions = self.model.add_predictions(self.future)

        # Use predicted group-stage results to create the final tournament bracket.
        team_store = TeamFeatureStore(self.df)
        simulator = WorldCupSimulator(self.model, team_store, official_groups=self.config.official_groups)
        group_table, group_matches = simulator.simulate_group_stage(future_predictions)
        knockout_bracket = simulator.simulate_knockout(group_table)
        third_place_assignments = self._third_place_assignment_table(simulator, group_table)

        outputs = {
            "evaluation": self.evaluation,
            "test_predictions": test_predictions,
            "future_predictions": future_predictions,
            "group_table": group_table,
            "group_matches": group_matches,
            "third_place_assignments": third_place_assignments,
            "knockout_bracket": knockout_bracket,
        }
        self._save_outputs(outputs)
        return outputs

    @staticmethod
    def _third_place_assignment_table(simulator: WorldCupSimulator, group_table: pd.DataFrame) -> pd.DataFrame:
        # Save which third-place group is assigned to each eligible Round-of-32 slot.
        assignment = simulator._assign_third_place_slots(group_table)
        rows = []
        for match_number, group in assignment.items():
            team = group_table[(group_table["group"] == group) & (group_table["group_position"] == 3)].iloc[0]["team"]
            rows.append({"match_number": match_number, "third_place_group": group, "team": team})
        return pd.DataFrame(rows).sort_values("match_number")

    def _save_outputs(self, outputs: Dict[str, pd.DataFrame | Dict[str, object]]) -> None:
        # Write all generated project tables to CSV files used by the app.
        for name, value in outputs.items():
            if isinstance(value, pd.DataFrame):
                value.to_csv(self.config.output_dir / f"{name}.csv", index=False)

        eval_rows = [
            {"metric": "test_matches", "value": outputs["evaluation"]["test_matches"]},
            {"metric": "accuracy", "value": outputs["evaluation"]["accuracy"]},
            {"metric": "log_loss", "value": outputs["evaluation"]["log_loss"]},
        ]
        pd.DataFrame(eval_rows).to_csv(self.config.output_dir / "evaluation_summary.csv", index=False)


def run_project(data_path: str | Path = "df_fin_v1.csv", output_dir: str | Path = "final_outputs_v2"):
    # Convenience function for running the project from a script or notebook.
    config = ProjectConfig(data_path=Path(data_path), output_dir=Path(output_dir))
    project = WorldCupPredictorProject(config)
    return project.run()
