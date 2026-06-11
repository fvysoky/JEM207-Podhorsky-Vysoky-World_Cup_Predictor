from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Main project paths used by the Streamlit app.
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "df_fin_v1.csv"
OUTPUT_DIR = ROOT / "final_outputs_v2"

# Output probability columns for the three possible match outcomes.
OUTCOME_COLUMNS = {
    "Home win": "prob_home_win",
    "Draw": "prob_draw",
    "Away win": "prob_away_win",
}

# Order used when displaying the knockout rounds.
ROUND_ORDER = [
    "Round of 32",
    "Round of 16",
    "Quarter-final",
    "Semi-final",
    "Third-place play-off",
    "Final",
]


def setup_page() -> None:
    # Basic Streamlit page settings.
    st.set_page_config(
        page_title="2026 World Cup Predictor",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )


@st.cache_data(show_spinner=False)
def read_csv(name: str) -> pd.DataFrame:
    # Load generated CSV outputs; return an empty table if the file is missing.
    path = OUTPUT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_pct(value: float | int | None) -> str:
    # Format model probabilities for display.
    if pd.isna(value):
        return "—"
    return f"{float(value) * 100:.1f}%"


def money(value: float | int | None) -> str:
    # Format market values in a readable form.
    if pd.isna(value):
        return "—"
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"€{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"€{value / 1_000_000:.1f}M"
    return f"€{value:,.0f}"


def render_match_probabilities(match: pd.Series) -> None:
    # Show the selected match, model pick and the 1X2 probabilities.
    home = str(match["home_team"])
    away = str(match["away_team"])
    label = str(match.get("pred_label", match.get("winner", "")))

    st.subheader(f"{home} vs {away}")
    if label:
        st.write(f"Model pick: **{label}**")

    probs = pd.DataFrame(
        {
            "Outcome": [f"{home} win", "Draw", f"{away} win"],
            "Probability": [
                safe_pct(match.get("prob_home_win", match.get("prob_home_win_90"))),
                safe_pct(match.get("prob_draw", match.get("prob_draw_90"))),
                safe_pct(match.get("prob_away_win", match.get("prob_away_win_90"))),
            ],
        }
    )
    st.dataframe(probs, width="stretch", hide_index=True)


def run_model_from_app() -> None:
    # Re-run the modelling pipeline from the app and refresh generated outputs.
    try:
        from predictor_pipeline_final_v2 import ProjectConfig, WorldCupPredictorProject
    except ImportError as exc:
        st.error("Model code or required packages are not available.")
        st.code(f"{sys.executable} -m pip install -r requirements.txt", language="bash")
        st.exception(exc)
        return

    with st.spinner("Training model and regenerating outputs..."):
        config = ProjectConfig(data_path=DATA_PATH, output_dir=OUTPUT_DIR)
        WorldCupPredictorProject(config).run()
    st.cache_data.clear()
    st.success("Outputs regenerated.")


def main() -> None:
    setup_page()

    # Sidebar navigation between the main parts of the project.
    st.sidebar.title("WC Predictor")
    page = st.sidebar.radio(
        "Page",
        ["Overview", "Match Predictor", "Groups", "Knockout", "Bet Builder", "Run / Setup"],
    )

    # Load all output tables created by the modelling pipeline.
    evaluation = read_csv("evaluation_summary.csv")
    future = read_csv("future_predictions.csv")
    groups = read_csv("group_table.csv")
    group_matches = read_csv("group_matches.csv")
    bracket = read_csv("knockout_bracket.csv")
    third_places = read_csv("third_place_assignments.csv")

    # Warn the user when generated outputs are not available yet.
    missing = [
        name
        for name, df in {
            "evaluation_summary.csv": evaluation,
            "future_predictions.csv": future,
            "group_table.csv": groups,
            "group_matches.csv": group_matches,
            "knockout_bracket.csv": bracket,
        }.items()
        if df.empty
    ]
    if missing and page != "Run / Setup":
        st.warning("Missing output files. Regenerate outputs in Run / Setup.")
        st.code("\n".join(missing))

    if page == "Overview":
        # Summary page with model metrics and the predicted champion.
        st.title("2026 World Cup Predictor")
        st.write(
            "The project trains a calibrated 1X2 match-outcome model, predicts group-stage fixtures, "
            "simulates the group tables and then follows the fixed knockout bracket."
        )

        # Convert the evaluation table into values used in the metric cards.
        metric_map = dict(zip(evaluation.get("metric", []), evaluation.get("value", []))) if not evaluation.empty else {}
        champion = "—"
        if not bracket.empty and "Final" in set(bracket["round"]):
            champion = str(bracket.loc[bracket["round"] == "Final", "winner"].iloc[0])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Test accuracy", f"{float(metric_map.get('accuracy', 0)):.1%}" if metric_map else "—")
        col2.metric("Log loss", f"{float(metric_map.get('log_loss', 0)):.3f}" if "log_loss" in metric_map else "—")
        col3.metric("Future matches", str(len(future)) if not future.empty else "—")
        col4.metric("Predicted champion", champion)

        if not bracket.empty:
            st.subheader("Final")
            final = bracket[bracket["round"] == "Final"].iloc[0]
            render_match_probabilities(final)

        if not third_places.empty:
            st.subheader("Best third-place assignments")
            st.dataframe(third_places, width="stretch", hide_index=True)

    elif page == "Match Predictor":
        # Inspect one fixture and the exact features used by the model.
        st.title("Match Predictor")
        st.write("Select a group-stage fixture to show the model probabilities.")
        if not future.empty:
            labels = future.apply(lambda r: f"{r['date']} · {r['home_team']} vs {r['away_team']}", axis=1).tolist()
            selected = st.selectbox("Fixture", labels)
            match = future.iloc[labels.index(selected)]
            render_match_probabilities(match)

            # Show the main input differences behind the selected prediction.
            features = pd.DataFrame(
                {
                    "Feature": ["rank_difference", "value_difference", "form_difference", "age_difference", "home_advantage"],
                    "Value": [
                        match.get("rank_difference"),
                        money(match.get("value_difference")),
                        match.get("form_difference"),
                        match.get("age_difference"),
                        match.get("home_advantage"),
                    ],
                }
            )
            st.subheader("Input features")
            st.dataframe(features, width="stretch", hide_index=True)

    elif page == "Groups":
        # Display simulated group standings and predicted group fixtures.
        st.title("Group Tables")
        st.write("Top two teams qualify directly. The third-place teams are ranked for the remaining slots.")
        if not groups.empty:
            selected_group = st.selectbox("Group", sorted(groups["group"].unique()))
            st.subheader(f"Group {selected_group}")
            st.dataframe(
                groups[groups["group"] == selected_group].sort_values("group_position"),
                width="stretch",
                hide_index=True,
            )
            st.subheader("Fixtures")
            st.dataframe(
                group_matches[group_matches["group"] == selected_group],
                width="stretch",
                hide_index=True,
            )

    elif page == "Knockout":
        # Display each knockout round in the official bracket order.
        st.title("Knockout Bracket")
        st.write("The knockout stage uses fixed match-number slots.")
        if not bracket.empty:
            rounds = [round_name for round_name in ROUND_ORDER if round_name in set(bracket["round"])]
            tabs = st.tabs(rounds)
            for tab, round_name in zip(tabs, rounds):
                with tab:
                    st.dataframe(
                        bracket[bracket["round"] == round_name].sort_values("match_number"),
                        width="stretch",
                        hide_index=True,
                    )

    elif page == "Bet Builder":
        # Compare bookmaker decimal odds with the model probability.
        st.title("Bet Builder")
        st.write("Enter decimal odds and compare them with the model probability.")
        if not future.empty:
            labels = future.apply(lambda r: f"{r['date']} · {r['home_team']} vs {r['away_team']}", axis=1).tolist()
            selected = st.selectbox("Fixture", labels, key="bet_fixture")
            match = future.iloc[labels.index(selected)]
            outcome = st.radio("Prediction", [f"{match['home_team']} win", "Draw", f"{match['away_team']} win"], horizontal=True)
            odds = st.number_input("Decimal odds", min_value=1.01, value=2.00, step=0.01)

            # Select the probability that corresponds to the chosen betting outcome.
            if outcome == "Draw":
                p = float(match["prob_draw"])
            elif outcome.startswith(str(match["home_team"])):
                p = float(match["prob_home_win"])
            else:
                p = float(match["prob_away_win"])

            # Fair odds and expected value are derived from the selected model probability.
            fair_odds = 1 / p if p > 0 else float("inf")
            expected_value = p * odds - 1

            col1, col2, col3 = st.columns(3)
            col1.metric("Model probability", f"{p:.1%}")
            col2.metric("Fair odds", f"{fair_odds:.2f}")
            col3.metric("Expected value", f"{expected_value:+.2%}")
            render_match_probabilities(match)

    elif page == "Run / Setup":
        # Basic commands and button for regenerating all project outputs.
        st.title("Run / Setup")
        st.write("Use this page to install packages or regenerate output files.")
        st.code(f"{sys.executable} -m pip install -r requirements.txt", language="bash")
        st.code("streamlit run app.py", language="bash")
        if st.button("Regenerate outputs", type="primary"):
            run_model_from_app()

    st.caption(
        "Model limitation: the model predicts 1X2 match outcomes. Goal-difference tiebreakers are approximated."
    )


if __name__ == "__main__":
    main()
