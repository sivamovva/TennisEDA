# %%
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from tennis_eda_dataprocessing import (
    get_player_match_subset,
    get_win_rate_by_year,
    get_match_charting_master_data,
    get_rally_stats_data,
    get_match_charting_overview_stats_data,
    process_match_charting_overview_stats,
    merge_atp_match_summary_and_match_charting_master_data,
    merge_match_charting_feature_master_data,
    create_additional_features,
    align_features_with_winner_loser_and_create_target,
    align_features_with_selected_players_and_create_target,
    get_feature_importance_random_forest,
    get_feature_importance_xgboost,
    get_player_match_subset_against_tour


)


# %%
# Use the full width of the page
st.set_page_config(layout="wide")

# %%
# Title of the app
st.title('Pro Tennis -  How do the most successful players win?')

st.write(
    "In this data app, we will analyze the players who have played at least 100 ATP matches since 1990 (a tiny fraction of the total number of players who played professional tennis) and see the factors that determined whether they won or lost a match")

try:
    # Read the player_subset.parquet file
    df_player_subset = pd.read_parquet('player_subset.parquet')
except FileNotFoundError:
    st.error("The file 'player_subset.parquet' was not found. Please ensure the file is in the correct location.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while reading the file: {e}")
    st.stop()


# Get the list of players from the first column of the dataframe
players = df_player_subset.iloc[:, 0].unique().tolist()


# Add a text input for player search
player_search = st.text_input(
    'Type in a player name you are interested in here to get a filtered down list of options below', '')

# Filter players based on search input
filtered_players = [
    player for player in players if player_search.lower() in player.lower()]

# Dropdown menu for selecting a player with filtered options
user_selected_player = st.selectbox(
    'Or Select Player directly here (warning: Long list !)', filtered_players, index=0 if filtered_players else -1)

df_concat_subset = get_player_match_subset_against_tour(user_selected_player)
