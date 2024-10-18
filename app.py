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
    align_features_with_target,

)


# %%
# Use the full width of the page
st.set_page_config(layout="wide")

# %%
# Title of the app
st.title('Pro Tennis -  Head 2 Head Analysis')

st.write(
    "In this data app, we will analyze the head to head matches between two players and see if we can find any patterns or trends "
    "that can help us understand the dynamics of this matchup better"
)

# Dropdown menus for selecting players
player1 = st.selectbox('Select Player 1', [
                       'Roger Federer', 'Novak Djokovic', 'Rafael Nadal'])
player2 = st.selectbox('Select Player 2', [
                       'Roger Federer', 'Novak Djokovic', 'Rafael Nadal'])

# Ensure that Player 1 and Player 2 are not the same
if player1 == player2:
    st.error(
        "Player 1 and Player 2 cannot be the same. Please select different players.")
else:
    # Load the preprocessed data
    @st.cache_data  # this is a decorator that will cache the data for the session
    def load_data(player1, player2):
        # this takes the most time. figure out ow to cache this
        df_concat_subset, top_winner = get_player_match_subset(
            player1, player2)

        # Get the length of the dataframe
        num_matches = len(df_concat_subset)

        # Display the number of matches and the top winner
        st.write(
            f"Found {num_matches} matches between {player1} and {player2}. {top_winner} leads the H2H record.")
        # Get the H2H record of the top winner
        h2h_record = df_concat_subset['winner_name'].value_counts()
        top_winner_wins = h2h_record.get(top_winner, 0)
        top_loser_wins = h2h_record.get(
            player1 if top_winner == player2 else player2, 0)

        # Display the H2H record
        st.write(
            f"{top_winner} leads the H2H record {top_winner_wins}-{top_loser_wins}")
        # If more than 20 matches, get yearly summary
        if len(df_concat_subset) >= 20:
            df_concat_subset_grouped_by_year = get_win_rate_by_year(
                df_concat_subset, player1, player2, top_winner)

        # Get match charting master file
        df_match_charting_master = get_match_charting_master_data(
            player1, player2)

        df_charting_rally_stats = get_rally_stats_data(player1, player2)
        df_match_charting_overview_stats = get_match_charting_overview_stats_data()

        # looks like Jeff Sackman recently changed the format of the overview stats. player column went from 1/2 (serve order) to
        # actual player names, this happened on Oct 18th, 2024 - his first commit on this repo since Sept 2023...what are the odds
        # that he would make a change to the data right when i am working off of it :) update below is to handle this change

        # merge overview stats with master data to get Player1 and player 2 columns. This will reduce down the dataset to just
        # the matches between player 1 and player 2 of interest as the df_match_charting_master is just between the 2 players of interest
        df_match_charting_overview_stats = pd.merge(
            df_match_charting_overview_stats, df_match_charting_master[['match_id', 'Player 1', 'Player 2']], on='match_id')

        # Set the 'player_serve_order' column
        def set_serve_order(row):
            if row['player'] == row['Player 1']:
                return 1
            elif row['player'] == row['Player 2']:
                return 2
            else:
                return None

        # Set the serve order based on the player names
        df_match_charting_overview_stats['player_serve_order'] = df_match_charting_overview_stats.apply(
            set_serve_order, axis=1)

        # Process the match charting overview stats data
        df_match_charting_overview_stats_processed, df_match_charting_overview_stats_processed_pivot = process_match_charting_overview_stats(
            df_match_charting_overview_stats)

        # Merge some useful columns like winner_name, score etc from the atp_summary data with the match charting master data
        df_match_charting_master_merged_with_atp_match_summary = merge_atp_match_summary_and_match_charting_master_data(
            df_match_charting_master, df_concat_subset)

        # Merge the match charting master data with the processed match charting overview stats data with the feature columns
        df_merged_features_jumbled = merge_match_charting_feature_master_data(
            df_match_charting_master_merged_with_atp_match_summary, df_match_charting_overview_stats_processed_pivot)

        # Apply the function to create additional features
        df_merged_features_jumbled_additional_features = create_additional_features(
            df_merged_features_jumbled.copy())

        # Apply the function to align features with the target variable
        df_merged_features_aligned = align_features_with_target(
            df_merged_features_jumbled_additional_features.copy())

        # Get final dataframe just with feature and target columns for training
        df_final_for_training = df_merged_features_aligned[[
            'winner_name', 'loser_name', 'Player 1', 'Player 2', 'tight_match', 'winner_loser_rank_diff',
            'winner_aces_perc', 'winner_dfs_perc', 'winner_first_in_perc', 'winner_first_won_perc', 'winner_second_won_perc',
            'winner_bp_saved_perc', 'winner_return_pts_won_perc', 'winner_winners_unforced_perc', 'winner_winner_fh_perc',
            'winner_winners_bh_perc', 'winner_unforced_fh_perc', 'winner_unforced_bh_perc',
            'loser_aces_perc', 'loser_dfs_perc', 'loser_first_in_perc', 'loser_first_won_perc', 'loser_second_won_perc',
            'loser_bp_saved_perc', 'loser_return_pts_won_perc', 'loser_winners_unforced_perc', 'loser_winner_fh_perc',
            'loser_winners_bh_perc', 'loser_unforced_fh_perc', 'loser_unforced_bh_perc'
        ]]

        return df_concat_subset, df_final_for_training

    df_concat_subset, df_final_for_training = load_data(player1, player2)

    # Display the final dataframe
    st.write(df_final_for_training)

    # %%
