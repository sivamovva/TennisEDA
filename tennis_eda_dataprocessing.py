# %%
# getting match summary data of all the atp matches from Jeff Sackmans github repo
import requests
import pandas as pd
from io import StringIO
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import streamlit as st
from xgboost import XGBClassifier
# %%
# Replace with your GitHub repository details
data_source_user = 'JeffSackmann'
data_source_repo_match_summaries = 'tennis_atp'
data_source_repo_match_point_by_point = 'tennis_MatchChartingProject'
data_source_branch = 'master'

my_repo = 'TennisEDA'
my_username = 'sivamovva'
my_branch = 'main'


# List of CSV file paths in the repository that i am interested in
csv_files_match_summary = [
    'atp_matches_1990.csv', 'atp_matches_1991.csv', 'atp_matches_1992.csv',
    'atp_matches_1993.csv', 'atp_matches_1994.csv', 'atp_matches_1995.csv',
    'atp_matches_1996.csv', 'atp_matches_1997.csv', 'atp_matches_1998.csv',
    'atp_matches_1999.csv', 'atp_matches_2000.csv', 'atp_matches_2001.csv',
    'atp_matches_2002.csv', 'atp_matches_2003.csv', 'atp_matches_2004.csv',
    'atp_matches_2005.csv', 'atp_matches_2006.csv', 'atp_matches_2007.csv',
    'atp_matches_2008.csv', 'atp_matches_2009.csv', 'atp_matches_2010.csv', 'atp_matches_2011.csv',
    'atp_matches_2012.csv', 'atp_matches_2013.csv', 'atp_matches_2014.csv',
    'atp_matches_2015.csv', 'atp_matches_2016.csv', 'atp_matches_2017.csv',
    'atp_matches_2018.csv', 'atp_matches_2019.csv', 'atp_matches_2020.csv',
    'atp_matches_2021.csv', 'atp_matches_2022.csv', 'atp_matches_2023.csv',

]

# %%
# list of CSV files in the match charting project repo. There is a women's file as well here but am only including the men's files for now.
match_charting_master = ['charting-m-matches.csv']
match_charting_rally_stats = [
    'charting-m-stats-Rally.csv']
match_charting_return_outcomes = [
    'charting-m-stats-ReturnOutcomes.csv']
match_charting_overview_stats = ['charting-m-stats-Overview.csv']


# Base URL for the raw content of the CSV files
csv_base_url_match_summary = f'https://raw.githubusercontent.com/{data_source_user}/{data_source_repo_match_summaries}/{data_source_branch}/'
csv_base_url_match_charting = f'https://raw.githubusercontent.com/{data_source_user}/{data_source_repo_match_point_by_point}/{data_source_branch}/'

# url for concatenated master file parquet data. This is the file that i created by concatenating all the match summary files
atp_match_summary_masterfile = f'https://raw.githubusercontent.com/{my_username}/{my_repo}/{my_branch}/'


# %%
# Function to concatenate all match summary CSV files and save as a parquet file
def concatenate_and_save_match_summaries():
    dfs = []

    for csv_file in csv_files_match_summary:
        url = csv_base_url_match_summary + csv_file
        response = requests.get(url)
        if response.status_code == 200:
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data, on_bad_lines='skip')
            dfs.append(df)
        else:
            print(f"Failed to fetch {csv_file}")

    # Concatenate all data into one DataFrame
    df_concat = pd.concat(dfs, ignore_index=True)

    # Save the concatenated DataFrame as a parquet file
    df_concat.to_parquet('atp_matches_master.parquet', index=False)


# %%
# Call the function to concatenate and save the match summaries
concatenate_and_save_match_summaries()


# %%
# Read the concatenated master file from the repo
def get_player_match_subset(user_selected_player1, user_selected_player2):

    master_file_url = f'https://raw.githubusercontent.com/{my_username}/{my_repo}/{my_branch}/atp_matches_master.parquet'
    # Read the master parquet file
    df_concat = pd.read_parquet(master_file_url)

    # getting just the subset of matches where the specified players played against each other
    player_list = [user_selected_player1, user_selected_player2]
    df_concat_subset = df_concat.query(
        'winner_name in @player_list and loser_name in @player_list')

    # Determine the player with the higher win count
    winner_counts = df_concat_subset['winner_name'].value_counts()
    top_winner = winner_counts.idxmax()

    # Create a binary column for the top winner
    df_concat_subset[f'{top_winner}_wins'] = (
        df_concat_subset['winner_name'] == top_winner).astype(int)

    # create a year column from the tourney_date column
    df_concat_subset['year'] = df_concat_subset['tourney_date'].astype(
        'str').str[:4]

    # create unique match id column: concatenate year, tournament name, winner name and loser name (sorted alphabetically)
    df_concat_subset['custom_match_id'] = df_concat_subset['year'] + '_' + df_concat_subset['tourney_name'] + '_' + \
        df_concat_subset['round'] + '_' + df_concat_subset[['winner_name', 'loser_name']].apply(
            lambda x: '_'.join(sorted(x)), axis=1)

    df_concat_subset.to_csv(
        f'selected_players_atp_matches_summary.csv', index=False)

    return df_concat_subset, top_winner

# %%


def get_win_rate_by_year(df_concat_subset, user_selected_player1, user_selected_player2, top_winner):
    # Check if there are at least 20 matches
    if len(df_concat_subset) >= 20:
        # group by year and get the mean of the 'Novak_Wins' column and number of rows in each year
        df_concat_subset_grouped_by_year = df_concat_subset.groupby('year').agg({
            f'{top_winner}_wins': 'mean',
            'tourney_date': 'size'
        }).reset_index()
        df_concat_subset_grouped_by_year[f'{top_winner}_wins'] = df_concat_subset_grouped_by_year[f'{top_winner}_wins'] * 100

        # Rename the 'tourney_date' column to 'match_count' for clarity
        df_concat_subset_grouped_by_year.rename(
            columns={'tourney_date': 'Number_of_Matches_Played', f'{top_winner}_wins': f'{top_winner}_win_percentage'}, inplace=True)

        # reorder columns
        df_concat_subset_grouped_by_year = df_concat_subset_grouped_by_year[[
            'year', 'Number_of_Matches_Played', f'{top_winner}_win_percentage']]

        df_concat_subset_grouped_by_year.to_csv(
            f'selected_matchup_win_rate_by_year.csv', index=False)

        return df_concat_subset_grouped_by_year

# %%
# get the match charting master data for the specified players


def get_match_charting_master_data(user_selected_player1, user_selected_player2):
    dfs = []

    for csv_file in match_charting_master:
        url = csv_base_url_match_charting + csv_file
        response = requests.get(url)
        if response.status_code == 200:
            csv_data = StringIO(response.text)

            df = pd.read_csv(csv_data, on_bad_lines='skip', engine='python')
            dfs.append(df)
        else:
            print(f"Failed to fetch {csv_file}")

    df_concat = pd.DataFrame()
    for i in range(len(dfs)):
        df_concat = pd.concat([df_concat, dfs[i]])

    player_list = [user_selected_player1, user_selected_player2]
    df_concat_subset = df_concat.query(
        '`Player 1` in @player_list and `Player 2` in @player_list')

    df_concat_subset['year'] = df_concat_subset['Date'].astype(
        'str').str[:4]

    df_concat_subset['custom_match_id'] = df_concat_subset['year'] + '_' + df_concat_subset['Tournament'] + '_' + \
        df_concat_subset['Round'] + '_' + df_concat_subset[['Player 1', 'Player 2']].apply(
            lambda x: '_'.join(sorted(x)), axis=1)

    df_concat_subset.to_csv(
        f'selected_players_match_charting_master_data.csv', index=False)

    return df_concat_subset

# %%
# get the rally stats file and do some preprocessing on it


def get_rally_stats_data(user_selected_player1, user_selected_player2):
    dfs = []

    # this file has some trailing commas which is causing the read_csv function to fail, so
    # adding columns_to_read explicitly
    columns_to_read = ['match_id', 'server', 'returner', 'row', 'pts', 'pl1_won', 'pl1_winners',
                       'pl1_forced', 'pl1_unforced', 'pl2_won', 'pl2_winners',
                       'pl2_forced', 'pl2_unforced']

    for csv_file in match_charting_rally_stats:
        url = csv_base_url_match_charting + csv_file
        response = requests.get(url)
        if response.status_code == 200:
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data, usecols=columns_to_read)
            dfs.append(df)
        else:
            print(f"Failed to fetch {csv_file}")

    df_concat = pd.DataFrame()
    for i in range(len(dfs)):
        df_concat = pd.concat([df_concat, dfs[i]])

    # Keep only rows where 'server' or 'returner' column is either user_selected_player1 or user_selected_player2
    player_list = [user_selected_player1, user_selected_player2]
    df_concat = df_concat.query(
        'server in @player_list or returner in @player_list')

    # Keep only rows where 'row' column is equal to 'Total'
    df_concat = df_concat[df_concat['row'] == 'Total']

    # Create percentage columns
    df_concat['pl1_won_perc'] = df_concat['pl1_won'] * 100 / df_concat['pts']
    df_concat['pl2_won_perc'] = df_concat['pl2_won'] * 100 / df_concat['pts']
    df_concat['pl1_winners_perc'] = df_concat['pl1_winners'] * \
        100 / df_concat['pts']
    df_concat['pl2_winners_perc'] = df_concat['pl2_winners'] * \
        100 / df_concat['pts']
    df_concat['pl1_forced_perc'] = df_concat['pl1_forced'] * \
        100 / df_concat['pts']
    df_concat['pl2_forced_perc'] = df_concat['pl2_forced'] * \
        100 / df_concat['pts']
    df_concat['pl1_unforced_perc'] = df_concat['pl1_unforced'] * \
        100 / df_concat['pts']
    df_concat['pl2_unforced_perc'] = df_concat['pl2_unforced'] * \
        100 / df_concat['pts']

    # Round percentage columns to the nearest integer
    df_concat['pl1_won_perc'] = df_concat['pl1_won_perc'].round(0).astype(int)
    df_concat['pl2_won_perc'] = df_concat['pl2_won_perc'].round(0).astype(int)
    df_concat['pl1_winners_perc'] = df_concat['pl1_winners_perc'].round(
        0).astype(int)
    df_concat['pl2_winners_perc'] = df_concat['pl2_winners_perc'].round(
        0).astype(int)
    df_concat['pl1_forced_perc'] = df_concat['pl1_forced_perc'].round(
        0).astype(int)
    df_concat['pl2_forced_perc'] = df_concat['pl2_forced_perc'].round(
        0).astype(int)
    df_concat['pl1_unforced_perc'] = df_concat['pl1_unforced_perc'].round(
        0).astype(int)
    df_concat['pl2_unforced_perc'] = df_concat['pl2_unforced_perc'].round(
        0).astype(int)

    return df_concat


# %%
def get_match_charting_overview_stats_data():
    dfs = []

    for csv_file in match_charting_overview_stats:
        url = csv_base_url_match_charting + csv_file
        response = requests.get(url)
        if response.status_code == 200:
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data, on_bad_lines='skip')
            dfs.append(df)
        else:
            print(f"Failed to fetch {csv_file}")

    df_concat = pd.DataFrame()
    for i in range(len(dfs)):
        df_concat = pd.concat([df_concat, dfs[i]])

    # Keep only rows where 'set' column is 'Total'
    df_concat = df_concat[df_concat['set'] == 'Total']

    return df_concat

# %%
# process the overview stats dataframe


def process_match_charting_overview_stats(df_match_charting_overview_stats):
    df = df_match_charting_overview_stats.copy()

    # under player column, instead of 1 and 2, make it p1 and p2 for easier understanding of column names when the data is pivoted later
    df['player_serve_order'] = df['player_serve_order'].replace(
        {1: 'p1', 2: 'p2'})

    df['aces_perc'] = (df['aces'] * 100 / df['serve_pts'])
    df['dfs_perc'] = (df['dfs'] * 100 / df['serve_pts'])
    df['first_in_perc'] = (df['first_in'] * 100 /
                           df['serve_pts'])
    df['first_won_perc'] = (df['first_won'] * 100 /
                            df['first_in'])
    df['second_won_perc'] = (df['second_won'] * 100 /
                             df['second_in'])
    df['bp_saved_perc'] = (df['bp_saved'] * 100 /
                           df['bk_pts'])
    df['return_pts_won_perc'] = (
        df['return_pts_won'] * 100 / df['return_pts'])
    df['winners_unforced_perc'] = df['winners']*100 / \
        (df['unforced'] + df['winners'])
    df['winner_fh_perc'] = (df['winners_fh'] * 100 /
                            df['winners'])
    df['winners_bh_perc'] = (df['winners_bh'] * 100 /
                             df['winners'])
    df['unforced_fh_perc'] = (
        df['unforced_fh'] * 100 / df['unforced'])
    df['unforced_bh_perc'] = (
        df['unforced_bh'] * 100 / df['unforced'])

    # Replace infinite values with NaN and then fill NaN values with 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Round the specified columns to integers
    columns_to_round_int = [
        'first_in_perc', 'first_won_perc', 'second_won_perc', 'bp_saved_perc',
        'return_pts_won_perc', 'winner_fh_perc', 'winners_unforced_perc',
        'winners_bh_perc', 'unforced_fh_perc', 'unforced_bh_perc'
    ]

    df[columns_to_round_int] = df[columns_to_round_int].round(0).astype(int)

    # Round 'aces_perc' and 'dfs_perc' to 2 decimal places
    df['aces_perc'] = df['aces_perc'].round(2)
    df['dfs_perc'] = df['dfs_perc'].round(2)

    # drop duplicates before pivoting. Some times same match_id player combinations have multiple rows
    df = df.drop_duplicates(subset=['match_id', 'player_serve_order'])

    # pivot data such that there are separate columns for p1 (served first) and p2 (served second)
    df_pivot = df.pivot(index='match_id', columns='player_serve_order', values=['serve_pts', 'aces', 'dfs', 'first_in',
                                                                                'first_won', 'second_in', 'second_won', 'bk_pts', 'bp_saved',
                                                                                'return_pts', 'return_pts_won', 'winners', 'winners_fh', 'winners_bh',
                                                                                'unforced', 'unforced_fh', 'unforced_bh', 'aces_perc', 'dfs_perc',
                                                                                'first_in_perc', 'first_won_perc', 'second_won_perc', 'bp_saved_perc',
                                                                                'return_pts_won_perc', 'winners_unforced_perc', 'winner_fh_perc',
                                                                                'winners_bh_perc', 'unforced_fh_perc', 'unforced_bh_perc'])

    # Flatten the multi-index columns
    df_pivot.columns = [f'{col[1]}_{col[0]}' for col in df_pivot.columns]

    # reset index so match_id is a column
    df_pivot.reset_index(inplace=True)

    return df, df_pivot

# %%


# %%
# function to merge the match charting master data (that has only subset of the matches of the 2 players) with the processed
# match charting overview stats data (that has data from all matches)


def merge_match_charting_feature_master_data(df_match_charting_master, df_match_charting_overview_stats_processed_pivot):
    # merge the match charting master data with the processed match charting overview stats data
    df_merged = pd.merge(df_match_charting_master, df_match_charting_overview_stats_processed_pivot,
                         how='left', left_on='match_id', right_on='match_id')

    return df_merged

# %%
# function to merge the match charting master data with the atp match summary data that has the winner name and score


def merge_atp_match_summary_and_match_charting_master_data(df_match_charting_master, df_concat_subset):
    # merge the match charting master data with the atp match summary data
    df_merged = pd.merge(df_match_charting_master, df_concat_subset[['custom_match_id', 'winner_name', 'winner_age', 'winner_seed', 'winner_rank', 'loser_name', 'loser_age', 'loser_seed', 'loser_rank', 'score']],
                         how='left', left_on='custom_match_id', right_on='custom_match_id')

    return df_merged

# %%
# i want to create some new features with the df_merged_features_jumbled data set before i do feature alignment. The 3 new features are:
# 1) winner_rank_diff: difference in rank between the winner and loser
# 2) measure of how tight the match was: if total sets played is equal to best of sets, then it was a tight match (1). If not, then it was not a tight match (0)


def create_additional_features(df):

    # Calculate the rank difference between the winner and the loser
    df['winner_loser_rank_diff'] = df['winner_rank'].fillna(
        0) - df['loser_rank'].fillna(0)

    # Fill missing values in 'score' with an empty string before splitting the score to determine sets played
    df['score'] = df['score'].fillna('')
    df['sets_played'] = df['score'].fillna('').str.split().apply(len)

    # Determine if the match was tight based on the total sets played and best of sets
    df['tight_match'] = df.apply(
        lambda row: 1 if row['sets_played'] == row['Best of'] else 0, axis=1)

    return df
# %%


def align_features_with_winner_loser_and_create_target(df, user_selected_player1, user_selected_player2):

    # List of feature columns to swap between Player 1 and Player 2
    player_dependent_feature_columns = [
        'aces_perc', 'dfs_perc',
        'first_in_perc', 'first_won_perc', 'second_won_perc', 'bp_saved_perc',
        'return_pts_won_perc', 'winners_unforced_perc', 'winner_fh_perc', 'winners_bh_perc',
        'unforced_fh_perc', 'unforced_bh_perc'
    ]

    player_independent_feature_columns = [
        'winner_loser_rank_diff', 'tight_match']

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Check if Player 2 is the winner
        if row['winner_name'] == row['Player 2']:
            # Swap the feature values between Player 1 and Player 2
            for col in player_dependent_feature_columns:
                p1_col = f'p1_{col}'
                p2_col = f'p2_{col}'
                # Swap the values of Player 1 and Player 2 for the current feature
                df.at[index, p1_col], df.at[index,
                                            p2_col] = row[p2_col], row[p1_col]

    # Create target variable columns - 1 for each player so that model can learn feature importance from
    # both players perspective
    df[f'target_{user_selected_player1}_win'] = df['winner_name'].apply(
        lambda x: 1 if x == user_selected_player1 else 0)
    df[f'target_{user_selected_player2}_win'] = df['winner_name'].apply(
        lambda x: 1 if x == user_selected_player2 else 0)

    # Rename p1_ and p2_ prefixes to winner_ and loser_
    df.rename(
        columns={f'p1_{col}': f'winner_{col}' for col in player_dependent_feature_columns}, inplace=True)
    df.rename(
        columns={f'p2_{col}': f'loser_{col}' for col in player_dependent_feature_columns}, inplace=True)

    return df

# %%


def align_features_with_selected_players_and_create_target(df, user_selected_player1, user_selected_player2):

    # List of feature columns to swap between Player 1 and Player 2
    player_dependent_feature_columns = [
        'aces_perc', 'dfs_perc',
        'first_in_perc', 'first_won_perc', 'second_won_perc', 'bp_saved_perc',
        'return_pts_won_perc', 'winners_unforced_perc', 'winner_fh_perc', 'winners_bh_perc',
        'unforced_fh_perc', 'unforced_bh_perc'
    ]

    player_independent_feature_columns = [
        'winner_loser_rank_diff', 'tight_match']

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Check if Player 2 is the user_selected_player1
        if row['Player 2'] == user_selected_player1:
            # Swap the feature values between Player 1 and Player 2
            for col in player_dependent_feature_columns:
                p1_col = f'p1_{col}'
                p2_col = f'p2_{col}'
                # Swap the values of Player 1 and Player 2 for the current feature
                df.at[index, p1_col], df.at[index,
                                            p2_col] = row[p2_col], row[p1_col]

    # Create target variable columns - 1 for each player so that model can learn feature importance from
    # both players perspective
    df[f'target_{user_selected_player1}_win'] = df['winner_name'].apply(
        lambda x: 1 if x == user_selected_player1 else 0)
    df[f'target_{user_selected_player2}_win'] = df['winner_name'].apply(
        lambda x: 1 if x == user_selected_player2 else 0)

    # Rename p1_ and p2_ prefixes to user_selected_player1_ and user_selected_player2_
    df.rename(
        columns={f'p1_{col}': f'{user_selected_player1}_{col}' for col in player_dependent_feature_columns}, inplace=True)
    df.rename(
        columns={f'p2_{col}': f'{user_selected_player2}_{col}' for col in player_dependent_feature_columns}, inplace=True)

    return df


# %%


def display_metrics_in_streamlit(precision, recall, f1, accuracy, target_column):
    st.write(f"Logistic Regression Metrics for {target_column}:")
    st.write(f"Precision: {precision}")
    st.write(f"Recall: {recall}")
    st.write(f"F1 Score: {f1}")
    st.write(f"Accuracy: {accuracy}")


def fit_logistic_regression(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    display_metrics_in_streamlit(
        precision, recall, f1, accuracy, target_column)

    return model


# %%


def get_feature_importance_random_forest(X, y):

    # specifying the correct feature (X) and target (y) columns is done in app.py before the function call.

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    feature_importances = model.feature_importances_
    feature_names = X.columns

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    return model, feature_importance_df

# %%


def get_feature_importance_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    feature_importances = model.feature_importances_
    feature_names = X.columns

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    return model, feature_importance_df
