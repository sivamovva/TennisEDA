# %%
# getting match summary data of all the atp matches from Jeff Sackmans github repo
import requests
import pandas as pd
from io import StringIO
# %%
# Replace with your GitHub repository details
user = 'JeffSackmann'
repo_match_summaries = 'tennis_atp'
repo_match_point_by_point = 'tennis_MatchChartingProject'
explore_repo = 'tennis_MatchChartingProject'
branch = 'master'

# List of CSV file paths in the repository that i am interested in - I am interested only in
# Fed-Novak matches, so starting in 2006 till 2020
csv_files_match_summary = [
    'atp_matches_2006.csv', 'atp_matches_2007.csv', 'atp_matches_2008.csv',
    'atp_matches_2009.csv', 'atp_matches_2010.csv', 'atp_matches_2011.csv',
    'atp_matches_2012.csv', 'atp_matches_2013.csv', 'atp_matches_2014.csv',
    'atp_matches_2015.csv', 'atp_matches_2016.csv', 'atp_matches_2017.csv',
    'atp_matches_2018.csv', 'atp_matches_2019.csv', 'atp_matches_2020.csv'

]


csv_base_url_match_summary = f'https://raw.githubusercontent.com/{user}/{repo_match_summaries}/{branch}/'
csv_base_url_match_point_by_point = f'https://raw.githubusercontent.com/{user}/{repo_match_point_by_point}/{branch}/'
csv_base_url_explore_data = f'https://raw.githubusercontent.com/{user}/{explore_repo}/{branch}/'


# %%
def get_player_match_subset(player1, player2):
    dfs = []

    for csv_file in csv_files_match_summary:
        url = csv_base_url_match_summary + csv_file
        response = requests.get(url)
        if response.status_code == 200:
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            dfs.append(df)
        else:
            print(f"Failed to fetch {csv_file}")

    # concatenate all data into 1 file and then just take the subset of the 50 odd matches
    # that fed and novak played against each other
    df_concat = pd.DataFrame()
    for i in range(len(dfs)):
        df_concat = pd.concat([df_concat, dfs[i]])

    # getting just the subset of matches where the specified players played against each other
    player_list = [player1, player2]
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

    df_concat_subset.to_csv(f'{player1}_{player2}_matches.csv', index=False)

    return df_concat_subset, top_winner

# %%


def get_win_rate_by_year(df_concat_subset, player1, player2, top_winner):
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
            f'{player1}_{player2}_win_rate_by_year.csv', index=False)

        return df_concat_subset_grouped_by_year


# %%
# at a later point, i want to pass these 2 players from the streamlit app user selection
df_concat_subset, top_winner = get_player_match_subset(
    'Roger Federer', 'Novak Djokovic')

# %%
# if more than 20 matches, get yearly summary
if len(df_concat_subset) >= 20:
    df_concat_subset_grouped_by_year = get_win_rate_by_year(
        df_concat_subset, 'Roger Federer', 'Novak Djokovic', top_winner)


# %%
