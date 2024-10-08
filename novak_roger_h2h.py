import matplotlib.pyplot as plt
import streamlit as st
import requests
import pandas as pd
from io import StringIO


# Replace with your GitHub repository details
user = 'JeffSackmann'
repo = 'tennis_atp'
branch = 'master'


# List of CSV file paths in the repository that i am interested in - I am interested only in
# Fed-Novak matches, so starting in 2006 till 2020
csv_files = [
    'atp_matches_2006.csv', 'atp_matches_2007.csv', 'atp_matches_2008.csv',
    'atp_matches_2009.csv', 'atp_matches_2010.csv', 'atp_matches_2011.csv',
    'atp_matches_2012.csv', 'atp_matches_2013.csv', 'atp_matches_2014.csv',
    'atp_matches_2015.csv', 'atp_matches_2016.csv', 'atp_matches_2017.csv',
    'atp_matches_2018.csv', 'atp_matches_2019.csv', 'atp_matches_2020.csv'

]

csv_base_url = f'https://raw.githubusercontent.com/{user}/{repo}/{branch}/'

dfs = []

for csv_file in csv_files:
    url = csv_base_url + csv_file
    response = requests.get(url)
    if response.status_code == 200:
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        dfs.append(df)
    else:
        print(f"Failed to fetch {csv_file}")

# concatenate all data into 1 file and then just take the subset of the 50 odd matches
# that fed and novak played against eachother
df_concat = pd.DataFrame()
for i in range(len(dfs)):
    df_concat = pd.concat([df_concat, dfs[i]])


# getting just the subset of matches where Fed played Novak
player_list = ['Roger Federer', 'Novak Djokovic']
df_concat_subset = df_concat.query(
    'winner_name in @player_list and loser_name in @player_list')

# create a binary column 1 or 0 for 'Novak Wins'. We can use this to do the investigation
# of what features helped gradually gain an advantage in the matchup
df_concat_subset['Novak_Wins'] = (
    df_concat_subset['winner_name'] == 'Novak Djokovic').astype(int)

# create a year column from the tourney_date column
df_concat_subset['year'] = df_concat_subset['tourney_date'].astype(
    'str').str[:4]


avg_novak_wins = df_concat_subset.groupby(
    'year')['Novak_Wins'].mean().reset_index()
avg_novak_wins['year'] = pd.to_numeric(avg_novak_wins['year'], errors='coerce')
ax = avg_novak_wins.plot(x='year', y='Novak_Wins', kind='line',
                         marker='o', title='Novak Win Rate Over Federer by Year')
ax.set_ylabel('% of Matches Won by Novak')
ax.axhline(y=0.5, color='r', linestyle='--', label='50% Win Rate')

ax.set_xticks(avg_novak_wins['year'])
ax.tick_params(axis='x', rotation=45)
ax.axvline(x=2011, color='g', linestyle='--',
           label='2011: Start of Novak\'s Dominance')
ax.legend()
plt.show()

st.write('Hello World, my first streamlit app')
