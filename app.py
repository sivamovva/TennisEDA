# %%
import time  # noqa
import streamlit as st  # noqa

# %%
# Use the full width of the page
st.set_page_config(layout="wide")


# %%
# Title of the app
st.title('Pro Tennis -  How do the most successful players win?')


# Text to display with typing effect
intro_text = "In this data app, we will analyze the players who have played at least 150 ATP matches since 1990 (about 400 out of the >3000 professionals) and see the factors that determined whether they won or lost a match"
st.write(intro_text)


with st.spinner('Getting player list who have played at least 150 matches...'):
    try:
        global pd, get_player_match_subset_against_tour, df_player_subset  # noqa
        import pandas as pd  # noqa

        # Read the player_subset.parquet file
        df_player_subset = pd.read_parquet('player_subset.parquet')

    except FileNotFoundError:
        st.error(
            "The file 'player_subset.parquet' was not found. Please ensure the file is in the correct location.")
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

with st.spinner('Getting player match subset...'):
    try:
        from tennis_eda_dataprocessing import (
            get_player_match_subset_against_tour
            )   # noqa
        df_concat_subset = get_player_match_subset_against_tour(
            user_selected_player)

    except Exception as e:
        st.error(
            f"An error occurred while getting the player match subset: {e}")
        st.stop()
