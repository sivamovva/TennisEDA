# %%
import time  # noqa
import streamlit as st  # noqa

# %%
# Use the full width of the page
st.set_page_config(
    page_title="Pro Tennis Analysis",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': 'https://www.example.com/bug',
        'About': 'This is an app for Pro Tennis Analysis.'
    }
)


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

with st.spinner('Loading player data. Will take a few seconds. Thanks for your patience!...'):
    try:
        from tennis_eda_dataprocessing import (
            get_player_match_subset_against_tour
            )   # noqa

    except Exception as e:
        st.error(
            f"An error occurred while getting the player match subset: {e}")
        st.stop()


# Dropdown menu with search option for selecting a player
user_selected_player = st.selectbox(
    'Select a Player from drop down list below', players, index=None, placeholder='Select Player...'
)

if user_selected_player:
    df_concat_subset = get_player_match_subset_against_tour(
        user_selected_player)

    with st.spinner('Player win percentage by year...'):
        try:
            from tennis_eda_dataprocessing import (
                get_win_rate_by_year
                )   # noqa
            df_concat_subset_grouped_by_year = get_win_rate_by_year(
                df_concat_subset, user_selected_player)
            from tennis_eda_plotting import plot_yearly_win_rate_trend  # noqa
            plot_yearly_win_rate_trend(
                df_concat_subset_grouped_by_year, user_selected_player)

        except Exception as e:
            st.error(
                f"An error occurred while getting the player win percentage by year: {e}")
            st.stop()

    with st.spinner('Getting detailed match data for learning patterns...'):
        try:
            from tennis_eda_dataprocessing import (
                load_data_selected_player_against_tour
                )   # noqa
            df_final_for_training_user_selected_p1p2_feature_aligned = load_data_selected_player_against_tour(
                user_selected_player, df_concat_subset)
            st.write(
                df_final_for_training_user_selected_p1p2_feature_aligned.head())

        except Exception as e:
            st.error(
                f"An error occurred while getting the match data for player: {e}")
            st.stop()
