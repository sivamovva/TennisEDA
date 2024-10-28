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

# Dropdown menu for selecting tour type
tours = ['ATP', 'WTA']
user_selected_tour = st.selectbox(
    'Select a tour type from the dropdown below',
    tours,
    index=None,
    placeholder='Select ATP tour (Men) or WTA tour (Women)'
)


# once tour_type is selected, we can filter the player_subset.parquet file based on the tour_type
if user_selected_tour:

    user_selected_tour = user_selected_tour.lower()

    with st.spinner('Getting player list who have played at least 150 matches...'):
        try:
            global pd, df_player_subset  # noqa
            import pandas as pd  # noqa

            # Read the player_subset.parquet file
            df_player_subset = pd.read_parquet(
                f'{user_selected_tour}_player_subset.parquet')

        except FileNotFoundError:
            st.error(
                "The file 'player_subset.parquet' was not found. Please ensure the file is in the correct location.")
            st.stop()
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")
            st.stop()

    # Get the list of players from the first column of the dataframe
    players = df_player_subset.iloc[:, 0].unique().tolist()

    # Dropdown menu with search option for selecting a player
    user_selected_player = st.selectbox(
        'Select a Player from drop down list below', players, index=None, placeholder='Select Player...'
    )

    if user_selected_player:

        with st.spinner('Loading player data. Will take ~ a minute the very first time. Thanks for your patience!...'):
            try:

                from tennis_eda_dataprocessing import (
                    get_player_match_subset_against_tour
                    )   # noqa

                df_concat_subset = get_player_match_subset_against_tour(
                    user_selected_player, user_selected_tour)

            except Exception as e:
                st.error(
                    f"An error occurred while getting the player match subset: {e}")
                st.stop()

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
                    user_selected_player, user_selected_tour, df_concat_subset)

                if df_final_for_training_user_selected_p1p2_feature_aligned.shape[0] > 100:
                    st.write(
                        f'Found {df_final_for_training_user_selected_p1p2_feature_aligned.shape[0]} matches with detailed stats for {user_selected_player}. This is a good number of matches to learn their play patterns against rest of the tour')
                else:
                    st.write(
                        f'Found just {df_final_for_training_user_selected_p1p2_feature_aligned.shape[0]} matches with detailed stats for {user_selected_player}. May not be enough to learn their patterns against rest of the tour')

            except Exception as e:
                st.error(
                    f"An error occurred while getting the match data for player: {e}")
                st.stop()
                # Create two tabs
                tab1, tab2 = st.tabs(
                    ["Factors important for user_selected_player win", "Factors important for opponent win"])

        tab1, tab2 = st.tabs(
            [f'Factors specific to {user_selected_player} win',
             'Factors specific to opponent win'
             ])
        with st.spinner('Training on data to learn patterns...'):
            with tab1:

                st.header(f"Factors important for {user_selected_player} win")
                try:
                    X1 = df_final_for_training_user_selected_p1p2_feature_aligned.drop(
                        columns=[f'target_{user_selected_player}_win', 'target_opponent_win',
                                 'match_id', 'custom_match_id', 'year', 'winner_name', 'opponent_aces_perc', 'opponent_dfs_perc', 'opponent_first_in_perc',
                                 'opponent_first_won_perc', 'opponent_second_won_perc', 'opponent_bp_saved_perc',
                                 'opponent_return_pts_won_perc', 'opponent_winners_unforced_perc', 'opponent_winner_fh_perc',
                                 'opponent_winners_bh_perc', 'opponent_unforced_fh_perc', 'opponent_unforced_bh_perc'
                                 ])

                    y1 = df_final_for_training_user_selected_p1p2_feature_aligned[
                        f'target_{user_selected_player}_win']

                    from tennis_eda_dataprocessing import get_feature_importance_xgboost  # noqa

                    model_xgb1, feature_importance_df_xgb1 = get_feature_importance_xgboost(
                        X1, y1)

                    top_features_xgb1 = feature_importance_df_xgb1.head(5)
                    top_features_xgb1 = top_features_xgb1.sort_values(
                        by='Importance', ascending=False)
                    top_features_xgb1 = top_features_xgb1.iloc[::-1]
                    import plotly.express as px  # noqa
                    fig_xgb1 = px.bar(top_features_xgb1, x='Importance', y='Feature',
                                      orientation='h', title='Top 5 Feature Importances (XGBoost)')
                    st.plotly_chart(fig_xgb1, key='fig_xgb1')
                except Exception as e:
                    st.error(
                        f"An error occurred while getting the factors for win: {e}")

            with tab2:
                st.header(f"Factors important for opponent win")
                try:
                    st.header(
                        f'XGBoost Feature Importance specific to opponent win')
                    X2 = df_final_for_training_user_selected_p1p2_feature_aligned.drop(
                        columns=[f'target_{user_selected_player}_win', 'target_opponent_win',
                                 'match_id', 'custom_match_id', 'year', 'winner_name',
                                 f'{user_selected_player}_aces_perc', f'{user_selected_player}_dfs_perc', f'{user_selected_player}_first_in_perc',
                                 f'{user_selected_player}_first_won_perc', f'{user_selected_player}_second_won_perc', f'{user_selected_player}_bp_saved_perc',
                                 f'{user_selected_player}_return_pts_won_perc', f'{user_selected_player}_winners_unforced_perc', f'{user_selected_player}_winner_fh_perc',
                                 f'{user_selected_player}_winners_bh_perc', f'{user_selected_player}_unforced_fh_perc', f'{user_selected_player}_unforced_bh_perc'
                                 ])

                    y2 = df_final_for_training_user_selected_p1p2_feature_aligned[
                        'target_opponent_win']

                    model_xgb2, feature_importance_df_xgb2 = get_feature_importance_xgboost(
                        X2, y2)

                    top_features_xgb2 = feature_importance_df_xgb2.head(5)
                    top_features_xgb2 = top_features_xgb2.sort_values(
                        by='Importance', ascending=False)
                    top_features_xgb2 = top_features_xgb2.iloc[::-1]
                    fig_xgb2 = px.bar(top_features_xgb2, x='Importance', y='Feature',
                                      orientation='h', title='Top 5 Feature Importances (XGBoost)')
                    st.plotly_chart(fig_xgb2, key='fig_xgb2')

                except Exception as e:
                    st.error(
                        f"An error occurred while getting the factors for opponent win: {e}")
        st.stop()
