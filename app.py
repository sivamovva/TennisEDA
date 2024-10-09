# %%
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# %%
# title of the app
st.title('Novak Djokovic vs Roger Federer Head 2 Head Analysis')

st.write('Roger Federer and Novak Djokovic are inarguably two of the greatest tennis players of all time. They played each other 50 times, with Novak winning 27 times and Roger winning 23 times. In this app, we will analyze the head to head matches between these two players and see if we can find any patterns or trends that can help us understand the dynamics of this matchup better.')


# Load the preprocessed data


@st.cache_data  # this is a decorator that will cache the data for the session
def load_data():
    avg_novak_wins = pd.read_csv('novak_win_rate_by_year.csv')
    # Convert 'year' column to numeric
    avg_novak_wins['year'] = pd.to_numeric(
        avg_novak_wins['year'], errors='coerce')
    return avg_novak_wins


# %%
avg_novak_wins = load_data()
st.write(avg_novak_wins)
