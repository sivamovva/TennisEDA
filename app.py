# %%
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# %%
# Use the full width of the page
st.set_page_config(layout="wide")

# %%
# title of the app
st.title('Novak Djokovic vs Roger Federer Head 2 Head Analysis')

st.write("Roger Federer and Novak Djokovic are two of the greatest tennis players of all time. They played each other 50 times, with Novak winning 27 times and Roger winning 23 times. In this data app, we will analyze the head to head matches between these two players and see if we can find any patterns or trends that can help us understand the dynamics of this matchup better")

st.write('Below is a table with results of all the matches they played. To the right is a line plot showing Novak Djokovic win percentage by year')
st.sidebar.title("Data Source Acknowledgement")


# Define the content for the 'About' page

st.sidebar.write("""
Thanks to Jeff Sackman/Tennis Abstract for inspiration and for the data. Using this under the Creative Commons Attribution - Non Commercial License. 
What Jeff and a team of volunteer contributors have done is amazing - especially the data charting exercise, painstakingly recording every point and attributing useful 
features like shot depth, shot direction etc to every shot. Jeff's work can be found at https://github.com/JeffSackmann. I picked a couple of interesting data sets
from his repo to make sense of the Federer-Djokovic head to head matchup. 
                 
Federer is my favourite tennis player and Djokovic is my wife's favourite player. So anytime they played, it was a house divided :) 
                 
""")

# Load the preprocessed data


@st.cache_data  # this is a decorator that will cache the data for the session
def load_data():
    avg_novak_wins = pd.read_csv('novak_win_rate_by_year.csv')
    # Convert 'year' column to numeric
    avg_novak_wins['year'] = pd.to_numeric(
        avg_novak_wins['year'], errors='coerce')

    all_matches_for_display = pd.read_csv('roger_novak_matches.csv')

    all_matches_for_display['tourney_date'] = pd.to_datetime(
        all_matches_for_display['tourney_date'], format='%Y%m%d').dt.strftime('%Y/%m/%d')
    # just subset of columns to display on webapp
    all_matches_for_display = all_matches_for_display[[
        'year', 'tourney_name', 'tourney_date', 'winner_name', 'loser_name', 'score', 'round', 'surface']]

    return avg_novak_wins, all_matches_for_display


# %%
avg_novak_wins, all_matches_for_display = load_data()

# Set the format for the dataframe display

format_dict = {
    'year': '{:.0f}'  # No comma, no decimal places

}


# Create an interactive line plot using Plotly
fig = px.line(avg_novak_wins, x='year', y='Novak_win_percentage',
              title='Novak Djokovic Win Percentage by Year',
              labels={'year': 'Year',
                      'Novak_win_percentage': 'Novak Win Percentage'},
              hover_data={'year': False, 'Novak_win_percentage': True,
                          'Number_of_Matches_Played': True},
              markers=True)

# Add reference lines
fig.add_hline(y=50, line_dash="dash", line_color="red",
              annotation_text="50% Win Rate")
fig.add_vline(x=2011, line_dash="dash", line_color="green",
              annotation_text="Year 2011- Novak's Dominance Begins")

# Update layout for better appearance
fig.update_layout(yaxis_range=[-10, 110], xaxis=dict(tickmode='linear'))


# Create two columns with a larger ratio for the first column
col1, col2 = st.columns([2, 2])

# Display the updated dataframe in the first column
with col1:
    st.dataframe(all_matches_for_display.style.format(format_dict), height=525)

# Display the plot in the second column
with col2:
    st.plotly_chart(fig, use_container_width=True)
