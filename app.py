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


# %%
# Use the full width of the page
st.set_page_config(layout="wide")

# %%
# Title of the app
st.title('Novak Djokovic vs Roger Federer Head 2 Head Analysis')

st.write(
    "Roger Federer and Novak Djokovic are two of the greatest tennis players of all time. "
    "They played each other 50 times, with Novak winning 27 times and Roger winning 23 times. "
    "In this data app, we will analyze the head to head matches between these two players and see if we can find any patterns or trends "
    "that can help us understand the dynamics of this matchup better"
)

st.write(
    'Below is a table with results of all the matches they played. To the right is a line plot showing Novak Djokovic win percentage by year. '
    'Though the overall H2H is close - 27 - 23, Novak Djokovic had an upper hand in the H2H from 2011 till their last match in 2020'
)

# Load the preprocessed data

# %%


@st.cache_data  # this is a decorator that will cache the data for the session
def load_data():
    avg_novak_wins = pd.read_csv('novak_win_rate_by_year.csv')
    # Convert 'year' column to numeric
    avg_novak_wins['year'] = pd.to_numeric(
        avg_novak_wins['year'], errors='coerce')

    all_h2h_matches_all_columns = pd.read_csv('roger_novak_matches.csv')

    all_h2h_matches_all_columns['tourney_date'] = pd.to_datetime(
        all_h2h_matches_all_columns['tourney_date'], format='%Y%m%d'
    ).dt.strftime('%Y/%m/%d')
    # Just subset of columns to display on webapp
    all_matches_for_display = all_h2h_matches_all_columns[
        ['year', 'tourney_name', 'tourney_date', 'winner_name',
            'loser_name', 'score', 'round', 'surface']
    ]

    return avg_novak_wins, all_matches_for_display, all_h2h_matches_all_columns


# %%
avg_novak_wins, all_matches_for_display, all_h2h_matches_all_columns = load_data()

# %%
# Set the format for the dataframe display
format_dict = {
    'year': '{:.0f}'  # No comma, no decimal places
}

# Create an interactive line plot using Plotly
fig = px.line(
    avg_novak_wins, x='year', y='Novak_win_percentage',
    title='Novak Djokovic Win Percentage by Year',
    labels={'year': 'Year', 'Novak_win_percentage': 'Novak Win Percentage'},
    hover_data={'year': False, 'Novak_win_percentage': True,
                'Number_of_Matches_Played': True},
    markers=True
)

# Add reference lines
fig.add_hline(y=50, line_dash="dash", line_color="red",
              annotation_text="50% Win Rate")
fig.add_vline(x=2011, line_dash="dash", line_color="green",
              annotation_text="Year 2011- Novak's Dominance Begins")

# Update layout for better appearance
fig.update_layout(yaxis_range=[-10, 110], xaxis=dict(tickmode='linear'))

st.sidebar.title("Data Source Acknowledgement")

# Define the content for the 'About' page
st.sidebar.write(
    """
    Thanks to Jeff Sackman/Tennis Abstract for inspiration and for the data. Using this under the Creative Commons Attribution - Non Commercial License. 
    What Jeff and a team of volunteer contributors have done is amazing - especially the data charting exercise, painstakingly recording every point and attributing useful 
    features like shot depth, shot direction etc to every shot. Jeff's work can be found at https://github.com/JeffSackmann. I picked a couple of interesting data sets
    from his repo to make sense of the Federer-Djokovic head to head matchup. 
                     
    Federer is my favourite tennis player and Djokovic is my wife's favourite player. So anytime they played, it was a house divided :) 
    """
)

# Create two columns with a larger ratio for the first column
col1, col2 = st.columns([2, 2])

# Display the updated dataframe in the first column
with col1:
    st.dataframe(all_matches_for_display.style.format(format_dict), height=425)

# Display the plot in the second column
with col2:
    st.plotly_chart(fig, use_container_width=True)

# Create a table with head to head record split by surface
surface_summary = all_matches_for_display.groupby(
    'surface').size().reset_index(name='count')
surface_summary['Novak_wins'] = all_matches_for_display[
    all_matches_for_display['winner_name'] == 'Novak Djokovic'
].groupby('surface').size().reset_index(name='count')['count']
surface_summary['Roger_wins'] = all_matches_for_display[
    all_matches_for_display['winner_name'] == 'Roger Federer'
].groupby('surface').size().reset_index(name='count')['count']

# Fill NaN values with 0
surface_summary = surface_summary.fillna(0)

with col1:
    # Display the table
    st.write(
        "Below is a table showing the head to head record split by surface. Most matches were played on hard courts and the head to head record is quite close on all surfaces"
    )
    st.dataframe(surface_summary)

# Create a table with head to head record split by best_of
best_of_summary = all_h2h_matches_all_columns.groupby(
    'best_of').size().reset_index(name='count')
best_of_summary['Novak_wins'] = all_h2h_matches_all_columns[
    all_h2h_matches_all_columns['winner_name'] == 'Novak Djokovic'
].groupby('best_of').size().reset_index(name='count')['count']
best_of_summary['Roger_wins'] = all_h2h_matches_all_columns[
    all_h2h_matches_all_columns['winner_name'] == 'Roger Federer'
].groupby('best_of').size().reset_index(name='count')['count']

# Fill NaN values with 0
best_of_summary = best_of_summary.fillna(0)

with col1:
    # Display the table
    st.write(
        "Below is a table showing the head to head record split by best of sets. In best of 3 sets, record is pretty close but in best of 5 sets, Novak has a >60% win rate"
    )
    st.dataframe(best_of_summary)

# %%
st.write(
    "On the men's pro tour, the serve and return are the most important components of the game. "
    "I got the serve/return aggregated stats from each of these matches from Jeff Sackman's site. "
    "I picked some feature columns that I thought will be important and ran some basic ML model training/predicting. "
    "Also, obtained feature importance by using ensemble decision tree algorithm. Results and Explanation below:"
)

# Define the features and target
features = ['surface', 'best_of', 'minutes', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
            'w_bpSaved', 'w_bpFaced', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_bpSaved', 'l_bpFaced']
target = 'Novak_Wins'

# Prepare the data
# Drop rows with NaN or missing values in the relevant columns
all_h2h_matches_all_columns = all_h2h_matches_all_columns.dropna(
    subset=features + [target])

X = all_h2h_matches_all_columns[features]
y = all_h2h_matches_all_columns[target]

# Convert categorical features to numeric
X = pd.get_dummies(X, columns=['surface', 'best_of'], drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Train a logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# Display the classification report
st.write("### Logistic Regression Model Performance")
st.write("#### Classification Report")
st.text(classification_report(y_test, y_pred))

# Display the accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f"#### Accuracy: {accuracy:.2f}")

# Train a random forest classifier for feature importance
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Get feature importances
importances = rf_clf.feature_importances_
feature_importance_df = pd.DataFrame(
    {'feature': X.columns, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(
    by='importance', ascending=False)

# Plot feature importances
st.write("### Feature Importance from Random Forest")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df, ax=ax)
st.pyplot(fig)

# Summary of findings
st.write("### Summary of Findings")
st.write(
    """
    The logistic regression model achieved an accuracy of {:.2f}. The most important features identified by the random forest model are:
    - {}
    - {}
    - {}
    - {}
    - {}
    """.format(
        accuracy,
        feature_importance_df.iloc[0]['feature'],
        feature_importance_df.iloc[1]['feature'],
        feature_importance_df.iloc[2]['feature'],
        feature_importance_df.iloc[3]['feature'],
        feature_importance_df.iloc[4]['feature']
    )
)

# Exploratory Data Analysis
# Exploratory Data Analysis
st.write("### Exploratory Data Analysis")
# Create a comparison of all features with the Novak_Wins target
st.write("### Feature Comparison with Novak_Wins Target")


# Plot each feature against Novak_Wins
# Define a color palette for Novak_Wins and surface types
palette = {0: "orange", 1: "green"}
surface_palette = {"Hard": "blue", "Clay": "brown", "Grass": "green"}


# Plot 'minutes' against Novak_Wins
fig_minutes = px.box(
    all_h2h_matches_all_columns,
    x='Novak_Wins',
    y='minutes',
    color='Novak_Wins',
    title='minutes vs Novak_Wins',
    labels={'Novak_Wins': 'Novak Wins', 'minutes': 'Minutes'},
    color_discrete_map=palette
)
fig_minutes.update_layout(
    title_font_size=14,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12
)
st.plotly_chart(fig_minutes)

# Summary stats for 'minutes'
minutes_summary = all_h2h_matches_all_columns.groupby(
    'Novak_Wins')['minutes'].agg(['count', 'mean']).reset_index()
st.write("### Summary Stats for Minutes")
st.dataframe(minutes_summary)

st.write("Analysis: Matches that Novak wins tend to be longer in duration.")

# Plot 'w_1stIn' against Novak_Wins
fig_w_1stIn = px.box(
    all_h2h_matches_all_columns,
    x='Novak_Wins',
    y='w_1stIn',
    color='Novak_Wins',
    title='w_1stIn vs Novak_Wins',
    labels={'Novak_Wins': 'Novak Wins', 'w_1stIn': 'First Serve In'},
    color_discrete_map=palette
)
fig_w_1stIn.update_layout(
    title_font_size=14,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12
)
st.plotly_chart(fig_w_1stIn)

# Summary stats for 'w_1stIn'
w_1stIn_summary = all_h2h_matches_all_columns.groupby(
    'Novak_Wins')['w_1stIn'].agg(['count', 'mean']).reset_index()
st.write("### Summary Stats for First Serve In")
st.dataframe(w_1stIn_summary)

st.write("Analysis: Novak's first serve in percentage is higher in matches he wins.")

# Plot 'w_2ndWon' against Novak_Wins
fig_w_2ndWon = px.box(
    all_h2h_matches_all_columns,
    x='Novak_Wins',
    y='w_2ndWon',
    color='Novak_Wins',
    title='w_2ndWon vs Novak_Wins',
    labels={'Novak_Wins': 'Novak Wins', 'w_2ndWon': 'Second Serve Won'},
    color_discrete_map=palette
)
fig_w_2ndWon.update_layout(
    title_font_size=14,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12
)
st.plotly_chart(fig_w_2ndWon)

# Summary stats for 'w_2ndWon'
w_2ndWon_summary = all_h2h_matches_all_columns.groupby(
    'Novak_Wins')['w_2ndWon'].agg(['count', 'mean']).reset_index()
st.write("### Summary Stats for Second Serve Won")
st.dataframe(w_2ndWon_summary)

st.write("Analysis: Novak's second serve win percentage is also higher in matches he wins.")
