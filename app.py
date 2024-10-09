import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the preprocessed data
avg_novak_wins = pd.read_csv('avg_novak_wins.csv')

# Convert 'year' column to numeric
avg_novak_wins['year'] = pd.to_numeric(avg_novak_wins['year'], errors='coerce')

# Plot the data using Matplotlib
fig, ax = plt.subplots()
ax.plot(avg_novak_wins['year'], avg_novak_wins['Novak_Wins'], marker='o')
ax.set_title('Novak Win Rate Over Federer by Year')
ax.set_xlabel('Year')
ax.set_ylabel('% of Matches Won by Novak')
ax.axhline(y=0.5, color='r', linestyle='--', label='50% Win Rate')
ax.set_xticks(avg_novak_wins['year'])
ax.tick_params(axis='x', rotation=45)
ax.axvline(x=2011, color='g', linestyle='--',
           label='2011: Start of Novak\'s Dominance')
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)

# Display the data in Streamlit
st.write('Hello World, my first Streamlit app')
st.line_chart(avg_novak_wins.set_index('year')['Novak_Wins'])
