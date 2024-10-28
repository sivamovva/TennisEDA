import streamlit as st

import plotly.graph_objects as go


def plot_yearly_win_rate_trend(df_concat_subset_grouped_by_year, user_selected_player):

    fig = go.Figure()

    # Add win percentage trace
    win_percentage_color = 'green'
    matches_played_color = 'blue'

    fig.add_trace(go.Scatter(
        x=df_concat_subset_grouped_by_year['year'],
        y=df_concat_subset_grouped_by_year[f'{user_selected_player}_win_percentage'],
        mode='lines+markers',
        name=f'{user_selected_player} Win Percentage',
        yaxis='y1',
        line=dict(color=win_percentage_color)
    ))
    # Add bar trace for number of matches played
    fig.add_trace(go.Bar(
        x=df_concat_subset_grouped_by_year['year'],
        y=df_concat_subset_grouped_by_year['Number_of_Matches_Played'],
        name='Number of Matches Played',
        yaxis='y2',
        marker=dict(color=matches_played_color),
        opacity=0.6
    ))

    # Update layout for dual y-axes
    fig.update_layout(
        title=f'{user_selected_player} Performance timeline',
        xaxis=dict(title='Year'),
        yaxis=dict(
            title=f'{user_selected_player} Win Percentage',
            titlefont=dict(color=win_percentage_color),
            tickfont=dict(color=win_percentage_color)
        ),
        yaxis2=dict(
            title='Number of Matches Played',
            titlefont=dict(color=matches_played_color),
            tickfont=dict(color=matches_played_color),
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.1, y=1.1, orientation='h')
    )

    fig.update_xaxes(showgrid=True, gridwidth=1,
                     gridcolor='LightGray', tickmode='linear', dtick=1)

    st.plotly_chart(fig)
