# TennisEDA
Exploratory data analysis of Tennis data

Acknowledgement for data source: Jeff Sackmann's github repository where there are a library of tennis data sets at various levels of granularity. Licensed under Creative Commons Attribution. Thank you Jeff.

Figuring out if I can glean any insights in the Federer-Djokovic matchup. Their H2H record ended up really close, and they played a lot of matches (>50), so there could be some real patterns. 


datasets of interest:
1) match summary with serve stats only - repo (tennis_atp), files (atp_matches_{year}.csv)
columns: 
tourney_id,tourney_name,surface,draw_size,tourney_level,tourney_date,match_num,winner_id,winner_seed,winner_entry,winner_name,winner_hand,winner_ht,winner_ioc,winner_age,loser_id,loser_seed,loser_entry,loser_name,loser_hand,loser_ht,loser_ioc,loser_age,score,best_of,round,minutes,w_ace,w_df,w_svpt,w_1stIn,w_1stWon,w_2ndWon,w_SvGms,w_bpSaved,w_bpFaced,l_ace,l_df,l_svpt,l_1stIn,l_1stWon,l_2ndWon,l_SvGms,l_bpSaved,l_bpFaced,winner_rank,winner_rank_points,loser_rank,loser_rank_points


2) Master datatable for point by point charting - repo (tennis_matchchartingproject), file(charting-m-matches.csv)
This is to mainly see who is player1 (served first), and player 2 (served second)
columns:
match_id,Player 1,Player 2,Pl 1 hand,Pl 2 hand,Gender,Date,Tournament,Round,Time,Court,Surface,Umpire,Best of,Final TB?,Charted by 


3) Now, for rally stats (winner, forced/unforced errors) - repo(tennis_matchchartingproject) , file (charting-m-stats-rally.csv)
columns:
match_id,row,pts,pl1_won,pl1_winners,pl1_forced,pl1_unforced,pl2_won,pl2_winners,pl2_forced,pl2_unforced

can combine with #2 above to get nnames of pl1, pl2

4) Serve return outcome details - repo(tennis_matchchartingproject), file(charting-m-stats-returnoutcomes.csv)
has details on return performance
columns:
match_id	player	row	pts	pts_won	returnable	returnable_won	in_play	in_play_won	winners	total_shots

5) 
