# TennisEDA
Exploratory data analysis of Tennis data

Acknowledgement for data source: Jeff Sackmann's github repository where there are a library of tennis data sets at various levels of granularity. Licensed under Creative Commons Attribution. Thank you Jeff and collaborators.

Figuring out if I can glean any insights in the Federer-Djokovic matchup. Their H2H record ended up really close, and they played a lot of matches (>50), so there could be some real patterns. Once I do this, eventually want to scale this to all professional tennis matches. 


datasets of interest:
1) match summary with serve stats only - repo (tennis_atp), files (atp_matches_{year}.csv)
columns: 
tourney_id,tourney_name,surface,draw_size,tourney_level,tourney_date,match_num,winner_id,winner_seed,winner_entry,winner_name,winner_hand,winner_ht,winner_ioc,winner_age,loser_id,loser_seed,loser_entry,loser_name,loser_hand,loser_ht,loser_ioc,loser_age,score,best_of,round,minutes,w_ace,w_df,w_svpt,w_1stIn,w_1stWon,w_2ndWon,w_SvGms,w_bpSaved,w_bpFaced,l_ace,l_df,l_svpt,l_1stIn,l_1stWon,l_2ndWon,l_SvGms,l_bpSaved,l_bpFaced,winner_rank,winner_rank_points,loser_rank,loser_rank_points

Note - this is the only dataset that has score and winner_name. All the datasets in the matchchartingproject repo just have player1, player2 based on who served first.

2) Master datatable for point by point charting - repo (tennis_matchchartingproject), file(charting-m-matches.csv)
This is to mainly see who is player1 (served first), and player 2 (served second)
columns:
match_id,Player 1,Player 2,Pl 1 hand,Pl 2 hand,Gender,Date,Tournament,Round,Time,Court,Surface,Umpire,Best of,Final TB?,Charted by 

3) overview stats from charting data - repo (tennis_matchchartingproject), file(charting-m-stats-overview.csv)
columns:
match_id,player,set,serve_pts,aces,dfs,first_in,first_won,second_in,second_won,bk_pts,bp_saved,return_pts,return_pts_won,winners,winners_fh,winners_bh,unforced,unforced_fh,unforced_bh


4) rally stats (winner, forced/unforced errors) - repo(tennis_matchchartingproject) , file (charting-m-stats-rally.csv)
columns:
match_id,row,pts,pl1_won,pl1_winners,pl1_forced,pl1_unforced,pl2_won,pl2_winners,pl2_forced,pl2_unforced

can combine with #2 above to get nnames of pl1, pl2

5) Serve return outcome details - repo(tennis_matchchartingproject), file(charting-m-stats-returnoutcomes.csv)
has details on return performance
columns:
match_id	player	row	pts	pts_won	returnable	returnable_won	in_play	in_play_won	winners	total_shots

6) Rally Shot direction outcome details - repo (tennis_matchchartingproject), file (charting-m-stats-shortdirectionoutcomes.csv) - This data doesnt include all shots - serve, return and many third shots not included. So generally gives a sense of shot direction once the players settle into a rally. Useful stuff, especially starting in the 2000's where rallies got longer

7) Key Points - In tennis, all points are not equal. Performance on key points matters a lot - In this data set, key points are breakpoints (on serve and return), game points (serve and return) and deuce points(serve and return). 30-ALL is also a key point according to me, and there is raw data available to parse the stats on 30-ALL but i will save that for later. Currently going to use the summary stats available.

repo (tennis_matchchartingproject), files (charting-m-stats-keypointsreturn.csv, charting-m-stats-keypointsserve.csv)

8) Return depth details = repo(tennis_matchchartingproject), file(charting-m-stats-returndepth.csv)
columns:
match_id	player	row	returnable	shallow	deep	very_deep	unforced	err_net	err_deep	err_wide	err_wide_deep   


Approach:

1) First, aggregate the data at the match level. Each row is 1 match. Create a unique id column for each match - to use for dataset merging

2) Work with each of the feature data frames to engineer features (for eg create percentage columns etc)

Features from #1 (many features will have 2 columns - 1 for winner, 1 for loser, some features will be just 1 column - like surface, how close the match was etc): 1st serve pct, surface, duration, measure of how close the match was, best_of_sets

Target from #1 - Novak_Wins column (1 or 0 for each match) - this will be auto calculated for who ever has the H2H advantage

Features from #3 (overview stats): (each will have 2 columns - for winner and loser)
'aces_perc', 'dfs_perc','first_in_perc', 'first_won_perc', 'second_won_perc', 'bp_saved_perc',
'return_pts_won_perc', 'winners_unforced_perc', 'winner_fh_perc',
'winners_bh_perc', 'unforced_fh_perc', 'unforced_bh_perc'

3) Key points, return outcomes, Rally shot direction, and return depth, and serve direction - these all can be next level of detail. 

- return depth (% of deep, shallow, etc per match)
- serve influence file (look into this - it has % win by shot length)
- shot direction outcomes (we can get a ratio of shots_in_pts_won/(shots_in_pts_won + shots_in_pts_lost) and make this almost a shot effectiveness metric - probability of winning a point if a player played that shot in that match; so for each match, we can have F-XCeff, F-DLeff etc features and we can draw conclusions like if F-XC was really effective on a particular day, player1 had a good chance of winning that match )
- key points on serve and return - just filter the total column and get % of key points won per player per match -this shoudl be a pretty strong 
feature


10/24 - revamping the idea of the app.

realized that any head to head match up has a max of 50 matches and most of them are less than that. This is not enough data to train any model on. For example, when looking at Federer vs roddick matchup, xgboost or random forests struggle to generate a feature importnace list - because the dataset is very unbalanced (roddick won just 3 out of the 23 meetings between these two). More over, if i were to go with the original idea of selecting any 2 players and the main insights between those 2 players, for a lot of matchups, there was just 1 match (fed vs sampras) and hard to glean statistical insights from that. 

Instead of this, what i can do is:
1) for any selected player (include WTA also) what were the top factors that determined whether they won or lost against rest of the tour? on average?
2) once we had the above we can go into individual matchups to see how the model from (1) is performing for that specific matchup (even though the specific matchup will also be in the training data, it will still be fun to show the user how the model performs) - hopefully here, we can show that what worked for federer against most the tour did not work for him against Nadal :)



