#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 19:49:26 2022

@author: christopherluke

Quick code if you want to see the correlation between total Team Wins and fWAR.
Thank you to the pybaseball folks for putting this package together.
You can play around with dates all you want, this one defaults to just the 2021 season.
One little quirk, Total Team WAR was added to the Team Batting Stats DataFrame. 
Unfortunately I'm not THAT good at Pandas so it's just for conveinence sake.
Another quirk, this does not include partial seasons, so players who have been traded aren't totaled in Team WAR (fine).
This unfortunately leads to a problem of running this for multiple seasons though (not fine).
In the process of testing what works for mutliple seasons so someone like Mookie Betts doesn't get lost.
"""

import pandas as pd
pd.__version__
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('chained_assignment', None)

import pybaseball as pyb
df_2021 = pyb.batting_stats(2021, qual=0)
df_2021.groupby('Team')['WAR'].sum().head(5)
df_2021.groupby('Team')
df_2021.groupby('Team')['WAR'].sum().sort_values(ascending=False)

df_2021_pitch = pyb.pitching_stats(2021, qual=0)
df_2021_pitch.groupby('Team')['WAR'].sum().head(5)
df_2021_pitch.groupby('Team')
df_2021_pitch.groupby('Team')['WAR'].sum().sort_values(ascending=False)

team_batting_war_2021 = df_2021.groupby('Team')['WAR'].apply(lambda group: group.nlargest(4).sum())
team_batting_war_2021 = team_batting_war_2021.reset_index()
team_batting_war_2021 = team_batting_war_2021.loc[team_batting_war_2021['Team'] != '- - -']

team_pitching_war_2021 = df_2021_pitch.groupby('Team')['WAR'].apply(lambda group: group.nlargest(4).sum())
team_pitching_war_2021 = team_pitching_war_2021.reset_index()
team_pitching_war_2021 = team_pitching_war_2021.loc[team_pitching_war_2021['Team'] != '- - -']

team_batting_war_2021['Total fWAR'] = team_batting_war_2021['WAR'] + team_pitching_war_2021['WAR']

standings = pyb.standings(2021)
wins = pd.DataFrame()
for division_df in standings:
    wins = pd.concat([wins, division_df])
    
# Standings are full team name
wins = wins.rename({'Tm': 'Team'}, axis=1)
wins.iloc[0, 0] = 'TBR'
wins.iloc[1, 0] = 'BOS'
wins.iloc[2, 0] = 'NYY'
wins.iloc[3, 0] = 'TOR'
wins.iloc[4, 0] = 'BAL'
wins.iloc[5, 0] = 'CWS'
wins.iloc[6, 0] = 'CLE'
wins.iloc[7, 0] = 'DET'
wins.iloc[8, 0] = 'KCR'
wins.iloc[9, 0] = 'MIN'
wins.iloc[10, 0] = 'HOU'
wins.iloc[11, 0] = 'SEA'
wins.iloc[12, 0] = 'OAK'
wins.iloc[13, 0] = 'LAA'
wins.iloc[14, 0] = 'TEX'
wins.iloc[15, 0] = 'ATL'
wins.iloc[16, 0] = 'PHI'
wins.iloc[17, 0] = 'NYM'
wins.iloc[18, 0] = 'MIA'
wins.iloc[19, 0] = 'WSN'
wins.iloc[20, 0] = 'MIL'
wins.iloc[21, 0] = 'STL'
wins.iloc[22, 0] = 'CIN'
wins.iloc[23, 0] = 'CHC'
wins.iloc[24, 0] = 'PIT'
wins.iloc[25, 0] = 'SFG'
wins.iloc[26, 0] = 'LAD'
wins.iloc[27, 0] = 'SDP'
wins.iloc[28, 0] = 'COL'
wins.iloc[29, 0] = 'ARI'
wins = wins.reset_index(drop=True)

team_batting_war_2021 = team_batting_war_2021.merge(wins, on='Team')
team_pitching_war_2021 = team_pitching_war_2021.merge(wins, on='Team')

# Batter fWAR

columns = ['WAR', 'W']
for column in columns:
    team_batting_war_2021[column] = team_batting_war_2021[column].astype(float)

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns; sns.set_style('whitegrid');

x = team_batting_war_2021['WAR']
y = team_batting_war_2021['W']

# line of best fit

m, b = np.polyfit(x.values, y.values, 1)

plt.figure(figsize=(15, 10))

plt.scatter(x, y)
plt.plot(x, x*m + b)

plt.title('2021 Team Wins vs fWAR of Top 4 Batters', fontsize=16)
plt.xlabel('fWAR')
plt.ylabel('Team Wins')

ax = plt.gca()
sns.regplot(data = team_batting_war_2021, x='WAR', y='W');

for _, row in team_batting_war_2021.iterrows():
    ax.annotate(row['Team'], xy=(row['WAR'], row['W']))

# Pitcher fWAR

columns = ['WAR', 'W']
for column in columns:
    team_pitching_war_2021[column] = team_pitching_war_2021[column].astype(float)
    
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns; sns.set_style('whitegrid');

x = team_pitching_war_2021['WAR']
y = team_pitching_war_2021['W']

# line of best fit

m, b = np.polyfit(x.values, y.values, 1)

plt.figure(figsize=(15, 10))

plt.scatter(x, y)
plt.plot(x, x*m + b)

plt.title('2021 Team Wins vs fWAR of Top 4 Pitchers', fontsize=16)
plt.xlabel('fWAR')
plt.ylabel('Team Wins')

ax = plt.gca()
sns.regplot(data = team_pitching_war_2021, x='WAR', y='W');

for _, row in team_pitching_war_2021.iterrows():
    ax.annotate(row['Team'], xy=(row['WAR'], row['W']))
    
# Total Team WAR

columns = ['Total fWAR', 'W']
for column in columns:
    team_batting_war_2021[column] = team_batting_war_2021[column].astype(float)

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns; sns.set_style('whitegrid');

x = team_batting_war_2021['Total fWAR']
y = team_batting_war_2021['W']

# line of best fit

m, b = np.polyfit(x.values, y.values, 1)

plt.figure(figsize=(15, 10))

plt.scatter(x, y)
plt.plot(x, x*m + b)

plt.title('2021 Team Wins vs Top 4 fWAR Pitchers and Batters', fontsize=16)
plt.xlabel('Team fWAR')
plt.ylabel('Team Wins')

ax = plt.gca()
sns.regplot(data = team_batting_war_2021, x='Total fWAR', y='W');

for _, row in team_batting_war_2021.iterrows():
    ax.annotate(row['Team'], xy=(row['Total fWAR'], row['W']))
