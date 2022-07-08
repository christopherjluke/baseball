#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 21:11:16 2022

@author: christopherluke
"""
conda install git
!git clone https://github.com/jldbc/pybaseball
cd pybaseball
pip install -e .

import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('chained_assignment', None)

import pybaseball as pyb

# fWAR batting and pitching

fwar_bat = pd.DataFrame(pyb.team_batting(2021).groupby('Team')['WAR'].sum().sort_values(ascending=False))
fwar_bat = fwar_bat.rename({'WAR': 'Batter fWAR'}, axis=1)

fwar_pitch = pd.DataFrame(pyb.team_pitching(2021).groupby('Team')['WAR'].sum().sort_values(ascending=False))
fwar_pitch = fwar_pitch.rename({'WAR': 'Pitcher fWAR'}, axis=1)

# bWAR batting and pitching

bwar_bat = pyb.bwar_bat(return_all=False)
bwar_bat = bwar_bat.loc[(bwar_bat['year_ID'] == 2021) & (bwar_bat['PA'] >= 50)]
bwar_bat = bwar_bat.groupby('team_ID')['WAR'].sum().sort_values(ascending=False)
bwar_bat = bwar_bat.to_frame()
bwar_bat = bwar_bat.rename({'WAR': 'Batter bWAR'}, axis=1)
bwar_bat = bwar_bat.rename(index={'team_ID': 'Team'})
bwar_bat['Batter bWAR'] = bwar_bat['Batter bWAR'].round(1)
bwar_bat.index.name = 'Team'


bwar_pitch = pyb.bwar_pitch(return_all=False)
bwar_pitch = bwar_pitch.loc[(bwar_pitch['year_ID'] == 2021)] 
bwar_pitch = bwar_pitch.groupby('team_ID')['WAR'].sum().sort_values(ascending=False)
bwar_pitch = bwar_pitch.to_frame()
bwar_pitch = bwar_pitch.rename({'WAR': 'Pitcher bWAR'}, axis=1)
bwar_pitch = bwar_pitch.rename(index={'team_ID': 'Team'})
bwar_pitch['Pitcher bWAR'] = bwar_pitch['Pitcher bWAR'].round(1)
bwar_pitch.index.name = 'Team'



# Standings

standings_2021 = pyb.standings(2021)

wins = pd.DataFrame()
for division_df in standings_2021:
    wins = pd.concat([wins, division_df])

wins = wins.rename({'Tm': 'Team'}, axis=1)
wins.iloc[0, 0] = 'TBR'
wins.iloc[1, 0] = 'BOS'
wins.iloc[2, 0] = 'NYY'
wins.iloc[3, 0] = 'TOR'
wins.iloc[4, 0] = 'BAL'
wins.iloc[5, 0] = 'CHW'
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

# Adding WARs to Standings

wins = fwar_bat.merge(wins, on='Team')
wins = fwar_pitch.merge(wins, on='Team')
wins = bwar_bat.merge(wins, on='Team')
wins = bwar_pitch.merge(wins, on='Team')

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns; sns.set_style('whitegrid');

# Batter fWAR

columns = ['Batter fWAR', 'W']
for column in columns:
    wins[column] = wins[column].astype(float)

x = wins['Batter fWAR']
y = wins['W']

# Line of best fit

m, b = np.polyfit(x.values, y.values, 1)

plt.figure(figsize=(15, 10))

plt.scatter(x, y)
plt.plot(x, x*m + b)

plt.title('Team Batting fWAR to Team Win Totals', fontsize=16)
plt.xlabel('Batting fWAR')
plt.ylabel('Team Wins')

ax = plt.gca()
sns.regplot(data = wins, x='Batter fWAR', y='W', 
            scatter_kws={"color": "black"}, line_kws={"color": "forestgreen"});

for _, row in wins.iterrows():
    ax.annotate(row['Team'], xy=(row['Batter fWAR'], row['W']))
    
# Pitcher fWAR

columns = ['Pitcher fWAR', 'W']
for column in columns:
    wins[column] = wins[column].astype(float)

x = wins['Pitcher fWAR']
y = wins['W']

# Line of best fit

m, b = np.polyfit(x.values, y.values, 1)

plt.figure(figsize=(15, 10))

plt.scatter(x, y)
plt.plot(x, x*m + b)

plt.title('Team Pitcher fWAR to Team Win Totals', fontsize=16)
plt.xlabel('Pitcher fWAR')
plt.ylabel('Team Wins')

ax = plt.gca()
sns.regplot(data = wins, x='Pitcher fWAR', y='W',
            scatter_kws={"color": "black"}, line_kws={"color": "forestgreen"});

for _, row in wins.iterrows():
    ax.annotate(row['Team'], xy=(row['Pitcher fWAR'], row['W']))

# Batter bWAR

columns = ['Batter bWAR', 'W']
for column in columns:
    wins[column] = wins[column].astype(float)

x = wins['Batter bWAR']
y = wins['W']

# Line of best fit

m, b = np.polyfit(x.values, y.values, 1)

plt.figure(figsize=(15, 10))

plt.scatter(x, y)
plt.plot(x, x*m + b)

plt.title('Team Batting bWAR to Team Win Totals', fontsize=16)
plt.xlabel('Batting bWAR')
plt.ylabel('Team Wins')

ax = plt.gca()
sns.regplot(data = wins, x='Batter bWAR', y='W',
            scatter_kws={"color": "dimgray"}, line_kws={"color": "maroon"});

for _, row in wins.iterrows():
    ax.annotate(row['Team'], xy=(row['Batter bWAR'], row['W']))

# Pitcher bWAR

columns = ['Pitcher bWAR', 'W']
for column in columns:
    wins[column] = wins[column].astype(float)

x = wins['Pitcher bWAR']
y = wins['W']

# Line of best fit

m, b = np.polyfit(x.values, y.values, 1)

plt.figure(figsize=(15, 10))

plt.scatter(x, y)
plt.plot(x, x*m + b)

plt.title('Team Pitcher bWAR to Team Win Totals', fontsize=16)
plt.xlabel('Pitcher bWAR')
plt.ylabel('Team Wins')

ax = plt.gca()
sns.regplot(data = wins, x='Pitcher bWAR', y='W',
            scatter_kws={"color": "dimgray"}, line_kws={"color": "maroon"});

for _, row in wins.iterrows():
    ax.annotate(row['Team'], xy=(row['Pitcher bWAR'], row['W']))

wins.corr(method='pearson').round(2)

