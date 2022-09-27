import os
import pandas as pd
path = os.getcwd()

df = pd.read_csv(f'{path}//Data//cleanedData.csv')
df.index = df.Player
columns_to_drop = ['Receiving Y/R','Rushing Y/A','Receiving Ctch%','Total Yds Y/Tch',
                    'Receiving Y/Tgt','Games G','Rushing Lng','Age','Fmb','Rushing A/G',
                    'Rushing Y/G','Tm','Player']
df.drop(columns=columns_to_drop,inplace=True)

# Derive ML Data & independent features to project 2022 season
PLAYERS = df.index.unique()
YEARS = df.Year.unique()
ml_df = pd.DataFrame()

for player in PLAYERS:
    try:
        temp = df.loc[player]
        temp['Fpts'] = temp['Fpts'].shift(-1)
        ml_df = pd.concat([ml_df,temp])
    except:
        continue

# Save modeling data
ml_df.dropna(inplace=True)
ml_df.to_csv(f'{path}\\Data\\mlData.csv')

# Get 2021 seasib
season_2021 = df.loc[df['Year']==2021].dropna()
season_2021.to_csv(f'{path}\\Data\\2021Season.csv')