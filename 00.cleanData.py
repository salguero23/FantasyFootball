from tqdm import tqdm
import pandas as pd
import os

path = os.getcwd()
workbook = pd.ExcelFile(f'{path}\\Data\\data.xlsx')
sheets = [pd.read_excel(workbook,sheet_name=sheet) for sheet in workbook.sheet_names]
FLEX = ['WR','RB','TE']
YEAR = 2000
df = pd.DataFrame()

for sheet in tqdm(sheets):
    empty_rows = list(sheet.loc[sheet['Rk']=='Rk'].index)
    NaN_rows = list(sheet.loc[sheet['Pos'].astype(str)=='nan'].index)
    empty_rows.extend(NaN_rows)

    sheet.drop(empty_rows,inplace=True)

    sheet['Pos'] = sheet['Pos'].apply(lambda x: x.upper())
    sheet.fillna(value=0.0,inplace=True)
    sheet['Player'] = sheet['Player'].replace('\*','',regex=True)
    sheet['Player'] = sheet['Player'].replace('\+','',regex=True)
    sheet['Receiving Ctch%'] = sheet['Receiving Ctch%'].apply(lambda x: x.strip('%')).astype(float)

    sheet.reset_index(inplace=True,drop=True)

    str_columns = ['Player','Tm','Pos']
    float_columns = list(sheet.drop(columns=str_columns).columns)
    sheet[float_columns] = sheet[float_columns].astype(float)
    sheet['Year'] = YEAR

    df = pd.concat([df,sheet],axis=0)
    df.reset_index(inplace=True,drop=True)
    YEAR += 1

df['Fpts'] = (df['Receiving Rec']*0.5) + (df['Receiving Yds']*0.1) + (df['Rushing Yds'] * 0.1) + (df['RRTD']*6) - (df['Fmb']*2)
df = df.loc[df['Pos'].isin(FLEX)]

team_update = {}
for team in df['Tm'].unique():
    if team == 'SDG':
        team_update[team] = 'LAC'
    elif team == 'OAK':
        team_update[team] = 'LVR'
    elif team == 'STL':
        team_update[team] = 'LAR'
    else:
        team_update[team] = team

df['Tm'] = df['Tm'].map(team_update)
df.drop(columns='Rk',inplace=True)

print(df.head())
print(df.shape)
df.to_csv('cleanedData.csv',index=False)
