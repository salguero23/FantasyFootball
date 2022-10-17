import pandas as pd
import os
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder



def clean_flex():
    # Define constants and retrieve our workbook
    path = os.getcwd()
    workbook = pd.ExcelFile(f'{path}\\Data\\data.xlsx')
    sheets = [pd.read_excel(workbook,sheet_name=sheet) for sheet in workbook.sheet_names]
    POSITIONS = ['WR','RB','TE']
    YEAR = 2000
    
    # Define an empty dataframe to hold our cleaned data
    df = pd.DataFrame()

    # Iterate through each sheet in the workbook, clean it, and concat to one spreadsheet
    for sheet in tqdm(sheets):
        # Remove rows of empty and missing data
        empty_rows = list(sheet.loc[sheet['Rk']=='Rk'].index)
        NaN_rows = list(sheet.loc[sheet['Pos'].astype(str)=='nan'.index])
        empty_rows.extend(NaN_rows)

        sheet.drop(empty_rows,inplace=True)

        # Regular Expressions
        sheet['Pos'] = sheet['Pos'].apply(lambda x: x.upper())
        sheet['Player'] = sheet['Player'].replace('\*','',regex=True)
        sheet['Player'] = sheet['Player'].replace('\+','',regex=True)
        sheet['Receiving Ctch%'] = sheet['Receiving Ctch%'].apply(lambda x: x.strip('%')).astype(float)


        # Fill missing values with zero, meaning they didn't record any stats in that column
        sheet.fillna(value=0.0,inplace=True)
        sheet.reset_index(inplace=True,drop=True)


        # Identify string and float columns and ensure they are the correct data type
        str_columns = ['Player','Tm','Pos']
        float_columns = list(sheet.drop(columns=str_columns).columns)
        sheet[float_columns] = sheet[float_columns].astype(float)
        sheet['Year'] = YEAR


        # Append the current sheet to the dataframe
        df = pd.concat([df,sheet],axis=1)
        df.reset_index(inplace=True,drop=True)
        YEAR += 1


    # Retain only the flex options
    df = df.loc[df['Pos'].isin(POSITIONS)]

    # Map the current team locations and names to the historical locations and names
    update_team = {}
    for team in df['Tm'].unique():
        if team == 'SDG':
            update_team[team] = 'LAC'
        elif team == 'OAK':
            update_team[team] = 'LVR'
        elif team == 'STL':
            update_team[team] = 'LAR'
        else:
            update_team[team] = team

    df['Tm'] = df['Tm'].map(update_team)
    df.drop(columns='Rk',inplace=True)


    # Calculate fantasy points (assuming half PPR league)
    df['Fpts'] =  (df['Receiving Rec']*0.5) + (df['Total Yds YScm']*0.1) + (df['RRTD']*6) - (df['Fmb']*2)

    # Save and print
    print(df.head())
    print(df.shape)
    df.to_csv(f'{path}\\Data\\cleanedData.csv',index=False)


def clean_qb():
    # Define constants and retrieve our workbook
    path = os.getcwd()
    workbook = pd.ExcelFile(f'{path}\\Data\\data.xlsx')
    sheets = [pd.read_excel(workbook,sheet_name=sheet) for sheet in workbook.sheet_names]
    POSITIONS = ['QB']
    YEAR = 2000

    # Define an empty dataframe to hold our cleaned data
    df = pd.DataFrame()

    # Iterate through each sheet in the workbook, clean it, and concat to one spreadsheet
    for sheet in tqdm(sheets):
        # Remove rows of empty and missing data
        empty_rows = list(sheet.loc[sheet['Rk']=='Rk'].index)
        NaN_rows = list(sheet.loc[sheet['Pos'].astype(str)=='nan'].index)
        empty_rows.extend(NaN_rows)

        sheet.drop(empty_rows,inplace=True)

        # Regular Expressions
        sheet['Pos'] = sheet['Pos'].apply(lambda x: x.upper())
        sheet['Player'] = sheet['Player'].replace('\*','',regex=True)
        sheet['Player'] = sheet['Player'].replace('\+','',regex=True)
        # sheet['Cmp%'] = sheet['Cmp%'].apply(lambda x: x.strip('%')).astype(float)
        # sheet['TD%'] = sheet['TD%'].apply(lambda x: x.strip('%')).astype(float)
        # sheet['Int%'] = sheet['Int%'].apply(lambda x: x.strip('%')).astype(float)
        # sheet['Sk%'] = sheet['Sk%'].apply(lambda x: x.strip('%')).astype(float)


        # Fill missing values with zero, meaning they didn't record any stats in that column
        sheet['QBrec'].fillna('0',inplace=True)
        sheet.fillna(value=0.0,inplace=True)
        sheet.reset_index(inplace=True,drop=True)

        # Clean win/loss data to return only wins
        sheet['Wins'] = sheet['QBrec'].apply(lambda x: x.split('-')[0])
        sheet.drop(columns=['QBrec'],inplace=True)


        # Identify string and float columns and ensure they are the correct data type
        str_columns = ['Player','Tm','Pos']
        float_columns = list(sheet.drop(columns=str_columns).columns)
        sheet[float_columns] = sheet[float_columns].astype(float)
        sheet['Year'] = YEAR

        # Append the current sheet to the dataframe
        df = pd.concat([df,sheet],axis=0)
        df.reset_index(inplace=True,drop=True)
        YEAR += 1

    # Retain only the qb option
    df = df.loc[df['Pos'].isin(POSITIONS)]

    # Map the current team locations and names to the historical locations and names
    update_team = {}
    for team in df['Tm'].unique():
        if team == 'SDG':
            update_team[team] = 'LAC'
        elif team == 'OAK':
            update_team[team] = 'LVR'
        elif team == 'STL':
            update_team[team] = 'LAR'
        else:
            update_team[team] = team

    df['Tm'] = df['Tm'].map(update_team)
    df.drop(columns='Rk',inplace=True)


    # Calculate fantasy points (assuming half PPR league)
    df['Fpts'] = (df['Yds']*0.04) + (df['TD']*6) - (df['Int']*2)

    # Drop QBR and rename Rate to QBR
    df.drop(columns=['QBR'],inplace=True)
    df.rename(columns={'Rate':'QBR'},inplace=True)


    # Save and print
    print(df.head())
    print(df.shape)
    df.to_csv(f'{path}\\Data\\cleanedData.csv',index=False)


def engineer_data():
    path = os.getcwd()

    # Import the cleaned data
    df = pd.read_csv(f'{path}\\Data\\cleanedData.csv')
    df.index = df.Player

    PLAYERS = df.index.unique()
    YEARS = df.Year.unique()
    ml_df = pd.DataFrame()

    # Iterate over every player and create a new df where we have last years stats paired
    # with next years fantasy points generated
    for player in PLAYERS:
        try:
            temp = df.loc[player]
            temp['Fpts'] = temp['Fpts'].shift(-1)
            ml_df = pd.concat([ml_df,temp])
        except:
            continue

    # Save modeling data
    ml_df.dropna(inplace=True)
    ml_df.to_csv(f'{path}\\Data\\mlData.csv',index=False)

    # Get the latest season to calculate the projections for the upcoming season
    latest_season = df.loc[df['Year']==YEARS[-1]].dropna()
    latest_season.to_csv(f'{path}\\Data\\latestSeason.csv')


def getdata_qb():
    path = os.getcwd()

    # Import data
    df = pd.read_csv(f'{path}\\Data\mlData.csv')

    PLAYERS =  df.Player
    YEAR = df.Year + 1

    df.drop(columns=['Pos','Player','Year','Tm'],inplace=True)
    df.sample(frac=1, random_state=627)

    PLAYERS.sample(frac=1,random_state=627)
    YEAR = YEAR.sample(frac=1,random_state=627)

    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    trainX, testX, trainy, testy = train_test_split(X,y,test_size=0.2,random_state=256)
    trainX, valX, trainy, valy = train_test_split(trainX,trainy,test_size=0.2,random_state=153)

    trainPlayers, valPlayers, testPlayers = PLAYERS.loc[trainX.index], PLAYERS.loc[valX.index], PLAYERS.loc[testX.index]
    # trainPosition, valPosition, testPosition = POSITION.loc[trainX.index], POSITION.loc[valX.index], POSITION.loc[testX.index]
    # trainYear, valYear, testYear = YEAR.loc[trainX.index], YEAR.loc[valX.index], YEAR.loc[testX.index]


    scaler = StandardScaler()
    trainX = scaler.fit_transform(trainX)
    valX = scaler.transform(valX)
    testX = scaler.transform(testX)


    trainX = pd.DataFrame(index=trainPlayers,data=trainX,columns=X.columns)
    valX = pd.DataFrame(index=valPlayers,data=valX,columns=X.columns)
    testX = pd.DataFrame(index=testPlayers,data=testX,columns=X.columns)

    trainy.index = trainPlayers
    valy.index = valPlayers
    testy.index = testPlayers


    return trainX, trainy, valX, valy, testX, testy





    



# def get_flex():
#     '''
#     This function takes no arguments, it is designed to clean the mlData.csv
#     file located in the Data subfolder within the Flex folder. Returns training,
#     validating, and testing data only. No context is given with which player is what
#     row.
#     '''

#     ohe = OneHotEncoder(sparse=False)   
#     path = os.getcwd()

#     # Import data and clean
#     df = pd.read_csv(f'{path}\\Data\\mlData.csv')
#     # Encode positions
#     encoded_positions = pd.DataFrame(ohe.fit_transform(df[['Pos']]),columns=ohe.categories_[0])
#     df = pd.concat([df,encoded_positions],axis=1)

#     # Drop irrelevant columns
#     df.drop(columns=['Pos','Player','Year'],inplace=True)
