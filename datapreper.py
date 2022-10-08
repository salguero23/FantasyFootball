import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

path = os.getcwd()
ohe = OneHotEncoder(sparse=False)

def get_data():
    # Import data and clean
    df = pd.read_csv(f'{path}\\Data\\mlData.csv')
    encoded_position = pd.DataFrame(ohe.fit_transform(df[['Pos']]),columns=ohe.categories_[0])
    df = pd.concat([df,encoded_position],axis=1)

    PLAYERS = df.Player
    POSITION = df.Pos
    YEAR = df.Year + 1

    df.drop(columns=['Pos','Player','Year'],inplace=True)
    COLUMNS = ['Games GS', 'Receiving Tgt', 'Receiving Rec', 'Receiving Yds',
            'Receiving TD', 'Receiving 1D', 'Receiving Lng', 'Receiving R/G',
            'Receiving Y/G', 'Rushing Att', 'Rushing Yds', 'Rushing TD',
            'Rushing 1D', 'Total Yds Touch', 'Total Yds YScm', 'RRTD', 'RB',
            'TE','WR','Fpts']

    df = df[COLUMNS]
    df = df.sample(frac=1,random_state=627)

    PLAYERS = PLAYERS.sample(frac=1,random_state=627)
    POSITION = POSITION.sample(frac=1,random_state=627)
    YEAR = YEAR.sample(frac=1,random_state=627)

    columns_to_drop = ['WR','TE','RB','Rushing TD','Receiving TD','Receiving Rec','Receiving 1D','Rushing Att','Total Yds Touch']
    df.drop(columns=columns_to_drop,inplace=True)


    # Split data into training, validating, and cleaning sets
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    trainX, testX, trainy, testy = train_test_split(X,y,test_size=0.2,random_state=256)
    trainX, valX, trainy, valy = train_test_split(trainX,trainy,test_size=0.2,random_state=153)

    trainPlayers, valPlayers, testPlayers = PLAYERS.loc[trainX.index], PLAYERS.loc[valX.index], PLAYERS.loc[testX.index]
    trainPosition, valPosition, testPosition = POSITION.loc[trainX.index], POSITION.loc[valX.index], POSITION.loc[testX.index]
    trainYear, valYear, testYear = YEAR.loc[trainX.index], YEAR.loc[valX.index], YEAR.loc[testX.index]

    scaler = StandardScaler()
    trainX = scaler.fit_transform(trainX)
    valX = scaler.transform(valX)
    testX = scaler.transform(testX)

    return trainX, trainy, valX, valy, testX, testy
