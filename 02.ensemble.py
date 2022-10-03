# Import packages
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import os
path = os.getcwd()
ohe = OneHotEncoder(sparse=False)


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

print((trainX.shape,valX.shape,testX.shape))


# Build an ensemble of models
MODELS = {'Ridge': Ridge(), 'RF': RandomForestRegressor(), 'SVM': SVR()}
best_model = {}
parameters = {
                'Ridge': {'alpha': list(np.arange(0.1,1000,10)),
                         'fit_intercept': [True, False]},

                'RF': {'n_estimators': list(np.arange(30,60,5))} ,

                'SVM': {'kernel': ['linear','poly','rbf','sigmoid'],
                        'C': list(np.arange(0.1,50,5))}
            }

# Grid search the desired models
for model, params in tqdm(parameters.items()):
    best_model[model] = GridSearchCV(MODELS[model], params, cv=5).fit(trainX,trainy).best_estimator_


# Train models, test, and calculate projections for the current season
predictions = pd.DataFrame()
predictions['Player'] = testPlayers
predictions['Pos'] = testPosition
predictions['Year'] = testYear
predictions['Y'] = testy

errors = pd.DataFrame()
errors['Player'] = testPlayers
errors['Pos'] = testPosition
errors['Year'] = testYear


# Import the lastest season data and calculate projections for the current season
latest_season = pd.read_csv(f'{path}\\Data\\2021Season.csv')
latest_season.index = latest_season.Player

POSITION = latest_season['Pos']

latest_season = latest_season[df.columns[:-1]]
latest_season = scaler.transform(latest_season)
projections = pd.DataFrame(POSITION)

# Loop through each model
for name, model in tqdm(best_model.items()):
    MODEL = model.fit(trainX,trainy)

    predictions[name] = model.predict(testX)
    errors[name] = testy - model.predict(testX)

    projections[name] = model.predict(latest_season)

    # Save model
    FILENAME = f'{name}_model.pkl'
    with open(f'{path}\\Models\\{FILENAME}', 'wb') as file:
        pickle.dump(model, file)


# Save results to CSVs
predictions.to_csv(f'{path}\\Data\\testPredictions.csv',index=False)
errors.to_csv(f'{path}\\Data\\testErrors.csv',index=False)
projections.to_csv(f'{path}\\Data\\ensembleProjections.csv')

print(projections.head())