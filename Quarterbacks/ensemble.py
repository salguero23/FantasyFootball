import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import os
import sys
path = os.path.dirname(os.getcwd()) + '\\functions'
sys.path.insert(0,path)
from datapreper import getdata_qb, scale_data


# Get data and prep columns
trainX, trainy, valX, valy, testX, testy = getdata_qb()
columns_to_retain = ['TD','Y/G','1D','Yds','Age','Wins']

trainX = trainX[columns_to_retain]
valX = valX[columns_to_retain]
testX = testX[columns_to_retain]


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
predictions['Y'] = testy

errors = pd.DataFrame()


# Import the lastest season data and calculate projections for the current season
path = os.getcwd()
latest_season = pd.read_csv(f'{path}\\Data\\latestSeason.csv')
latest_season.index = latest_season.Player

POSITION = latest_season['Pos']
latest_season = latest_season[columns_to_retain]

projections = pd.DataFrame(POSITION)
latestSeason = scale_data(columns_to_retain)



# Loop through each model
for name, model in tqdm(best_model.items()):
    MODEL = model.fit(trainX,trainy)

    predictions[name] = model.predict(testX)
    errors[name] = testy - model.predict(testX)

    projections[name] = model.predict(latestSeason)

    # Save model
    FILENAME = f'{name}_model.pkl'
    with open(f'{path}\\Models\\{FILENAME}', 'wb') as file:
        pickle.dump(model, file)



# Save data to csv
predictions.to_csv(f'{path}\\Data\\testPredictions.csv',index=False)
errors.to_csv(f'{path}\\Data\\testErrors.csv',index=False)
projections.to_csv(f'{path}\\Data\\ensembleProjections.csv')