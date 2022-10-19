import os
import sys

path = os.path.dirname(os.getcwd()) + '\\functions'
sys.path.insert(0,path)

import pandas as pd
import numpy as np
from RegscorePy.mallow import mallow
from tqdm import tqdm
from datapreper import getdata_qb

from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor


# Get data
trainX, trainy, valX, valy, testX, testy = getdata_qb()


# Build a Random Forest model and save the feautre importance attribute as a series
rf =  RandomForestRegressor()
rf.fit(trainX,trainy)

rf_features = pd.Series(index=trainX.columns,data=rf.feature_importances_).sort_values(ascending=False)


# Build a Lasso model and save the model coefficients as a series
model = Lasso(random_state=75)
model.fit(trainX,trainy)

lasso_features = pd.Series(index=trainX.columns,data=np.abs(model.coef_)).sort_values(ascending=False)


# Use Mallows Cp to find the most import features
k = len(trainX.columns) + 1
p = 2
mallows_cp = pd.Series(index=trainX.columns,dtype='float64')

model = LinearRegression()
model.fit(trainX,trainy)
fittedvalues = model.predict(trainX)

for column in tqdm(trainX.columns):
    variable = trainX[[column]]
    subModel = LinearRegression()
    subModel.fit(variable, trainy)
    sub_fittedvalues = subModel.predict(variable)

    coef = mallow(trainy, fittedvalues, sub_fittedvalues, k, p)
    mallows_cp.loc[column] = coef

mallows_cp = mallows_cp.sort_values()

# Concat the three series and save to a csv
feature_selection = pd.concat([rf_features, lasso_features, mallows_cp],axis=1)
feature_selection.rename(columns={0:'Baruta Algorithm',1:'Lasso Selection',2:'Mallows Cp'},inplace=True)
feature_selection.to_csv(f'{os.getcwd()}\\Data\\featureSelection.csv')