import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pickle

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

import os
import sys
path = os.path.dirname(os.getcwd()) + '\\functions'
sys.path.insert(0,path)
from datapreper import getdata_qb, scale_data

path = os.getcwd()

# Get data and prep columns
trainX, trainy, valX, valy, testX, testy = getdata_qb()
columns_to_retain = ['40-49 FGA','Scoring XP%','Scoring FG%','Scoring XPM','30-39 FGA']

trainX = trainX[columns_to_retain]
valX = valX[columns_to_retain]
testX = testX[columns_to_retain]


# Build callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.5,
    patience=20,
    restore_best_weights=True
)

log_dir = f'Logs\\Neural Networks\\{datetime.now().strftime("%Y%m%d-%H%M%S")}'
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


# Build model structure and complie
model = keras.Sequential([
    layers.Dense(units=25,input_shape=[len(columns_to_retain)],activation='relu'),
    layers.Dropout(0.25),
    layers.BatchNormalization(),
    layers.Dense(units=20,activation='relu'),
    layers.Dropout(0.25),
    layers.BatchNormalization(),
    layers.Dense(units=1)
])

model.compile(optimizer='adam',loss='mae')
print(model.summary())

# Train & save model
history = model.fit(trainX,trainy,validation_data=(valX,valy),batch_size=32,epochs=400,callbacks=[early_stopping,tensorboard_callback])
pd.DataFrame(history.history).to_csv(f'{path}\\Data\\trainingHistory.csv',index=False)

with open(f'{path}\\Models\\NeuralNet_model.pkl', 'wb') as file:
    pickle.dump(model, file)


# Import data to calculate testing predictions, errors, and project the latest season
predictions = pd.read_csv(f'{path}\\Data\\testPredictions.csv')
yHat = pd.DataFrame(model.predict(testX),columns=['NeuralNet'])
predictions = pd.concat([predictions,yHat],axis=1)
predictions.to_csv(f'{path}\\Data\\testPredictions.csv',index=False)

errors = pd.read_csv(f'{path}\\Data\\testErrors.csv')
error_df = pd.DataFrame(testy.values - yHat.values.reshape(len(yHat),),columns=['NeuralNet'])
errors = pd.concat([errors,error_df],axis=1)
errors.to_csv(f'{path}\\Data\\testErrors.csv',index=False)

projections = pd.read_csv(f'{path}\\Data\\ensembleProjections.csv')


# Import the lastest season data and calculate projections for the current season
latestSeason = scale_data(columns_to_retain)

proj = pd.DataFrame(model.predict(latestSeason),columns=['NeuralNet'])
projections = pd.concat([projections,proj],axis=1)
projections.to_csv(f'{path}\\Data\\ensembleProjections.csv',index=False)

print(projections.head())