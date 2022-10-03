# Import packages
import pandas as pd
from datetime import datetime
import pickle

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

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


# Build callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.5,
    patience=20,
    restore_best_weights=True
)

log_dir = f'Logs\\{datetime.now().strftime("%Y%m%d-%H%M%S")}'
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


# Build model structure and complie
model = keras.Sequential([
    layers.Dense(units=25,input_shape=[10],activation='relu'),
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
latest_season = pd.read_csv(f'{path}\\Data\\2021Season.csv')
latest_season.index = latest_season.Player

latest_season = latest_season[df.columns[:-1]]
latest_season = scaler.transform(latest_season)

proj = pd.DataFrame(model.predict(latest_season),columns=['NeuralNet'])
projections = pd.concat([projections,proj],axis=1)
projections.to_csv(f'{path}\\Data\\ensembleProjections.csv')

print(projections.head())
