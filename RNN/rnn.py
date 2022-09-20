#Recurrent neural network
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Data preprocessing 
#Import training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
traning_set_scaled = sc.fit_transform(training_set)

#Creating a data structure with 60 timesteps and 1 output
X_train, y_train = [], [] 
for i in range(60, 1258):
    X_train.append(traning_set_scaled[i-60:i, 0])
    y_train.append(traning_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping - good for more than one indicator
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Build RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initiliasing RNN
regressor = Sequential()
#add layers
#First
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
#second
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
#third
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
#fourth
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(0.2))
#output
regressor.add(Dense(units=1))
#Compile RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')
#Fit RNN to traning set
regressor.fit(X_train, y_train, epochs=100, batch_size=32)


#Parametere tuning
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
def build_regressor(units, optimizer):
    regressor = Sequential()
    regressor.add(LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=units, return_sequences=True))
    regressor.add(Dropout(0.2)) 
    regressor.add(LSTM(units=units, return_sequences=True))
    regressor.add(Dropout(0.2))    
    regressor.add(LSTM(units=units, return_sequences=False))
    regressor.add(Dropout(0.2))    
    regressor.add(Dense(units=1))
    regressor.compile(optimizer=optimizer, loss='mean_squared_error')
    return regressor
regressor = KerasClassifier(build_fn = build_regressor)
parameters = {
        'units': [45,50,55],
        'batch_size': [30,31,32,33,34,35],
        'epochs': [90,100,110],
        'optimizer': ['adam','rmsprop']        
        }
grid_search = GridSearchCV(estimator=regressor, param_grid=parameters, scoring='neg_mean_squared_error', cv=10)
grid_search = grid_search.fit(X_train, y_train)
best_paramaters = grid_search.best_params_
best_loss = grid_search.best_score_

#Make predict and visualise
#get real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stuck_price = dataset_test.iloc[:, 1:2].values

#predicted stock prices of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs =  sc.transform(inputs)
X_test = [] 
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stuck_price = regressor.predict(X_test)
predicted_stuck_price = sc.inverse_transform(predicted_stuck_price)

#Visualising the results
plt.plot(real_stuck_price, color='red', label='Real Google Stock Prices')
plt.plot(predicted_stuck_price, color='blue', label='Predicted Google Stock Prices')
plt.title('Real vs Predicted')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


