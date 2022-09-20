#ANN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical independet variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
ct = ColumnTransformer([("Geography", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)
X=X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Fitting ANN classifier to the traning set
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout #Apply on layers and helps in overfitting
#Dropout regularization to reduce overfiting if needed

classifier = Sequential()
#input layer 
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
#add drioout for overfitting in first layer
classifier.add(Dropout(0.1))
#hidden layers
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.1))
#output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid')) # for more than two categories use softmax
#Compile
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, epochs=100)


#Predicy to the test set result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#new user
new_observ = np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
sc.fit(new_observ)
new_prediction = classifier.predict(new_observ) 
new_prediction = (new_prediction > 0.5)


#Making Confusing Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Evaluate model K-cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))    
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid')) # for more than two categories use softmax
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()

import tensorflow as tf
tf.__version__
from tensorflow import keras
keras.__version__
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#imporve ANN
#Gridsearch
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer, output):
    classifier = Sequential()
    classifier.add(Dense(units=output, kernel_initializer='uniform', activation='relu', input_dim=11))    
    classifier.add(Dense(units=output, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid')) # for more than two categories use softmax
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {
        'output': [8],
        'batch_size': [34],
        'epochs': [503],
        'optimizer': ['rmsprop'],
        }
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search = grid_search.fit(X_train, y_train)
best_paramaters = grid_search.best_params_
best_accuracy = grid_search.best_score_



