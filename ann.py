# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 05:58:09 2021

@author: hp
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

geography = pd.get_dummies(x['Geography'],drop_first=True)
gender = pd.get_dummies(x['Gender'],drop_first=True)

x = pd.concat([x,geography, gender], axis=1)
x = x.drop(['Geography', 'Gender'], axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import ReLU, LeakyReLU, PReLU
from keras.layers import Dropout

ann = Sequential()

ann.add(Dense(units=7, activation='relu', kernel_initializer='he_uniform',input_dim=11))

ann.add(Dense(units=6, activation='relu', kernel_initializer='he_uniform'))

ann.add(Dense(units=1, activation='sigmoid', kernel_initializer='glorot_uniform'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model = ann.fit(x_train, y_train, validation_split=0.3, batch_size=10, epochs=100)

y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
dis = ConfusionMatrixDisplay(confusion_matrix = cm)
dis.plot()

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)

# if we are creating a deep neural network then we can add dropout layer to increase accuracy
# we can also add more layers but we should be cautious that it may sometime lead to overfitting
# the main thing is that we need to use the correct combination of parameters to get a higher accuracy
