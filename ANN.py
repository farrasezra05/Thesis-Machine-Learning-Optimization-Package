# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 15:17:57 2021

@author: Black
"""


#Artificial Neural Network

PYTHONHASHSEED = 0

#Library Importing 
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as py
from keras.models import Sequential
from keras.layers import Dense
import os
import tensorflow.compat.v1 as tf
import random as rn
from keras import backend as K
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
import pickle
import openpyxl
from openpyxl import load_workbook
import joblib
tf.disable_v2_behavior()

standard_scaler=StandardScaler()
normalizer=Normalizer()
min_max_scaler=MinMaxScaler()

#Direction to Project File
os.chdir(r'C:\Amanah\1-AE\2-BISMILLAH TA\1-TA Ezra\Python\0_ANN_ALL MODEL\31')

#1. Data Preprocessing Process

#Import Dataset values
dataset = pd.read_excel('DOE_Trial.xlsx', sheet_name = 'ANNModel')
X = dataset.iloc[0:,1:13].values
print(X.shape)
print(X)
dfY = dataset.iloc[0:,13:14].values
print(dfY.reshape(-1,1))
Y=min_max_scaler.fit_transform(dfY)

#Check correlation
#totaldata = pd.DataFrame(np.hstack((X, Y)))
#corrMatrix = totaldata.corr()
#py.figure(1)
#sn.heatmap(corrMatrix, annot=True)

DFY = pd.DataFrame(Y)
DFY.to_excel('Normalized Y.xlsx')

#Categorical Data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
trans = ColumnTransformer([('Data',OneHotEncoder(),[0,11])],remainder='passthrough')
X_pre = np.array(trans.fit_transform(X), dtype=np.float64)
X_pre = X_pre[:,:]
print("X_pre: ")
print(X_pre)
DFX = pd.DataFrame(X_pre)
print("DFX: ")
print(DFX)

#Dummy trap prevention
del DFX[DFX.columns[0]]
del DFX[DFX.columns[3]]
#normalizing composite layers orientation
#DFX[DFX.columns[10]] = DFX[DFX.columns[10]]/180
#DFX[DFX.columns[11]] = DFX[DFX.columns[11]]/180
#DFX[DFX.columns[12]] = DFX[DFX.columns[12]]/180
#DFX[DFX.columns[13]] = DFX[DFX.columns[13]]/180
#DFX[DFX.columns[14]] = DFX[DFX.columns[14]]/180
#DFX[DFX.columns[15]] = DFX[DFX.columns[15]]/180

X = np.matrix(DFX)
DFXPre = pd.DataFrame(X_pre)
DFXPre.to_excel('X_pre.xlsx')
pd.DataFrame(X).to_excel('X.xlsx')

#Check correlation 2
#totaldata2 = pd.DataFrame(np.hstack((DFX, Y)))
#corrMatrix2 = totaldata2.corr()
#py.figure(2)
#sn.heatmap(corrMatrix2, annot=True)

#2. Initializing ANN Architecture

#Initialize random seed in order to get the reproducible result
np.random.seed(123)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads = 1, inter_op_parallelism_threads = 1)
tf.set_random_seed(1234)
sess = tf.Session(graph = tf.get_default_graph(), config=session_conf)
K.set_session(sess)

#Initialize the sequence of layers
regressor = Sequential()

# Add input layer and first hidden layer to ANN Architecture
regressor.add(Dense(units=10, activation='linear', input_dim=16))

# Add another layers in ANN Architecture
regressor.add(Dense(units=10, activation='relu'))
regressor.add(Dense(units=10, activation='sigmoid'))
regressor.add(Dense(units=10, activation='relu'))
regressor.add(Dense(units=10, activation='sigmoid'))
regressor.add(Dense(units=10, activation='relu'))

# Add output layer in ANN Arcitecture
regressor.add(Dense(units=1, activation='sigmoid'))
#Compiling ANN Using Sthocastic Gradient Descent

#Compile the model
regressor.compile(optimizer = 'adam',
                  loss = 'mean_squared_error',
                  metrics = ['mse','mae'] )
#tf.keras.optimizers.Adam(learning_rate=0.01)

#3. Fitting ANN to the Optimization Regression Model
#Fitting to training set
history = regressor.fit(X, Y, steps_per_epoch = 300, validation_steps= 200,validation_split = 0.2, batch_size = 5, nb_epoch = 4000, shuffle = True, verbose = 2)
Y_predict = regressor.predict(X)
print(Y_predict)
print(len(DFY))
DFYP = pd.DataFrame(Y_predict)
DFYPNI = pd.concat([DFY, DFYP, pd.DataFrame(((DFY-DFYP).pow(2))/len(DFY))], ignore_index=True, axis = 1)
DFYPNI.to_excel('Input vs Predicted Y_Norm.xlsx')

Y_inverse=min_max_scaler.inverse_transform(Y_predict)

DFYP = pd.DataFrame(Y_inverse)
DFYP.to_excel('Predicted Y.xlsx')

#Model Evaluation

from keras.utils.vis_utils import plot_model
from keras.utils.layer_utils import print_summary
tf.keras.utils.plot_model(regressor, to_file='regressor.png')
model = regressor.get_weights()
print_summary(regressor)
DFmodel = pd.DataFrame(model)

#Pull weights and biases to excel
weightexcel = pd.ExcelWriter('Weight Function.xlsx', engine='openpyxl')
DFmodel.to_excel(weightexcel, sheet_name="All")
weightexcel.save()
for i in range(len(regressor.layers)):
    book = openpyxl.load_workbook('Weight Function.xlsx')
    weightexcel.book = book
    weight = pd.DataFrame(regressor.layers[i].get_weights()[0])
    bias = pd.DataFrame(regressor.layers[i].get_weights()[1])
    weight.to_excel(weightexcel, sheet_name="Weight "+str(i))
    bias.to_excel(weightexcel, sheet_name="Bias "+str(i))
    weightexcel.save()
    print("Wait for Weight Function to be written; "+str(i)+"/"+str(len(regressor.layers)))
weightexcel.close()

filename_pickle = 'model_pickle.h5'
filename_joblib = 'model_joblib.h5'
filename_biasa = 'model_biasa.h5'
openfile_pickle = open(filename_pickle, 'wb')
openfile_joblib = open(filename_joblib, 'wb')
pickle.dump(regressor, openfile_pickle, -1)
joblib.dump(regressor, openfile_joblib)
openfile_pickle.close()
openfile_joblib.close()

model_biasa = regressor.save('model_biasa.h5')

#Visualization Data
#Plot training & validation accuracy
py.figure(1)
py.plot(history.history['loss'], 'magenta')
py.plot(history.history['val_loss'], 'g--')
py.title('Model Loss')
py.ylabel('Loss')
py.xlabel('Epoch')
py.legend(['Train', 'Test'], loc = 'upper right')
py.show()

#Correlation
#corr = pd.DataFrame(corrMatrix)
#corr.to_excel('X & Y input correlation.xlsx')