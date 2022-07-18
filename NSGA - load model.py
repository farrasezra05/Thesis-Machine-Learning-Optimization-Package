#NSGA II Problems Point

PYTHONHASHSEED = 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from keras.models import Sequential
from platypus.algorithms import NSGAII
from platypus.core import Problem
from platypus.types import Real, Integer
from platypus.operators import GAOperator, CompoundOperator, SBX, PM, HUX, BitFlip, PMX
import os
import random as rand
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
import pickle
import joblib
import openpyxl

standard_scaler=StandardScaler()
normalizer=Normalizer()
min_max_scaler=MinMaxScaler()

os.chdir(r'C:\Amanah\1-AE\2-BISMILLAH TA\1-TA Ezra\Python\0_ANN_ALL MODEL\12')
#os.chdir(r'C:\Amanah\ANN')
dataset = pd.read_excel('DOE_Trial.xlsx', sheet_name = 'ANNModel')
X = dataset.iloc[0:,0:13].values
dfY = dataset.iloc[0:,13].values
SEA_base = 3400.84153
CFE_base = 0.398
Pmax_base = 96.623
MCF_base = 38.478
#Baseline = np.array([CFE_base, MCF_base, SEA_base, 120]).reshape((1,4))
Baseline = np.array([SEA_base]).reshape((-1,1))

Y = min_max_scaler.fit_transform(dfY.reshape(-1,1))
Baseline_transformed = min_max_scaler.transform(Baseline)
Baseline_transformed = Baseline_transformed.reshape(-1,1)
#Import Dataset values
#data = pd.read_excel('Regression Model.xlsx', sheet_name = 'Weight Bias')
#del data[data.columns[0]]
#print(data)

#Pulling weight & bias from "Regression Model".
# Pay attention how the excel is structured
#CFEbias = np.array(data.iloc[4,0])
#MCFbias = np.array(data.iloc[4,1])
#SEAbias = np.array(data.iloc[-1,0])
#Pmaxbias = np.array(data.iloc[4,3])
#CFEbias = 1
#MCFbias = 1
#Pmaxbias = 1

#CFEweight = np.array(data.iloc[0,:].values)
#MCFweight = np.array(data.iloc[1,:].values)
#SEAweight = np.array(data.iloc[:16,0].values)
#print("SEAweight shape = "+str(SEAweight.shape))
#Pmaxweight = np.array(data.iloc[3,:].values)
#CFEweight = 1
#MCFweight = 1
#Pmaxweight = 1

i = 0
#load regressor
filename_pickle = 'model_pickle.h5'
filename_joblib = 'model_joblib.h5'
filename_biasa = 'model_biasa.h5'
openfile_pickle = open(filename_pickle, 'rb')
openfile_joblib = open(filename_joblib, 'rb')

#regressor_pickle = pickle.load(openfile_pickle)
#regressor_joblib = joblib.load('model_joblib.h5')
regressor_biasa = tf.keras.models.load_model('model_biasa.h5')

from keras.utils.layer_utils import print_summary
print_summary(regressor_biasa)

def Opt (vars):
#Variable Definition
    d1 = vars[0]
    d2 = vars[1]
    d3 = vars[2]
    d4 = vars[3]
    d5 = vars[4]
    d6 = vars[5]
    spacing = vars[6]
    thickness = vars[7]
    angle = vars[8]
    layer = 2*vars[9]
    o1 = vars[10]
    o2 = vars[11]
    o3 = vars[12]
    o4 = vars[13]
    o5 = vars[14]
    o6 = vars[15]

#Input Variable Matrix Definition
    IntVar = np.array([d1, d2, d3, d4, d5, d6, spacing, thickness,
                       angle, layer, o1, o2, o3, o4, o5, o6])
    IntVar = IntVar.reshape(-1,16)

    #IntVar2 = np.array([d1, d2, d3, d4, d5, d6, spacing, thickness,
                       #angle, layer, o1, o2, o3, o4, o5, o6])

    #IntVars = np.vstack((IntVar, IntVar2))
    print("IntVar shape = "+str(IntVar.shape))
#Function Definition
    #(still using linear activation)
    #CFE = CFEbias + np.matmul(CFEweight,IntVar)
    #MCF = MCFbias + np.matmul(MCFweight,IntVar)
    #SEA = np.add(SEAbias, np.matmul(IntVar,SEAweight))
    #using relu -> pickle load
    SEA = regressor_biasa.predict(IntVar)[0][0]
    print("Problem ke-"+str(i)+", SEA Shape = "+str(SEA.shape))
    print(SEA)
    print(vars)
    #Pmax = Pmaxbias + np.matmul(Pmaxweight,IntVar)
    #Result = [CFE, MCF, SEA, Pmax]

    return [SEA], [SEA,
                   d1, d2, d3, d4, d5, d6, spacing, thickness,
                   angle, layer, o1, o2, o3, o4, o5, o6]

int1=Integer(0,6)

#Case1: Re-entrant GFRP 2 layers
problem1 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem1.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem1.constraints[0:1] = "!=0"
problem1.constraints[1:2] = "==0"
problem1.constraints[2:3] = "==0"
problem1.constraints[3:4] = "==0"
problem1.constraints[4:5] = "==0"
problem1.constraints[5:6] = "==0"
problem1.constraints[6:7] = "==0"
problem1.constraints[7:10] = ">=0"
problem1.constraints[10:11] = "==2"
problem1.constraints[11:12] = ">=1"
problem1.constraints[12:13] = "==0"
problem1.constraints[13:14] = "==0"
problem1.constraints[14:15] = "==0"
problem1.constraints[15:16] = "==0"
problem1.constraints[16:17] = "==0"
problem1.directions[:] = Problem.MAXIMIZE
problem1.function = Opt

#Case2: Re-entrant GFRP 4 layers
problem2 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem2.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem2.constraints[0:1] = ">=0"
problem2.constraints[1:2] = "==0"
problem2.constraints[2:3] = "==0"
problem2.constraints[3:4] = "==0"
problem2.constraints[4:5] = "==0"
problem2.constraints[5:6] = "==0"
problem2.constraints[6:7] = "==0"
problem2.constraints[7:10] = ">=0"
problem2.constraints[10:11] = "==4"
problem2.constraints[11:12] = ">=1"
problem2.constraints[12:13] = ">=1"
problem2.constraints[13:14] = "<=0.001"
problem2.constraints[14:15] = "<=0.001"
problem2.constraints[15:16] = "<=0.001"
problem2.constraints[16:17] = "<=0.001"
problem2.directions[:] = Problem.MAXIMIZE
problem2.function = Opt

#Case3: Re-entrant GFRP 6 layers
problem3 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem3.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem3.constraints[0:1] = ">=0"
problem3.constraints[1:2] = "==0"
problem3.constraints[2:3] = "==0"
problem3.constraints[3:4] = "==0"
problem3.constraints[4:5] = "==0"
problem2.constraints[5:6] = "==0"
problem3.constraints[6:7] = "==0"
problem3.constraints[7:10] = ">=0"
problem2.constraints[10:11] = "==4"
problem3.constraints[11:12] = ">=1"
problem3.constraints[12:13] = ">=1"
problem3.constraints[13:14] = ">=1"
problem3.constraints[14:15] = "<=0.001"
problem2.constraints[15:16] = "<=0.001"
problem3.constraints[16:17] = "<=0.001"
problem3.directions[:] = Problem.MAXIMIZE
problem3.function = Opt

#Case4: Re-entrant GFRP 8 layers
problem4 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem4.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem4.constraints[0:1] = ">=0"
problem4.constraints[1:2] = "==0"
problem4.constraints[2:3] = "==0"
problem4.constraints[3:4] = "==0"
problem4.constraints[4:5] = "==0"
problem4.constraints[5:6] = "==0"
problem4.constraints[6:7] = "==0"
problem4.constraints[7:10] = ">=0"
problem4.constraints[10:11] = "==8"
problem4.constraints[11:12] = ">=1"
problem4.constraints[12:13] = ">=1"
problem4.constraints[13:14] = ">=1"
problem4.constraints[14:15] = ">=1"
problem4.constraints[15:16] = "<=0.001"
problem4.constraints[16:17] = "<=0.001"
problem4.directions[:] = Problem.MAXIMIZE
problem4.function = Opt

#Case5: Re-entrant GFRP 10 layers
problem5 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem5.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem5.constraints[0:1] = ">=0"
problem5.constraints[1:2] = "==0"
problem5.constraints[2:3] = "==0"
problem5.constraints[3:4] = "==0"
problem5.constraints[4:5] = "==0"
problem5.constraints[5:6] = "==0"
problem5.constraints[6:7] = "==0"
problem5.constraints[7:10] = ">=0"
problem5.constraints[10:11] = "==10"
problem5.constraints[11:12] = ">=1"
problem5.constraints[12:13] = ">=1"
problem5.constraints[13:14] = ">=1"
problem5.constraints[14:15] = ">=1"
problem5.constraints[15:16] = ">=1"
problem5.constraints[16:17] = "<=0.001"
problem5.directions[:] = Problem.MAXIMIZE
problem5.function = Opt

#Case6: Re-entrant GFRP 12 layers
problem6 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem6.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem6.constraints[0:1] = ">=0"
problem6.constraints[1:2] = "==0"
problem6.constraints[2:3] = "==0"
problem6.constraints[3:4] = "==0"
problem6.constraints[4:5] = "==0"
problem6.constraints[5:6] = "==0"
problem6.constraints[6:7] = "==0"
problem6.constraints[7:10] = ">=0"
problem6.constraints[10:11] = "==12"
problem6.constraints[11:12] = ">=1"
problem6.constraints[12:13] = ">=1"
problem6.constraints[13:14] = ">=1"
problem6.constraints[14:15] = ">=1"
problem6.constraints[15:16] = ">=1"
problem6.constraints[16:17] = ">=1"
problem6.directions[:] = Problem.MAXIMIZE
problem6.function = Opt

#Case7: Re-entrant CFRP 2 layers
problem7 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem7.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem7.constraints[0:1] = ">=0"
problem7.constraints[1:2] = "==0"
problem7.constraints[2:3] = "==0"
problem7.constraints[3:4] = "==0"
problem7.constraints[4:5] = "==1"
problem7.constraints[5:6] = "==0"
problem7.constraints[6:7] = "==0"
problem7.constraints[7:10] = ">=0"
problem7.constraints[10:11] = "==2"
problem7.constraints[11:12] = ">=1"
problem7.constraints[12:13] = "<=0.001"
problem7.constraints[13:14] = "<=0.001"
problem7.constraints[14:15] = "<=0.001"
problem7.constraints[15:16] = "<=0.001"
problem7.constraints[16:17] = "<=0.001"
problem7.directions[:] = Problem.MAXIMIZE
problem7.function = Opt

#Case8: Re-entrant CFRP 4 layers
problem8 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem8.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem8.constraints[0:1] = ">=0"
problem8.constraints[1:2] = "==0"
problem8.constraints[2:3] = "==0"
problem8.constraints[3:4] = "==0"
problem8.constraints[4:5] = "==1"
problem8.constraints[5:6] = "==0"
problem8.constraints[6:7] = "==0"
problem8.constraints[7:10] = ">=0"
problem8.constraints[10:11] = "==4"
problem8.constraints[11:12] = ">=1"
problem8.constraints[12:13] = ">=1"
problem8.constraints[13:14] = "<=0.001"
problem8.constraints[14:15] = "<=0.001"
problem8.constraints[15:16] = "<=0.001"
problem8.constraints[16:17] = "<=0.001"
problem8.directions[:] = Problem.MAXIMIZE
problem8.function = Opt

#Case9: Re-entrant CFRP 6 layers
problem9 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem9.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem9.constraints[0:1] = ">=0"
problem9.constraints[1:2] = "==0"
problem9.constraints[2:3] = "==0"
problem9.constraints[3:4] = "==0"
problem9.constraints[4:5] = "==1"
problem9.constraints[5:6] = "==0"
problem9.constraints[6:7] = "==0"
problem9.constraints[7:10] = ">=0"
problem9.constraints[10:11] = "==6"
problem9.constraints[11:12] = ">=1"
problem9.constraints[12:13] = ">=1"
problem9.constraints[13:14] = ">=1"
problem9.constraints[14:15] = "<=0.001"
problem9.constraints[15:16] = "<=0.001"
problem9.constraints[16:17] = "<=0.001"
problem9.directions[:] = Problem.MAXIMIZE
problem9.function = Opt

#Case10: Re-entrant CFRP 8 layers
problem10 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem10.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem10.constraints[0:1] = ">=0"
problem10.constraints[1:2] = "==0"
problem10.constraints[2:3] = "==0"
problem10.constraints[3:4] = "==0"
problem10.constraints[4:5] = "==1"
problem10.constraints[5:6] = "==0"
problem10.constraints[6:7] = "==0"
problem10.constraints[7:10] = ">=0"
problem10.constraints[10:11] = "==8"
problem10.constraints[11:12] = ">=1"
problem10.constraints[12:13] = ">=1"
problem10.constraints[13:14] = ">=1"
problem10.constraints[14:15] = ">=1"
problem10.constraints[15:16] = "<=0.001"
problem10.constraints[16:17] = "<=0.001"
problem10.directions[:] = Problem.MAXIMIZE
problem10.function = Opt

#Case11: Re-entrant CFRP 10 layers
problem11 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem11.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem11.constraints[0:1] = ">=0"
problem11.constraints[1:2] = "==0"
problem11.constraints[2:3] = "==0"
problem11.constraints[3:4] = "==0"
problem11.constraints[4:5] = "==1"
problem11.constraints[5:6] = "==0"
problem11.constraints[6:7] = "==0"
problem11.constraints[7:10] = ">=0"
problem11.constraints[10:11] = "==10"
problem11.constraints[11:12] = ">=1"
problem11.constraints[12:13] = ">=1"
problem11.constraints[13:14] = ">=1"
problem11.constraints[14:15] = ">=1"
problem11.constraints[15:16] = ">=1"
problem11.constraints[16:17] = "<=0.001"
problem11.directions[:] = Problem.MAXIMIZE
problem11.function = Opt

#Case12: Re-entrant CFRP 12 layers
problem12 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem12.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem12.constraints[0:1] = ">=0"
problem12.constraints[1:2] = "==0"
problem12.constraints[2:3] = "==0"
problem12.constraints[3:4] = "==0"
problem12.constraints[4:5] = "==1"
problem12.constraints[5:6] = "==0"
problem12.constraints[6:7] = "==0"
problem12.constraints[7:10] = ">=0"
problem12.constraints[10:11] = "==12"
problem12.constraints[11:12] = ">=1"
problem12.constraints[12:13] = ">=1"
problem12.constraints[13:14] = ">=1"
problem12.constraints[14:15] = ">=1"
problem12.constraints[15:16] = ">=1"
problem12.constraints[16:17] = ">=1"
problem12.directions[:] = Problem.MAXIMIZE
problem12.function = Opt

#Case13: Re-entrant CS 0 layers
problem13 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem13.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem13.constraints[0:1] = ">=0"
problem13.constraints[1:2] = "==0"
problem13.constraints[2:3] = "==0"
problem13.constraints[3:4] = "==0"
problem13.constraints[4:5] = "==0"
problem13.constraints[5:6] = "==1"
problem13.constraints[6:7] = "==0"
problem13.constraints[7:10] = ">=0"
problem13.constraints[10:11] = "==0"
problem13.constraints[11:12] = "<=0.001"
problem13.constraints[12:13] = "<=0.001"
problem13.constraints[13:14] = "<=0.001"
problem13.constraints[14:15] = "<=0.001"
problem13.constraints[15:16] = "<=0.001"
problem13.constraints[16:17] = "<=0.001"
problem13.directions[:] = Problem.MAXIMIZE
problem13.function = Opt

#Case14: Re-entrant AL 0 layers
problem14 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem14.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem14.constraints[0:1] = ">=0"
problem14.constraints[1:2] = "==0"
problem14.constraints[2:3] = "==0"
problem14.constraints[3:4] = "==0"
problem14.constraints[4:5] = "==0"
problem14.constraints[5:6] = "==0"
problem14.constraints[6:7] = "==1"
problem14.constraints[7:10] = ">=0"
problem14.constraints[10:11] = "==0"
problem14.constraints[11:12] = "<=0.001"
problem14.constraints[12:13] = "<=0.001"
problem14.constraints[13:14] = "<=0.001"
problem14.constraints[14:15] = "<=0.001"
problem14.constraints[15:16] = "<=0.001"
problem14.constraints[16:17] = "<=0.001"
problem14.directions[:] = Problem.MAXIMIZE
problem14.function = Opt

#Case15: DAH GFRP 2 layers
problem15 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem15.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem15.constraints[0:1] = ">=0"
problem15.constraints[1:2] = "==1"
problem15.constraints[2:3] = "==0"
problem15.constraints[3:4] = "==0"
problem15.constraints[4:5] = "==0"
problem15.constraints[5:6] = "==0"
problem15.constraints[6:7] = "==0"
problem15.constraints[7:10] = ">=0"
problem15.constraints[10:11] = "==2"
problem15.constraints[11:12] = ">=1"
problem15.constraints[12:13] = "<=0.001"
problem15.constraints[13:14] = "<=0.001"
problem15.constraints[14:15] = "<=0.001"
problem15.constraints[15:16] = "<=0.001"
problem15.constraints[16:17] = "<=0.001"
problem15.directions[:] = Problem.MAXIMIZE
problem15.function = Opt

#Case16: DAH GFRP 4 layers
problem16 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem16.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem16.constraints[0:1] = ">=0"
problem16.constraints[1:2] = "==1"
problem16.constraints[2:3] = "==0"
problem16.constraints[3:4] = "==0"
problem16.constraints[4:5] = "==0"
problem16.constraints[5:6] = "==0"
problem16.constraints[6:7] = "==0"
problem16.constraints[7:10] = ">=0"
problem16.constraints[10:11] = "==4"
problem16.constraints[11:12] = ">=1"
problem16.constraints[12:13] = ">=1"
problem16.constraints[13:14] = "<=0.001"
problem16.constraints[14:15] = "<=0.001"
problem16.constraints[15:16] = "<=0.001"
problem16.constraints[16:17] = "<=0.001"
problem16.directions[:] = Problem.MAXIMIZE
problem16.function = Opt
#Case17: DAH GFRP 6 layers
problem17 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem17.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem17.constraints[0:1] = ">=0"
problem17.constraints[1:2] = "==1"
problem17.constraints[2:3] = "==0"
problem17.constraints[3:4] = "==0"
problem17.constraints[4:5] = "==0"
problem17.constraints[5:6] = "==0"
problem17.constraints[6:7] = "==0"
problem17.constraints[7:10] = ">=0"
problem17.constraints[10:11] = "==6"
problem17.constraints[11:12] = ">=1"
problem17.constraints[12:13] = ">=1"
problem17.constraints[13:14] = ">=1"
problem17.constraints[14:15] = "<=0.001"
problem17.constraints[15:16] = "<=0.001"
problem17.constraints[16:17] = "<=0.001"
problem17.directions[:] = Problem.MAXIMIZE
problem17.function = Opt

#Case18: DAH GFRP 8 layers
problem18 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem18.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem18.constraints[0:1] = ">=0"
problem18.constraints[1:2] = "==1"
problem18.constraints[2:3] = "==0"
problem18.constraints[3:4] = "==0"
problem18.constraints[4:5] = "==0"
problem18.constraints[5:6] = "==0"
problem18.constraints[6:7] = "==0"
problem18.constraints[7:10] = ">=0"
problem18.constraints[10:11] = "==8"
problem18.constraints[11:12] = ">=1"
problem18.constraints[12:13] = ">=1"
problem18.constraints[13:14] = ">=1"
problem18.constraints[14:15] = ">=1"
problem18.constraints[15:16] = "<=0.001"
problem18.constraints[16:17] = "<=0.001"
problem18.directions[:] = Problem.MAXIMIZE
problem18.function = Opt

#Case19: DAH GFRP 10 layers
problem19 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem19.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem19.constraints[0:1] = ">=0"
problem19.constraints[1:2] = "==1"
problem19.constraints[2:3] = "==0"
problem19.constraints[3:4] = "==0"
problem19.constraints[4:5] = "==0"
problem19.constraints[5:6] = "==0"
problem19.constraints[6:7] = "==0"
problem19.constraints[7:10] = ">=0"
problem19.constraints[10:11] = "==10"
problem19.constraints[11:12] = ">=1"
problem19.constraints[12:13] = ">=1"
problem19.constraints[13:14] = ">=1"
problem19.constraints[14:15] = ">=1"
problem19.constraints[15:16] = ">=1"
problem19.constraints[16:17] = "<=0.001"
problem19.directions[:] = Problem.MAXIMIZE
problem19.function = Opt

#Case20: DAH GFRP 12 layers
problem20 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem20.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem20.constraints[0:1] = ">=0"
problem20.constraints[1:2] = "==1"
problem20.constraints[2:3] = "==0"
problem20.constraints[3:4] = "==0"
problem20.constraints[4:5] = "==0"
problem20.constraints[5:6] = "==0"
problem20.constraints[6:7] = "==0"
problem20.constraints[7:10] = ">=0"
problem20.constraints[10:11] = "==12"
problem20.constraints[11:12] = ">=1"
problem20.constraints[12:13] = ">=1"
problem20.constraints[13:14] = ">=1"
problem20.constraints[14:15] = ">=1"
problem20.constraints[15:16] = ">=1"
problem20.constraints[16:17] = ">=1"
problem20.directions[:] = Problem.MAXIMIZE
problem20.function = Opt

#Case21: DAH CFRP 2 layers
problem21 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem21.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem21.constraints[0:1] = ">=0"
problem21.constraints[1:2] = "==1"
problem21.constraints[2:3] = "==0"
problem21.constraints[3:4] = "==0"
problem21.constraints[4:5] = "==1"
problem21.constraints[5:6] = "==0"
problem21.constraints[6:7] = "==0"
problem21.constraints[7:10] = ">=0"
problem21.constraints[10:11] = "==2"
problem21.constraints[11:12] = ">=1"
problem21.constraints[12:13] = "<=0.001"
problem21.constraints[13:14] = "<=0.001"
problem21.constraints[14:15] = "<=0.001"
problem21.constraints[15:16] = "<=0.001"
problem21.constraints[16:17] = "<=0.001"
problem21.directions[:] = Problem.MAXIMIZE
problem21.function = Opt

#Case22: DAH CFRP 4 layers
problem22 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem22.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem22.constraints[0:1] = ">=0"
problem22.constraints[1:2] = "==1"
problem22.constraints[2:3] = "==0"
problem22.constraints[3:4] = "==0"
problem22.constraints[4:5] = "==1"
problem22.constraints[5:6] = "==0"
problem22.constraints[6:7] = "==0"
problem22.constraints[7:10] = ">=0"
problem22.constraints[10:11] = "==4"
problem22.constraints[11:12] = ">=1"
problem22.constraints[12:13] = ">=1"
problem22.constraints[13:14] = "<=0.001"
problem22.constraints[14:15] = "<=0.001"
problem22.constraints[15:16] = "<=0.001"
problem22.constraints[16:17] = "<=0.001"
problem22.directions[:] = Problem.MAXIMIZE
problem22.function = Opt

#Case23: DAH CFRP 6 layers
problem23 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem23.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem23.constraints[0:1] = ">=0"
problem23.constraints[1:2] = "==1"
problem23.constraints[2:3] = "==0"
problem23.constraints[3:4] = "==0"
problem23.constraints[4:5] = "==1"
problem23.constraints[5:6] = "==0"
problem23.constraints[6:7] = "==0"
problem23.constraints[7:10] = ">=0"
problem23.constraints[10:11] = "==6"
problem23.constraints[11:12] = ">=1"
problem23.constraints[12:13] = ">=1"
problem23.constraints[13:14] = ">=1"
problem23.constraints[14:15] = "<=0.001"
problem23.constraints[15:16] = "<=0.001"
problem23.constraints[16:17] = "<=0.001"
problem23.directions[:] = Problem.MAXIMIZE
problem23.function = Opt

#Case24: DAH CFRP 8 layers
problem24 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem24.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem24.constraints[0:1] = ">=0"
problem24.constraints[1:2] = "==1"
problem24.constraints[2:3] = "==0"
problem24.constraints[3:4] = "==0"
problem24.constraints[4:5] = "==1"
problem24.constraints[5:6] = "==0"
problem24.constraints[6:7] = "==0"
problem24.constraints[7:10] = ">=0"
problem24.constraints[10:11] = "==8"
problem24.constraints[11:12] = ">=1"
problem24.constraints[12:13] = ">=1"
problem24.constraints[13:14] = ">=1"
problem24.constraints[14:15] = ">=1"
problem24.constraints[15:16] = "<=0.001"
problem24.constraints[16:17] = "<=0.001"
problem24.directions[:] = Problem.MAXIMIZE
problem24.function = Opt

#Case25: DAH CFRP 10 layers
problem25 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem25.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem25.constraints[0:1] = ">=0"
problem25.constraints[1:2] = "==1"
problem25.constraints[2:3] = "==0"
problem25.constraints[3:4] = "==0"
problem25.constraints[4:5] = "==1"
problem25.constraints[5:6] = "==0"
problem25.constraints[6:7] = "==0"
problem25.constraints[7:10] = ">=0"
problem25.constraints[10:11] = "==10"
problem25.constraints[11:12] = ">=1"
problem25.constraints[12:13] = ">=1"
problem25.constraints[13:14] = ">=1"
problem25.constraints[14:15] = ">=1"
problem25.constraints[15:16] = ">=1"
problem25.constraints[16:17] = "<=0.001"
problem25.directions[:] = Problem.MAXIMIZE
problem25.function = Opt

#Case26: DAH CFRP 12 layers
problem26 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem26.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem26.constraints[0:1] = ">=0"
problem26.constraints[1:2] = "==1"
problem26.constraints[2:3] = "==0"
problem26.constraints[3:4] = "==0"
problem26.constraints[4:5] = "==1"
problem26.constraints[5:6] = "==0"
problem26.constraints[6:7] = "==0"
problem26.constraints[7:10] = ">=0"
problem26.constraints[10:11] = "==12"
problem26.constraints[11:12] = ">=1"
problem26.constraints[12:13] = ">=1"
problem26.constraints[13:14] = ">=1"
problem26.constraints[14:15] = ">=1"
problem26.constraints[15:16] = ">=1"
problem26.constraints[16:17] = ">=1"
problem26.directions[:] = Problem.MAXIMIZE
problem26.function = Opt

#Case27: DAH CS 0 layers
problem27 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem27.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem27.constraints[0:1] = ">=0"
problem27.constraints[1:2] = "==1"
problem27.constraints[2:3] = "==0"
problem27.constraints[3:4] = "==0"
problem27.constraints[4:5] = "==0"
problem27.constraints[5:6] = "==1"
problem27.constraints[6:7] = "==0"
problem27.constraints[7:10] = ">=0"
problem27.constraints[10:11] = "==0"
problem27.constraints[11:12] = "<=0.001"
problem27.constraints[12:13] = "<=0.001"
problem27.constraints[13:14] = "<=0.001"
problem27.constraints[14:15] = "<=0.001"
problem27.constraints[15:16] = "<=0.001"
problem27.constraints[16:17] = "<=0.001"
problem27.directions[:] = Problem.MAXIMIZE
problem27.function = Opt

#Case28: DAH AL 0 layers
problem28 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem28.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem28.constraints[0:1] = ">=0"
problem28.constraints[1:2] = "==1"
problem28.constraints[2:3] = "==0"
problem28.constraints[3:4] = "==0"
problem28.constraints[4:5] = "==0"
problem28.constraints[5:6] = "==0"
problem28.constraints[6:7] = "==1"
problem28.constraints[7:10] = ">=0"
problem28.constraints[10:11] = "==0"
problem28.constraints[11:12] = "<=0.001"
problem28.constraints[12:13] = "<=0.001"
problem28.constraints[13:14] = "<=0.001"
problem28.constraints[14:15] = "<=0.001"
problem28.constraints[15:16] = "<=0.001"
problem28.constraints[16:17] = "<=0.001"
problem28.directions[:] = Problem.MAXIMIZE
problem28.function = Opt

#Case29: Star GFRP 2 layers
problem29 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem29.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem29.constraints[0:1] = ">=0"
problem29.constraints[1:2] = "==0"
problem29.constraints[2:3] = "==1"
problem29.constraints[3:4] = "==0"
problem29.constraints[4:5] = "==0"
problem29.constraints[5:6] = "==0"
problem29.constraints[6:7] = "==0"
problem29.constraints[7:10] = ">=0"
problem29.constraints[10:11] = "==2"
problem29.constraints[11:12] = ">=1"
problem29.constraints[12:13] = "<=0.001"
problem29.constraints[13:14] = "<=0.001"
problem29.constraints[14:15] = "<=0.001"
problem29.constraints[15:16] = "<=0.001"
problem29.constraints[16:17] = "<=0.001"
problem29.directions[:] = Problem.MAXIMIZE
problem29.function = Opt

#Case30: Star GFRP 4 layers
problem30 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem30.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem30.constraints[0:1] = ">=0"
problem30.constraints[1:2] = "==0"
problem30.constraints[2:3] = "==1"
problem30.constraints[3:4] = "==0"
problem30.constraints[4:5] = "==0"
problem30.constraints[5:6] = "==0"
problem30.constraints[6:7] = "==0"
problem30.constraints[7:10] = ">=0"
problem30.constraints[10:11] = "==4"
problem30.constraints[11:12] = ">=1"
problem30.constraints[12:13] = ">=1"
problem30.constraints[13:14] = "<=0.001"
problem30.constraints[14:15] = "<=0.001"
problem30.constraints[15:16] = "<=0.001"
problem30.constraints[16:17] = "<=0.001"
problem30.directions[:] = Problem.MAXIMIZE
problem30.function = Opt

#Case31: Star GFRP 6 layers
problem31 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem31.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem31.constraints[0:1] = ">=0"
problem31.constraints[1:2] = "==0"
problem31.constraints[2:3] = "==1"
problem31.constraints[3:4] = "==0"
problem31.constraints[4:5] = "==0"
problem31.constraints[5:6] = "==0"
problem31.constraints[6:7] = "==0"
problem31.constraints[7:10] = ">=0"
problem31.constraints[10:11] = "==6"
problem31.constraints[11:12] = ">=1"
problem31.constraints[12:13] = ">=1"
problem31.constraints[13:14] = ">=1"
problem31.constraints[14:15] = "<=0.001"
problem31.constraints[15:16] = "<=0.001"
problem31.constraints[16:17] = "<=0.001"
problem31.directions[:] = Problem.MAXIMIZE
problem31.function = Opt

#Case32: Star GFRP 8 layers
problem32 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem32.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem32.constraints[0:1] = ">=0"
problem32.constraints[1:2] = "==0"
problem32.constraints[2:3] = "==1"
problem32.constraints[3:4] = "==0"
problem32.constraints[4:5] = "==0"
problem32.constraints[5:6] = "==0"
problem32.constraints[6:7] = "==0"
problem32.constraints[7:10] = ">=0"
problem32.constraints[10:11] = "==8"
problem32.constraints[11:12] = ">=1"
problem32.constraints[12:13] = ">=1"
problem32.constraints[13:14] = ">=1"
problem32.constraints[14:15] = ">=1"
problem32.constraints[15:16] = "<=0.001"
problem32.constraints[16:17] = "<=0.001"
problem32.directions[:] = Problem.MAXIMIZE
problem32.function = Opt

#Case33: Star GFRP 10 layers
problem33 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem33.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem33.constraints[0:1] = ">=0"
problem33.constraints[1:2] = "==0"
problem33.constraints[2:3] = "==1"
problem33.constraints[3:4] = "==0"
problem33.constraints[4:5] = "==0"
problem33.constraints[5:6] = "==0"
problem33.constraints[6:7] = "==0"
problem33.constraints[7:10] = ">=0"
problem33.constraints[10:11] = "==10"
problem33.constraints[11:12] = ">=1"
problem33.constraints[12:13] = ">=1"
problem33.constraints[13:14] = ">=1"
problem33.constraints[14:15] = ">=1"
problem33.constraints[15:16] = ">=1"
problem33.constraints[16:17] = "<=0.001"
problem33.directions[:] = Problem.MAXIMIZE
problem33.function = Opt

#Case34: Star GFRP 12 layers
problem34 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem34.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem34.constraints[0:1] = ">=0"
problem34.constraints[1:2] = "==0"
problem34.constraints[2:3] = "==1"
problem34.constraints[3:4] = "==0"
problem34.constraints[4:5] = "==0"
problem34.constraints[5:6] = "==0"
problem34.constraints[6:7] = "==0"
problem34.constraints[7:10] = ">=0"
problem34.constraints[10:11] = "==12"
problem34.constraints[11:12] = ">=1"
problem34.constraints[12:13] = ">=1"
problem34.constraints[13:14] = ">=1"
problem34.constraints[14:15] = ">=1"
problem34.constraints[15:16] = ">=1"
problem34.constraints[16:17] = ">=1"
problem34.directions[:] = Problem.MAXIMIZE
problem34.function = Opt

#Case35: Star CFRP 2 layers
problem35 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem35.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem35.constraints[0:1] = ">=0"
problem35.constraints[1:2] = "==0"
problem35.constraints[2:3] = "==1"
problem35.constraints[3:4] = "==0"
problem35.constraints[4:5] = "==1"
problem35.constraints[5:6] = "==0"
problem35.constraints[6:7] = "==0"
problem35.constraints[7:10] = ">=0"
problem35.constraints[10:11] = "==2"
problem35.constraints[11:12] = ">=1"
problem35.constraints[12:13] = "<=0.001"
problem35.constraints[13:14] = "<=0.001"
problem35.constraints[14:15] = "<=0.001"
problem35.constraints[15:16] = "<=0.001"
problem35.constraints[16:17] = "<=0.001"
problem35.directions[:] = Problem.MAXIMIZE
problem35.function = Opt

#Case36: Star CFRP 4 layers
problem36 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem36.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem36.constraints[0:1] = ">=0"
problem36.constraints[1:2] = "==0"
problem36.constraints[2:3] = "==1"
problem36.constraints[3:4] = "==0"
problem36.constraints[4:5] = "==1"
problem36.constraints[5:6] = "==0"
problem36.constraints[6:7] = "==0"
problem36.constraints[7:10] = ">=0"
problem36.constraints[10:11] = "==4"
problem36.constraints[11:12] = ">=1"
problem36.constraints[12:13] = ">=1"
problem36.constraints[13:14] = "<=0.001"
problem36.constraints[14:15] = "<=0.001"
problem36.constraints[15:16] = "<=0.001"
problem36.constraints[16:17] = "<=0.001"
problem36.directions[:] = Problem.MAXIMIZE
problem36.function = Opt

#Case37: Star CFRP 6 layers
problem37 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem37.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem37.constraints[0:1] = ">=0"
problem37.constraints[1:2] = "==0"
problem37.constraints[2:3] = "==1"
problem37.constraints[3:4] = "==0"
problem37.constraints[4:5] = "==1"
problem37.constraints[5:6] = "==0"
problem37.constraints[6:7] = "==0"
problem37.constraints[7:10] = ">=0"
problem37.constraints[10:11] = "==6"
problem37.constraints[11:12] = ">=1"
problem37.constraints[12:13] = ">=1"
problem37.constraints[13:14] = ">=1"
problem37.constraints[14:15] = "<=0.001"
problem37.constraints[15:16] = "<=0.001"
problem37.constraints[16:17] = "<=0.001"
problem37.directions[:] = Problem.MAXIMIZE
problem37.function = Opt

#Case38: Star CFRP 8 layers
problem38 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem38.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem38.constraints[0:1] = ">=0"
problem38.constraints[1:2] = "==0"
problem38.constraints[2:3] = "==1"
problem38.constraints[3:4] = "==0"
problem38.constraints[4:5] = "==1"
problem38.constraints[5:6] = "==0"
problem38.constraints[6:7] = "==0"
problem38.constraints[7:10] = ">=0"
problem38.constraints[10:11] = "==8"
problem38.constraints[11:12] = ">=1"
problem38.constraints[12:13] = ">=1"
problem38.constraints[13:14] = ">=1"
problem38.constraints[14:15] = ">=1"
problem38.constraints[15:16] = "<=0.001"
problem38.constraints[16:17] = "<=0.001"
problem38.directions[:] = Problem.MAXIMIZE
problem38.function = Opt

#Case39: Star CFRP 10 layers
problem39 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem39.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem39.constraints[0:1] = ">=0"
problem39.constraints[1:2] = "==0"
problem39.constraints[2:3] = "==1"
problem39.constraints[3:4] = "==0"
problem39.constraints[4:5] = "==1"
problem39.constraints[5:6] = "==0"
problem39.constraints[6:7] = "==0"
problem39.constraints[7:10] = ">=0"
problem39.constraints[10:11] = "==10"
problem39.constraints[11:12] = ">=1"
problem39.constraints[12:13] = ">=1"
problem39.constraints[13:14] = ">=1"
problem39.constraints[14:15] = ">=1"
problem39.constraints[15:16] = ">=1"
problem39.constraints[16:17] = "<=0.001"
problem39.directions[:] = Problem.MAXIMIZE
problem39.function = Opt

#Case40: Star CFRP 12 layers
problem40 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem40.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem40.constraints[0:1] = ">=0"
problem40.constraints[1:2] = "==0"
problem40.constraints[2:3] = "==1"
problem40.constraints[3:4] = "==0"
problem40.constraints[4:5] = "==1"
problem40.constraints[5:6] = "==0"
problem40.constraints[6:7] = "==0"
problem40.constraints[7:10] = ">=0"
problem40.constraints[10:11] = "==12"
problem40.constraints[11:12] = ">=1"
problem40.constraints[12:13] = ">=1"
problem40.constraints[13:14] = ">=1"
problem40.constraints[14:15] = ">=1"
problem40.constraints[15:16] = ">=1"
problem40.constraints[16:17] = ">=1"
problem40.directions[:] = Problem.MAXIMIZE
problem40.function = Opt

#Case41: Star CS 0 layers
problem41 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem41.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem41.constraints[0:1] = ">=0"
problem41.constraints[1:2] = "==0"
problem41.constraints[2:3] = "==1"
problem41.constraints[3:4] = "==0"
problem41.constraints[4:5] = "==0"
problem41.constraints[5:6] = "==1"
problem41.constraints[6:7] = "==0"
problem41.constraints[7:10] = ">=0"
problem41.constraints[10:11] = "==0"
problem41.constraints[11:12] = "<=0.001"
problem41.constraints[12:13] = "<=0.001"
problem41.constraints[13:14] = "<=0.001"
problem41.constraints[14:15] = "<=0.001"
problem41.constraints[15:16] = "<=0.001"
problem41.constraints[16:17] = "<=0.001"
problem41.directions[:] = Problem.MAXIMIZE
problem41.function = Opt

#Case42: Star AL 0 layers
problem42 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem42.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem42.constraints[0:1] = ">=0"
problem42.constraints[1:2] = "==0"
problem42.constraints[2:3] = "==1"
problem42.constraints[3:4] = "==0"
problem42.constraints[4:5] = "==0"
problem42.constraints[5:6] = "==0"
problem42.constraints[6:7] = "==1"
problem42.constraints[7:10] = ">=0"
problem42.constraints[10:11] = "==0"
problem42.constraints[11:12] = "<=0.001"
problem42.constraints[12:13] = "<=0.001"
problem42.constraints[13:14] = "<=0.001"
problem42.constraints[14:15] = "<=0.001"
problem42.constraints[15:16] = "<=0.001"
problem42.constraints[16:17] = "<=0.001"
problem42.directions[:] = Problem.MAXIMIZE
problem42.function = Opt

#Case43: DUH GFRP 2 layers
problem43 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem43.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem43.constraints[0:1] = ">=0"
problem43.constraints[1:2] = "==0"
problem43.constraints[2:3] = "==0"
problem43.constraints[3:4] = "==1"
problem43.constraints[4:5] = "==0"
problem43.constraints[5:6] = "==0"
problem43.constraints[6:7] = "==0"
problem43.constraints[7:10] = ">=0"
problem43.constraints[10:11] = "==2"
problem43.constraints[11:12] = ">=1"
problem43.constraints[12:13] = "<=0.001"
problem43.constraints[13:14] = "<=0.001"
problem43.constraints[14:15] = "<=0.001"
problem43.constraints[15:16] = "<=0.001"
problem43.constraints[16:17] = "<=0.001"
problem43.directions[:] = Problem.MAXIMIZE
problem43.function = Opt

#Case44: DUH GFRP 4 layers
problem44 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem44.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem44.constraints[0:1] = ">=0"
problem44.constraints[1:2] = "==0"
problem44.constraints[2:3] = "==0"
problem44.constraints[3:4] = "==1"
problem44.constraints[4:5] = "==0"
problem44.constraints[5:6] = "==0"
problem44.constraints[6:7] = "==0"
problem44.constraints[7:10] = ">=0"
problem44.constraints[10:11] = "==4"
problem44.constraints[11:12] = ">=1"
problem44.constraints[12:13] = ">=1"
problem44.constraints[13:14] = "<=0.001"
problem44.constraints[14:15] = "<=0.001"
problem44.constraints[15:16] = "<=0.001"
problem44.constraints[16:17] = "<=0.001"
problem44.directions[:] = Problem.MAXIMIZE
problem44.function = Opt

#Case45: DUH GFRP 6 layers
problem45 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem45.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem45.constraints[0:1] = ">=0"
problem45.constraints[1:2] = "==0"
problem45.constraints[2:3] = "==0"
problem45.constraints[3:4] = "==1"
problem45.constraints[4:5] = "==0"
problem45.constraints[5:6] = "==0"
problem45.constraints[6:7] = "==0"
problem45.constraints[7:10] = ">=0"
problem45.constraints[10:11] = "==6"
problem45.constraints[11:12] = ">=1"
problem45.constraints[12:13] = ">=1"
problem45.constraints[13:14] = ">=1"
problem45.constraints[14:15] = "<=0.001"
problem45.constraints[15:16] = "<=0.001"
problem45.constraints[16:17] = "<=0.001"
problem45.directions[:] = Problem.MAXIMIZE
problem45.function = Opt

#Case46: DUH GFRP 8 layers
problem46 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem46.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem46.constraints[0:1] = ">=0"
problem46.constraints[1:2] = "==0"
problem46.constraints[2:3] = "==0"
problem46.constraints[3:4] = "==1"
problem46.constraints[4:5] = "==0"
problem46.constraints[5:6] = "==0"
problem46.constraints[6:7] = "==0"
problem46.constraints[7:10] = ">=0"
problem46.constraints[10:11] = "==8"
problem46.constraints[11:12] = ">=1"
problem46.constraints[12:13] = ">=1"
problem46.constraints[13:14] = ">=1"
problem46.constraints[14:15] = ">=1"
problem46.constraints[15:16] = "<=0.001"
problem46.constraints[16:17] = "<=0.001"
problem46.directions[:] = Problem.MAXIMIZE
problem46.function = Opt

#Case47: DUH GFRP 10 layers
problem47 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem47.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem47.constraints[0:1] = ">=0"
problem47.constraints[1:2] = "==0"
problem47.constraints[2:3] = "==0"
problem47.constraints[3:4] = "==1"
problem47.constraints[4:5] = "==0"
problem47.constraints[5:6] = "==0"
problem47.constraints[6:7] = "==0"
problem47.constraints[7:10] = ">=0"
problem47.constraints[10:11] = "==10"
problem47.constraints[11:12] = ">=1"
problem47.constraints[12:13] = ">=1"
problem47.constraints[13:14] = ">=1"
problem47.constraints[14:15] = ">=1"
problem47.constraints[15:16] = ">=1"
problem47.constraints[16:17] = "<=0.001"
problem47.directions[:] = Problem.MAXIMIZE
problem47.function = Opt

#Case48: DUH GFRP 12 layers
problem48 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem48.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem48.constraints[0:1] = ">=0"
problem48.constraints[1:2] = "==0"
problem48.constraints[2:3] = "==0"
problem48.constraints[3:4] = "==1"
problem48.constraints[4:5] = "==0"
problem48.constraints[5:6] = "==0"
problem48.constraints[6:7] = "==0"
problem48.constraints[7:10] = ">=0"
problem48.constraints[10:11] = "==12"
problem48.constraints[11:12] = ">=1"
problem48.constraints[12:13] = ">=1"
problem48.constraints[13:14] = ">=1"
problem48.constraints[14:15] = ">=1"
problem48.constraints[15:16] = ">=1"
problem48.constraints[16:17] = ">=1"
problem48.directions[:] = Problem.MAXIMIZE
problem48.function = Opt

#Case49: DUH CFRP 2 layers
problem49 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem49.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem49.constraints[0:1] = ">=0"
problem49.constraints[1:2] = "==0"
problem49.constraints[2:3] = "==0"
problem49.constraints[3:4] = "==1"
problem49.constraints[4:5] = "==1"
problem49.constraints[5:6] = "==0"
problem49.constraints[6:7] = "==0"
problem49.constraints[7:10] = ">=0"
problem49.constraints[10:11] = "==2"
problem49.constraints[11:12] = ">=1"
problem49.constraints[12:13] = "<=0.001"
problem49.constraints[13:14] = "<=0.001"
problem49.constraints[14:15] = "<=0.001"
problem49.constraints[15:16] = "<=0.001"
problem49.constraints[16:17] = "<=0.001"
problem49.directions[:] = Problem.MAXIMIZE
problem49.function = Opt

#Case50: DUH CFRP 4 layers
problem50 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem50.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem50.constraints[0:1] = ">=0"
problem50.constraints[1:2] = "==0"
problem50.constraints[2:3] = "==0"
problem50.constraints[3:4] = "==1"
problem50.constraints[4:5] = "==1"
problem50.constraints[5:6] = "==0"
problem50.constraints[6:7] = "==0"
problem50.constraints[7:10] = ">=0"
problem50.constraints[10:11] = "==4"
problem50.constraints[11:12] = ">=1"
problem50.constraints[12:13] = ">=1"
problem50.constraints[13:14] = "<=0.001"
problem50.constraints[14:15] = "<=0.001"
problem50.constraints[15:16] = "<=0.001"
problem50.constraints[16:17] = "<=0.001"
problem50.directions[:] = Problem.MAXIMIZE
problem50.function = Opt

#Case51: DUH CFRP 6 layers
problem51 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem51.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem51.constraints[0:1] = ">=0"
problem51.constraints[1:2] = "==0"
problem51.constraints[2:3] = "==0"
problem51.constraints[3:4] = "==1"
problem51.constraints[4:5] = "==1"
problem51.constraints[5:6] = "==0"
problem51.constraints[6:7] = "==0"
problem51.constraints[7:10] = ">=0"
problem51.constraints[10:11] = "==6"
problem51.constraints[11:12] = ">=1"
problem51.constraints[12:13] = ">=1"
problem51.constraints[13:14] = ">=1"
problem51.constraints[14:15] = "<=0.001"
problem51.constraints[15:16] = "<=0.001"
problem51.constraints[16:17] = "<=0.001"
problem51.directions[:] = Problem.MAXIMIZE
problem51.function = Opt

#Case52: DUH CFRP 8 layers
problem52 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem52.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem52.constraints[0:1] = ">=0"
problem52.constraints[1:2] = "==0"
problem52.constraints[2:3] = "==0"
problem52.constraints[3:4] = "==1"
problem52.constraints[4:5] = "==1"
problem52.constraints[5:6] = "==0"
problem52.constraints[6:7] = "==0"
problem52.constraints[7:10] = ">=0"
problem52.constraints[10:11] = "==8"
problem52.constraints[11:12] = ">=1"
problem52.constraints[12:13] = ">=1"
problem52.constraints[13:14] = ">=1"
problem52.constraints[14:15] = ">=1"
problem52.constraints[15:16] = "<=0.001"
problem52.constraints[16:17] = "<=0.001"
problem52.directions[:] = Problem.MAXIMIZE
problem52.function = Opt

#Case53: DUH CFRP 10 layers
problem53 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem53.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem53.constraints[0:1] = ">=0"
problem53.constraints[1:2] = "==0"
problem53.constraints[2:3] = "==0"
problem53.constraints[3:4] = "==1"
problem53.constraints[4:5] = "==1"
problem53.constraints[5:6] = "==0"
problem53.constraints[6:7] = "==0"
problem53.constraints[7:10] = ">=0"
problem53.constraints[10:11] = "==10"
problem53.constraints[11:12] = ">=1"
problem53.constraints[12:13] = ">=1"
problem53.constraints[13:14] = ">=1"
problem53.constraints[14:15] = ">=1"
problem53.constraints[15:16] = ">=1"
problem53.constraints[16:17] = "<=0.001"
problem53.directions[:] = Problem.MAXIMIZE
problem53.function = Opt

#Case54: DUH CFRP 12 layers
problem54 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem54.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem54.constraints[0:1] = ">=0"
problem54.constraints[1:2] = "==0"
problem54.constraints[2:3] = "==0"
problem54.constraints[3:4] = "==1"
problem54.constraints[4:5] = "==1"
problem54.constraints[5:6] = "==0"
problem54.constraints[6:7] = "==0"
problem54.constraints[7:10] = ">=0"
problem54.constraints[10:11] = "==12"
problem54.constraints[11:12] = ">=1"
problem54.constraints[12:13] = ">=1"
problem54.constraints[13:14] = ">=1"
problem54.constraints[14:15] = ">=1"
problem54.constraints[15:16] = ">=1"
problem54.constraints[16:17] = ">=1"
problem54.directions[:] = Problem.MAXIMIZE
problem54.function = Opt

#Case55: DUH CS 0 layers
problem55 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem55.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem55.constraints[0:1] = ">=0"
problem55.constraints[1:2] = "==0"
problem55.constraints[2:3] = "==0"
problem55.constraints[3:4] = "==1"
problem55.constraints[4:5] = "==0"
problem55.constraints[5:6] = "==1"
problem55.constraints[6:7] = "==0"
problem55.constraints[7:10] = ">=0"
problem55.constraints[10:11] = "==0"
problem55.constraints[11:12] = "<=0.001"
problem55.constraints[12:13] = "<=0.001"
problem55.constraints[13:14] = "<=0.001"
problem55.constraints[14:15] = "<=0.001"
problem55.constraints[15:16] = "<=0.001"
problem55.constraints[16:17] = "<=0.001"
problem55.directions[:] = Problem.MAXIMIZE
problem55.function = Opt

#Case56: DUH AL 0 layers
problem56 = Problem(16, 1, 17)
#Variable type of each variable (int, real)
problem56.types[:] = [Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Integer(0,1), Real(4.5,6), Real(1,3), Real(45,60), int1, Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2), Real(0,2)]
problem56.constraints[0:1] = ">=0"
problem56.constraints[1:2] = "==0"
problem56.constraints[2:3] = "==0"
problem56.constraints[3:4] = "==1"
problem56.constraints[4:5] = "==0"
problem56.constraints[5:6] = "==0"
problem56.constraints[6:7] = "==1"
problem56.constraints[7:10] = ">=0"
problem56.constraints[10:11] = "==0"
problem56.constraints[11:12] = "<=0.001"
problem56.constraints[12:13] = "<=0.001"
problem56.constraints[13:14] = "<=0.001"
problem56.constraints[14:15] = "<=0.001"
problem56.constraints[15:16] = "<=0.001"
problem56.constraints[16:17] = "<=0.001"
problem56.directions[:] = Problem.MAXIMIZE
problem56.function = Opt

iteration = 1000

i = 1
algorithm1 = NSGAII(problem1, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm1.run(iteration)

i = 2
algorithm2 = NSGAII(problem2, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm2.run(iteration)

i = 3
algorithm3 = NSGAII(problem3, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm3.run(iteration)

i = 4
algorithm4 = NSGAII(problem4, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm4.run(iteration)

i = 5
algorithm5 = NSGAII(problem5, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm5.run(iteration)

i = 6
algorithm6 = NSGAII(problem6, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm6.run(iteration)

i = 7
algorithm7 = NSGAII(problem7, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm7.run(iteration)

i = 8
algorithm8 = NSGAII(problem8, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm8.run(iteration)

i = 9
algorithm9 = NSGAII(problem9, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm9.run(iteration)

i = 10
algorithm10 = NSGAII(problem10, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm10.run(iteration)

i = 11
algorithm11 = NSGAII(problem11, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm11.run(iteration)

i = 12
algorithm12 = NSGAII(problem12, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm12.run(iteration)

i = 13
algorithm13 = NSGAII(problem13, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm13.run(iteration)

i = 14
algorithm14 = NSGAII(problem14, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm14.run(iteration)

i = 15
algorithm15 = NSGAII(problem15, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm15.run(iteration)

i = 16
algorithm16 = NSGAII(problem16, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm16.run(iteration)

i = 17
algorithm17 = NSGAII(problem17, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm17.run(iteration)

i = 18
algorithm18 = NSGAII(problem18, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm18.run(iteration)

i = 19
algorithm19 = NSGAII(problem19, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm19.run(iteration)

i = 20
algorithm20 = NSGAII(problem20, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm20.run(iteration)

i = 21
algorithm21 = NSGAII(problem21, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm21.run(iteration)

i = 22
algorithm22 = NSGAII(problem22, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm22.run(iteration)

i = 23
algorithm23 = NSGAII(problem23, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm23.run(iteration)

i = 24
algorithm24 = NSGAII(problem24, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm24.run(iteration)

i = 25
algorithm25 = NSGAII(problem25, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm25.run(iteration)

i = 26
algorithm26 = NSGAII(problem26, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm26.run(iteration)

i = 27
algorithm27 = NSGAII(problem27, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm27.run(iteration)

i = 28
algorithm28 = NSGAII(problem28, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm28.run(iteration)

i = 29
algorithm29 = NSGAII(problem29, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm29.run(iteration)

i = 30
algorithm30 = NSGAII(problem30, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm30.run(iteration)

i = 31
algorithm31 = NSGAII(problem31, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm31.run(iteration)

i = 32
algorithm32 = NSGAII(problem32, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm32.run(iteration)

i = 33
algorithm33 = NSGAII(problem33, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm33.run(iteration)

i = 34
algorithm34 = NSGAII(problem34, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm34.run(iteration)

i = 35
algorithm35 = NSGAII(problem35, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm35.run(iteration)

i = 36
algorithm36 = NSGAII(problem36, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm36.run(iteration)

i = 37
algorithm37 = NSGAII(problem37, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm37.run(iteration)

i = 38
algorithm38 = NSGAII(problem38, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm38.run(iteration)

i = 39
algorithm39 = NSGAII(problem39, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm39.run(iteration)

i = 40
algorithm40 = NSGAII(problem40, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm40.run(iteration)

i = 41
algorithm41 = NSGAII(problem41, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm41.run(iteration)

i = 42
algorithm42 = NSGAII(problem42, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm42.run(iteration)

i = 43
algorithm43 = NSGAII(problem43, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm43.run(iteration)

i = 44
algorithm44 = NSGAII(problem44, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm44.run(iteration)

i = 45
algorithm45 = NSGAII(problem45, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm45.run(iteration)

i = 46
algorithm46 = NSGAII(problem46, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm46.run(iteration)

i = 47
algorithm47 = NSGAII(problem47, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm47.run(iteration)

i = 48
algorithm48 = NSGAII(problem48, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm48.run(iteration)

i = 49
algorithm49 = NSGAII(problem49, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm49.run(iteration)

i = 50
algorithm50 = NSGAII(problem50, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm50.run(iteration)

i = 51
algorithm51 = NSGAII(problem51, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm51.run(iteration)

i = 52
algorithm52 = NSGAII(problem52, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm52.run(iteration)

i = 53
algorithm53 = NSGAII(problem53, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm53.run(iteration)

i = 54
algorithm54 = NSGAII(problem54, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm54.run(iteration)

i = 55
algorithm55 = NSGAII(problem55, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm55.run(iteration)

i = 56
algorithm56 = NSGAII(problem56, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm56.run(iteration)

result1 = algorithm1.result
result2 = algorithm2.result
result3 = algorithm3.result
result4 = algorithm4.result
result5 = algorithm5.result
result6 = algorithm6.result
result7 = algorithm7.result
result8 = algorithm8.result
result9 = algorithm9.result
result10 = algorithm10.result
result11 = algorithm11.result
result12 = algorithm12.result
result13 = algorithm13.result
result14 = algorithm14.result
result15 = algorithm15.result
result16 = algorithm16.result
result17 = algorithm17.result
result18 = algorithm18.result
result19 = algorithm19.result
result20 = algorithm20.result
result21 = algorithm21.result
result22 = algorithm22.result
result23 = algorithm23.result
result24 = algorithm24.result
result25 = algorithm25.result
result26 = algorithm26.result
result27 = algorithm27.result
result28 = algorithm28.result
result29 = algorithm29.result
result30 = algorithm30.result
result31 = algorithm31.result
result32 = algorithm32.result
result33 = algorithm33.result
result34 = algorithm34.result
result35 = algorithm35.result
result36 = algorithm36.result
result37 = algorithm37.result
result38 = algorithm38.result
result39 = algorithm39.result
result40 = algorithm40.result
result41 = algorithm41.result
result42 = algorithm42.result
result43 = algorithm43.result
result44 = algorithm44.result
result45 = algorithm45.result
result46 = algorithm46.result
result47 = algorithm47.result
result48 = algorithm48.result
result49 = algorithm49.result
result50 = algorithm50.result
result51 = algorithm51.result
result52 = algorithm52.result
result53 = algorithm53.result
result54 = algorithm54.result
result55 = algorithm55.result
result56 = algorithm56.result
feasible_solutions = [s for s in result1 if s.feasible]
for solution in feasible_solutions:
    print(solution.objectives)

SEA_hasil1 = [s.constraints[0] for s in algorithm1.result if s.feasible]
SEA_hasil2 = [s.constraints[0] for s in algorithm2.result if s.feasible]
SEA_hasil3 = [s.constraints[0] for s in algorithm3.result if s.feasible]
SEA_hasil4 = [s.constraints[0] for s in algorithm4.result if s.feasible]
SEA_hasil5 = [s.constraints[0] for s in algorithm5.result if s.feasible]
SEA_hasil6 = [s.constraints[0] for s in algorithm6.result if s.feasible]
SEA_hasil7 = [s.constraints[0] for s in algorithm7.result if s.feasible]
SEA_hasil8 = [s.constraints[0] for s in algorithm8.result if s.feasible]
SEA_hasil9 = [s.constraints[0] for s in algorithm9.result if s.feasible]
SEA_hasil10 = [s.constraints[0] for s in algorithm10.result if s.feasible]
SEA_hasil11 = [s.constraints[0] for s in algorithm11.result if s.feasible]
SEA_hasil12 = [s.constraints[0] for s in algorithm12.result if s.feasible]
SEA_hasil13 = [s.constraints[0] for s in algorithm13.result if s.feasible]
SEA_hasil14 = [s.constraints[0] for s in algorithm14.result if s.feasible]
SEA_hasil15 = [s.constraints[0] for s in algorithm15.result if s.feasible]
SEA_hasil16 = [s.constraints[0] for s in algorithm16.result if s.feasible]
SEA_hasil17 = [s.constraints[0] for s in algorithm17.result if s.feasible]
SEA_hasil18 = [s.constraints[0] for s in algorithm18.result if s.feasible]
SEA_hasil19 = [s.constraints[0] for s in algorithm19.result if s.feasible]
SEA_hasil20 = [s.constraints[0] for s in algorithm20.result if s.feasible]
SEA_hasil21 = [s.constraints[0] for s in algorithm21.result if s.feasible]
SEA_hasil22 = [s.constraints[0] for s in algorithm22.result if s.feasible]
SEA_hasil23 = [s.constraints[0] for s in algorithm23.result if s.feasible]
SEA_hasil24 = [s.constraints[0] for s in algorithm24.result if s.feasible]
SEA_hasil25 = [s.constraints[0] for s in algorithm25.result if s.feasible]
SEA_hasil26 = [s.constraints[0] for s in algorithm26.result if s.feasible]
SEA_hasil27 = [s.constraints[0] for s in algorithm27.result if s.feasible]
SEA_hasil28 = [s.constraints[0] for s in algorithm28.result if s.feasible]
SEA_hasil29 = [s.constraints[0] for s in algorithm29.result if s.feasible]
SEA_hasil30 = [s.constraints[0] for s in algorithm30.result if s.feasible]
SEA_hasil31 = [s.constraints[0] for s in algorithm31.result if s.feasible]
SEA_hasil32 = [s.constraints[0] for s in algorithm32.result if s.feasible]
SEA_hasil33 = [s.constraints[0] for s in algorithm33.result if s.feasible]
SEA_hasil34 = [s.constraints[0] for s in algorithm34.result if s.feasible]
SEA_hasil35 = [s.constraints[0] for s in algorithm35.result if s.feasible]
SEA_hasil36 = [s.constraints[0] for s in algorithm36.result if s.feasible]
SEA_hasil37 = [s.constraints[0] for s in algorithm37.result if s.feasible]
SEA_hasil38 = [s.constraints[0] for s in algorithm38.result if s.feasible]
SEA_hasil39 = [s.constraints[0] for s in algorithm39.result if s.feasible]
SEA_hasil40 = [s.constraints[0] for s in algorithm40.result if s.feasible]
SEA_hasil41 = [s.constraints[0] for s in algorithm41.result if s.feasible]
SEA_hasil42 = [s.constraints[0] for s in algorithm42.result if s.feasible]
SEA_hasil43 = [s.constraints[0] for s in algorithm43.result if s.feasible]
SEA_hasil44 = [s.constraints[0] for s in algorithm44.result if s.feasible]
SEA_hasil45 = [s.constraints[0] for s in algorithm45.result if s.feasible]
SEA_hasil46 = [s.constraints[0] for s in algorithm46.result if s.feasible]
SEA_hasil47 = [s.constraints[0] for s in algorithm47.result if s.feasible]
SEA_hasil48 = [s.constraints[0] for s in algorithm48.result if s.feasible]
SEA_hasil49 = [s.constraints[0] for s in algorithm49.result if s.feasible]
SEA_hasil50 = [s.constraints[0] for s in algorithm50.result if s.feasible]
SEA_hasil51 = [s.constraints[0] for s in algorithm51.result if s.feasible]
SEA_hasil52 = [s.constraints[0] for s in algorithm52.result if s.feasible]
SEA_hasil53 = [s.constraints[0] for s in algorithm53.result if s.feasible]
SEA_hasil54 = [s.constraints[0] for s in algorithm54.result if s.feasible]
SEA_hasil55 = [s.constraints[0] for s in algorithm55.result if s.feasible]
SEA_hasil56 = [s.constraints[0] for s in algorithm56.result if s.feasible]

SEA_hasill=np.hstack((SEA_hasil1, 	SEA_hasil2, 	SEA_hasil3, 	SEA_hasil4, 	SEA_hasil5, 	SEA_hasil6, 	SEA_hasil7, 	SEA_hasil8, 	SEA_hasil9, 	SEA_hasil10, 	SEA_hasil11, 	SEA_hasil12, 	SEA_hasil13, 	SEA_hasil14, 	SEA_hasil15, 	SEA_hasil16, 	SEA_hasil17, 	SEA_hasil18, 	SEA_hasil19, 	SEA_hasil20, 	SEA_hasil21, 	SEA_hasil22, 	SEA_hasil23, 	SEA_hasil24, 	SEA_hasil25, 	SEA_hasil26, 	SEA_hasil27, 	SEA_hasil28, 	SEA_hasil29, 	SEA_hasil30, 	SEA_hasil31, 	SEA_hasil32, 	SEA_hasil33, 	SEA_hasil34, 	SEA_hasil35, 	SEA_hasil36, 	SEA_hasil37, 	SEA_hasil38, 	SEA_hasil39, 	SEA_hasil40, 	SEA_hasil41, 	SEA_hasil42, 	SEA_hasil43, 	SEA_hasil44, 	SEA_hasil45, 	SEA_hasil46, 	SEA_hasil47, 	SEA_hasil48, 	SEA_hasil49, 	SEA_hasil50, 	SEA_hasil51, 	SEA_hasil52, 	SEA_hasil53, 	SEA_hasil54, 	SEA_hasil55, 	SEA_hasil56))
print("SEA_hasill = ")
print(SEA_hasill)

Variable1 = np.reshape(np.array([s.variables for s in algorithm1.result if s.feasible]), (-1,16))
Variable2 = np.reshape(np.array([s.variables for s in algorithm2.result if s.feasible]), (-1,16))
Variable3 = np.reshape(np.array([s.variables for s in algorithm3.result if s.feasible]), (-1,16))
Variable4 = np.reshape(np.array([s.variables for s in algorithm4.result if s.feasible]), (-1,16))
Variable5 = np.reshape(np.array([s.variables for s in algorithm5.result if s.feasible]), (-1,16))
Variable6 = np.reshape(np.array([s.variables for s in algorithm6.result if s.feasible]), (-1,16))
Variable7 = np.reshape(np.array([s.variables for s in algorithm7.result if s.feasible]), (-1,16))
Variable8 = np.reshape(np.array([s.variables for s in algorithm8.result if s.feasible]), (-1,16))
Variable9 = np.reshape(np.array([s.variables for s in algorithm9.result if s.feasible]), (-1,16))
Variable10 = np.reshape(np.array([s.variables for s in algorithm10.result if s.feasible]), (-1,16))
Variable11 = np.reshape(np.array([s.variables for s in algorithm11.result if s.feasible]), (-1,16))
Variable12 = np.reshape(np.array([s.variables for s in algorithm12.result if s.feasible]), (-1,16))
Variable13 = np.reshape(np.array([s.variables for s in algorithm13.result if s.feasible]), (-1,16))
Variable14 = np.reshape(np.array([s.variables for s in algorithm14.result if s.feasible]), (-1,16))
Variable15 = np.reshape(np.array([s.variables for s in algorithm15.result if s.feasible]), (-1,16))
Variable16 = np.reshape(np.array([s.variables for s in algorithm16.result if s.feasible]), (-1,16))
Variable17 = np.reshape(np.array([s.variables for s in algorithm17.result if s.feasible]), (-1,16))
Variable18 = np.reshape(np.array([s.variables for s in algorithm18.result if s.feasible]), (-1,16))
Variable19 = np.reshape(np.array([s.variables for s in algorithm19.result if s.feasible]), (-1,16))
Variable20 = np.reshape(np.array([s.variables for s in algorithm20.result if s.feasible]), (-1,16))
Variable21 = np.reshape(np.array([s.variables for s in algorithm21.result if s.feasible]), (-1,16))
Variable22 = np.reshape(np.array([s.variables for s in algorithm22.result if s.feasible]), (-1,16))
Variable23 = np.reshape(np.array([s.variables for s in algorithm23.result if s.feasible]), (-1,16))
Variable24 = np.reshape(np.array([s.variables for s in algorithm24.result if s.feasible]), (-1,16))
Variable25 = np.reshape(np.array([s.variables for s in algorithm25.result if s.feasible]), (-1,16))
Variable26 = np.reshape(np.array([s.variables for s in algorithm26.result if s.feasible]), (-1,16))
Variable27 = np.reshape(np.array([s.variables for s in algorithm27.result if s.feasible]), (-1,16))
Variable28 = np.reshape(np.array([s.variables for s in algorithm28.result if s.feasible]), (-1,16))
Variable29 = np.reshape(np.array([s.variables for s in algorithm29.result if s.feasible]), (-1,16))
Variable30 = np.reshape(np.array([s.variables for s in algorithm30.result if s.feasible]), (-1,16))
Variable31 = np.reshape(np.array([s.variables for s in algorithm31.result if s.feasible]), (-1,16))
Variable32 = np.reshape(np.array([s.variables for s in algorithm32.result if s.feasible]), (-1,16))
Variable33 = np.reshape(np.array([s.variables for s in algorithm33.result if s.feasible]), (-1,16))
Variable34 = np.reshape(np.array([s.variables for s in algorithm34.result if s.feasible]), (-1,16))
Variable35 = np.reshape(np.array([s.variables for s in algorithm35.result if s.feasible]), (-1,16))
Variable36 = np.reshape(np.array([s.variables for s in algorithm36.result if s.feasible]), (-1,16))
Variable37 = np.reshape(np.array([s.variables for s in algorithm37.result if s.feasible]), (-1,16))
Variable38 = np.reshape(np.array([s.variables for s in algorithm38.result if s.feasible]), (-1,16))
Variable39 = np.reshape(np.array([s.variables for s in algorithm39.result if s.feasible]), (-1,16))
Variable40 = np.reshape(np.array([s.variables for s in algorithm40.result if s.feasible]), (-1,16))
Variable41 = np.reshape(np.array([s.variables for s in algorithm41.result if s.feasible]), (-1,16))
Variable42 = np.reshape(np.array([s.variables for s in algorithm42.result if s.feasible]), (-1,16))
Variable43 = np.reshape(np.array([s.variables for s in algorithm43.result if s.feasible]), (-1,16))
Variable44 = np.reshape(np.array([s.variables for s in algorithm44.result if s.feasible]), (-1,16))
Variable45 = np.reshape(np.array([s.variables for s in algorithm45.result if s.feasible]), (-1,16))
Variable46 = np.reshape(np.array([s.variables for s in algorithm46.result if s.feasible]), (-1,16))
Variable47 = np.reshape(np.array([s.variables for s in algorithm47.result if s.feasible]), (-1,16))
Variable48 = np.reshape(np.array([s.variables for s in algorithm48.result if s.feasible]), (-1,16))
Variable49 = np.reshape(np.array([s.variables for s in algorithm49.result if s.feasible]), (-1,16))
Variable50 = np.reshape(np.array([s.variables for s in algorithm50.result if s.feasible]), (-1,16))
Variable51 = np.reshape(np.array([s.variables for s in algorithm51.result if s.feasible]), (-1,16))
Variable52 = np.reshape(np.array([s.variables for s in algorithm52.result if s.feasible]), (-1,16))
Variable53 = np.reshape(np.array([s.variables for s in algorithm53.result if s.feasible]), (-1,16))
Variable54 = np.reshape(np.array([s.variables for s in algorithm54.result if s.feasible]), (-1,16))
Variable55 = np.reshape(np.array([s.variables for s in algorithm55.result if s.feasible]), (-1,16))
Variable56 = np.reshape(np.array([s.variables for s in algorithm56.result if s.feasible]), (-1,16))
Variables = np.vstack((Variable1, 	Variable2, 	Variable3, 	Variable4, 	Variable5, 	Variable6, 	Variable7, 	Variable8, 	Variable9, 	Variable10, 	Variable11, 	Variable12, 	Variable13, 	Variable14, 	Variable15, 	Variable16, 	Variable17, 	Variable18, 	Variable19, 	Variable20, 	Variable21, 	Variable22, 	Variable23, 	Variable24, 	Variable25, 	Variable26, 	Variable27, 	Variable28, 	Variable29, 	Variable30, 	Variable31, 	Variable32, 	Variable33, 	Variable34, 	Variable35, 	Variable36, 	Variable37, 	Variable38, 	Variable39, 	Variable40, 	Variable41, 	Variable42, 	Variable43, 	Variable44, 	Variable45, 	Variable46, 	Variable47, 	Variable48, 	Variable49, 	Variable50, 	Variable51, 	Variable52, 	Variable53, 	Variable54, 	Variable55, 	Variable56))

print("Variables = ")
print(Variables)

solutionwrite = pd.ExcelWriter('NSGA Solutions.xlsx', engine='openpyxl')
pd.DataFrame(SEA_hasill).to_excel(solutionwrite, sheet_name="SEA")
pd.DataFrame(Variables).to_excel(solutionwrite, sheet_name="Variables")
solutionwrite.save()

x = SEA_hasill
y = SEA_hasill
#y = CFE_hasill

#CFE_Hasil = np.array([CFE_hasill]).reshape((500,1))+np.array(0.43344633)
#MCF_Hasil = np.array([MCF_hasill]).reshape((500,1))+np.array(0.10575172)
SEA_Hasil = np.array([SEA_hasill]).reshape((-1,1))+np.array(0.006)
#Pmax_Hasil = np.array([Pmax_hasill]).reshape((500,1))+np.array(0.20383371)
#Hasil = np.hstack((CFE_Hasil, MCF_Hasil, SEA_Hasil, Pmax_Hasil))
#Hasil_inverse=min_max_scaler.inverse_transform(Hasil)
#SEA_Hasil_inverse = Hasil_inverse[:,2]
SEA_Hasil_inverse = min_max_scaler.inverse_transform(SEA_Hasil)
plt.figure(1)
plt.scatter(SEA_Hasil_inverse,SEA_Hasil_inverse)
plt.xlabel('$f_1(SEA)$')
plt.ylabel('$f_2(SEA)$')
plt.show()

#MOORA algorithm for optimal point rank
n = len(x)
C1 = []
C2 = []
for i in range(0,n):
    C1.append(float(x[i]))
    C2.append(float(y[i]))

C1 = np.array(C1)
C2 = np.array(C2)

Categories = np.transpose(np.array((C1,C2)))
Norm_Categories = np.zeros(np.shape(Categories))
Criterion = np.array([0.35, 0.65])
for i in range (0,np.shape(Categories)[1]):
    sum_norm = 0
    for k in range (0,np.shape(Categories)[0]):
        sum_norm = sum_norm + (Categories[k][i])**2

    for j in range (0,np.shape(Categories)[0]):
        Norm_Categories[j][i] = Categories[j][i]/(sum_norm)**0.5

for i in range (0,np.shape(Categories)[1]):
    wj = Criterion[i]
    for j in range (0,np.shape(Categories)[0]):
        Norm_Categories[j][i] = Norm_Categories[j][i]*wj

Score = np.zeros((np.shape(Norm_Categories)[0],1))
for i in range(0,np.shape(Norm_Categories)[0]):
    Score[i] = Norm_Categories[i][0] + Norm_Categories[i][1]

Optimized = np.max(Score)
Selected = 0
for i in range (0,np.shape(Score)[0]):
    if Score[i] <= Optimized :
        Selected = i
        Optimized = Score[i]
    else:
        continue

Optimized_Solution = Variables[Selected]
Result_inverse = SEA_Hasil_inverse[Selected]
selectedmatrix = np.hstack((Result_inverse, Optimized_Solution))

pd.DataFrame(selectedmatrix).to_excel(solutionwrite, sheet_name="Selected")
solutionwrite.save()

print(Selected, Optimized_Solution, Result_inverse)
first_variable = Optimized_Solution[5]
print(int1.decode(first_variable))

plt.figure(2)
plt.scatter(SEA_Hasil_inverse[Selected],SEA_Hasil_inverse[Selected])
plt.xlabel('$f_1(SEA)$')
plt.ylabel('$f_2(SEA)$')
plt.show()
print(Baseline_transformed)