# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 13:01:22 2022

@author: lenovo
"""
import math
from sklearn import datasets
from pandas import DataFrame
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn import preprocessing
from numpy.ma.core import log
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler 
from keras.models import Sequential 
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout 
from keras import regularizers


coord = pd.read_csv("/content/drive/MyDrive/theta.csv",encoding='latin1')
xyt = pd.read_csv("/content/drive/MyDrive/xyt2.csv",encoding='latin1')

coord = shuffle(coord)
xyt = shuffle(xyt)


####################Set the first four fifth as the training set and the rest as the testing set############################################################
x_train = pd.DataFrame(xyt[:53])
y_train = pd.DataFrame(coord[:53])
x_valid = pd.DataFrame(xyt[53:])
y_valid = pd.DataFrame(coord[53:])


####################The structure of the MLP############################################################
model = Sequential()  # initialization
model.add(Dense(units = 512,   # output size 
                activation='relu',  # activation function  
                 input_shape=(x_train.shape[1],)  # Enter the size, which is the size of the column
                )  
          )    
  
model.add(Dense(units = 128,  
                activation='relu'    
                )  
          )  
  
model.add(Dense(units = 10,     
                activation='relu'      
                )  
          )  
model.add(Dense(units = 4,     
                activation='linear'  # Linear activation function. Regression generally uses this activation function in the output layer    
                )  
          ) 

   
print(model.summary())  # print network hierarchy  
   
model.compile(loss='mse',  # loss mean squared error
       #optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False),  # 优化器  
       optimizer=tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False),  # 优化器
              )  
history = model.fit(x_train, y_train,  
           epochs=300,  # number of iterations 
           batch_size=53,  # batch size for gradient descent 
           verbose=2,  #verbose: log verbosity length, int: verbosity length, 0: do not output the training process, 1: output the training progress, 2: output each epoch  
           validation_data = (x_valid, y_valid)  # validation set 
         )


####################Plot loss values for training & validation ####################
import matplotlib.pyplot as plt  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('Model loss')  
plt.ylabel('Loss')  
plt.xlabel('Epoch')  
plt.legend(['Train', 'Test'], loc='upper left')  
plt.show()
   
y_new = model.predict(x_valid)  
