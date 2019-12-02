#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 05:13:54 2019

@author: berk
"""

from keras.datasets import boston_housing

(train_data, train_targets),(test_data , test_targets) = boston_housing.load_data()


#normalization
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data/=std

test_data -= mean
test_data /= std


#initial the model


from keras import layers
from keras import models

def build_model():
        model = models.Sequential()
        model.add(layers.Dense(64,activation="relu",input_shape=(train_data.shape[1],)))        
        model.add(layers.Dense(64,activation="relu"))
        model.add(layers.Dense(1))        
        model.compile(optimizer="rmsprop",loss="mse",metrics=["mae"])
        return model

#k-fold cross validation

import numpy as np

k=4
num_val_samples = len(train_data) // k
epochs=500
all_scores=[]
all_mae_histories=[]

for i in range(k):
        print("proccesing fild #",i)
        
        val_data=        train_data[i*num_val_samples:(i+1)*num_val_samples]
        val_targets=     train_targets[i*num_val_samples:(i+1)*num_val_samples]
        
        partial_train_data = np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]],axis=0)
        partial_train_targets = np.concatenate([train_targets[:i*num_val_samples],train_targets[(i+1)*num_val_samples:]],axis=0)


        model = build_model()
        model.fit(partial_train_data,partial_train_targets,epochs=epochs,batch_size=1,verbose=0)
        mae_history = model.history.history['mae']
        all_mae_histories.append(mae_history)
        
average_mae_history=[np.mean([x[i] for x in all_mae_histories])for i in range(epochs)]

#visualation

import matplotlib.pyplot as plt

plt.plot(range(1,len(average_mae_history)+1),average_mae_history)
plt.xlabel("epochs")
plt.ylabel("MAE-(Validation)")
plt.show()
        
#plot 20:500 epochs (throw first 10 epochs) and smooth the curve

def smooth_curve(points,factor=0.9):
        smooth_points=[]
        for point in points:
                if smooth_points:
                        previous = smooth_points[-1]
                        smooth_points.append(previous*factor+point*(1-factor))
                else:
                        smooth_points.append(point)
                
                return smooth_points

smooth_mae_history=smooth_curve(average_mae_history[10:])

plt.plot(range(1,len(average_mae_history)+1),average_mae_history)
plt.xlabel("epochs")
plt.ylabel("MAE-(Validation)")
plt.show()

#overfitting began after 80th epoch. Set the new number of epochs as 80 and train again

model = build_model()
model = build_model()
model.fit(train_data,train_targets,epochs=80,batch_size=16,verbose=0)
test_mse_score , test_mae_score = model.evaluate(test_data,test_targets)      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        