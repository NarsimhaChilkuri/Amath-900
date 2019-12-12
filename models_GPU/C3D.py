#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 18:16:49 2019

@author: nrchilku
"""
import numpy as np
from keras import Sequential
from keras.layers import Conv3D, BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import tensorflow as tf
#Data
#X = np.load('../data/X_5_40_16_4096.npy')
#Y = np.load('../data/Y_5_40_16_4096.npy')
#Y = np.reshape(Y, (4096, 16, 40, 40, 1))
#Y = np.where(Y > 1, 1, 0)

# Model is somewhat similar to C3D and Early fusion CNN.
model = Sequential()
model.add(Conv3D(64, (3,3,3), activation='relu', padding='same', input_shape=(16, 40, 40, 3), data_format='channels_last'))
model.add(Conv3D(128, (3,3,3), activation='relu', padding='same'))
model.add(Conv3D(256, (3,3,3), activation='relu', padding='same'))
model.add(Conv3D(256, (3,3,3), activation='relu', padding='same'))
model.add(Conv3D(128, (3,3,3), activation='relu', padding='same'))
model.add(Conv3D(64, (2,2,2), activation='relu', padding='same'))
model.add(Conv3D(32, (2,2,2), activation='relu', padding='same'))
model.add(Conv3D(1, (3,3,3), activation='sigmoid', padding='same'))

model.summary()
#sgd = SGD(lr=0.005, momentum=0.9, nesterov=True)
#model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
#model.fit(X, Y, batch_size=16, epochs=30, validation_split=0.05, 
#        callbacks=[EarlyStopping(restore_best_weights=True, patience=8)])
#model.save('../saved_weights/C3D_more_epochs_more_data_without_batch_norm.h5')
model = tf.keras.models.load_model('../saved_weights_GPU/C3D_more_epochs_more_data_bigger.h5')









