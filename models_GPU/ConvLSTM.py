#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 14:38:32 2019

@author: nrchilku
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv3D, ConvLSTM2D, BatchNormalization, Activation, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

#Data
X = np.load('../data/X_5_40_16_4096.npy')
Y = np.load('../data/Y_5_40_16_4096.npy')
Y = np.reshape(Y, (4096, 16, 40, 40, 1))
Y = np.where(Y > 1, 1, 0)

X = tf.cast(X, tf.float32)
Y = tf.cast(Y, tf.float32)

model = Sequential()

model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same', input_shape=(16, 40, 40, 3))))
model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same')))
model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same')))
model.add(TimeDistributed(Conv2D(256, (3, 3), padding='same')))
model.add(TimeDistributed(Conv2D(256, (3, 3), padding='same')))
model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same')))
model.add(TimeDistributed(Conv2D(40, (3, 3), padding='same')))

model.add(ConvLSTM2D(40, (3, 3), padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(40, (3, 3), padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(40, (3, 3), padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same', data_format='channels_last'))
model.compile(loss='binary_crossentropy', optimizer='RMSprop')
#model.summary()
#model.fit(X, Y, batch_size=16, epochs=100, validation_split=0.1, 
#        callbacks=[EarlyStopping(restore_best_weights=True, monitor='val_loss', patience=30)])
#model.save('../saved_weights/ConvLSTM_CNN_bigger.h5')

model = tf.keras.models.load_model('../saved_weights_GPU/ConvLSTM_CNN_bigger.h5')
       

                




