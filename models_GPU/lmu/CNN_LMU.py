#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:52:42 2019

@author: nrchilku
"""
import nengo
import os
import sys
lmu_path = os.path.abspath("../lmu")
sys.path.append(lmu_path)
from lmu import LMUCell

import tensorflow
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Reshape, Activation, Conv2D, Conv2DTranspose, MaxPooling2D, RNN, LSTM, GRU, TimeDistributed
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.optimizers import SGD
import numpy as np

# Data
X = np.load('../data/X_5_40_16_2048.npy')
Y = np.load('../data/Y_5_40_16_2048.npy')
Y = np.reshape(Y, (2048, 16, 40, 40, 1))
Y = np.where(Y>1, 1, 0)

# LMU layer
def LMU(return_sequences=False,**kwargs):
    return RNN(LMUCell(units=800,
                       order=6,
                       theta=16,
                       input_encoders_initializer=Constant(1),
                       hidden_encoders_initializer=Constant(0),
                       memory_encoders_initializer=Constant(0),
                       input_kernel_initializer=Constant(0),
                       hidden_kernel_initializer=Constant(0),
                       memory_kernel_initializer='glorot_normal',
                      ),
               return_sequences=return_sequences,
               **kwargs)
# Model
model = Sequential()
model.add(TimeDistributed(Conv2D(64, (3, 3)), input_shape=(16, 40, 40, 3)))
model.add(TimeDistributed(Conv2D(128, (3, 3))))
#model.add(TimeDistributed(MaxPooling2D((3, 3), strides=2)))

model.add(TimeDistributed(Conv2D(256, (3, 3))))
model.add(TimeDistributed(Conv2D(256, (3, 3))))
#model.add(TimeDistributed(MaxPooling2D((3, 3), strides=1)))

model.add(TimeDistributed(Conv2D(128, (3, 3))))
model.add(TimeDistributed(Conv2D(1, (3, 3))))
#model.add(TimeDistributed(MaxPooling2D((3, 3), strides=2)))

model.add(TimeDistributed(Flatten()))
model.add(LMU(return_sequences=True))
model.add(LMU(return_sequences=True))

model.add(TimeDistributed(Dense(400)))
model.add(TimeDistributed(Dense(400)))

model.add(TimeDistributed(Reshape((20, 20, 1))))
model.add(TimeDistributed(Conv2DTranspose(1, (2, 2), strides=2)))
model.add(Activation('sigmoid'))
#model.add(TimeDistributed(Conv2DTranspose(1, (2, 2), strides=2)))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
model.fit(X, Y, batch_size=16, epochs=15, validation_split=0.05, callbacks=[EarlyStopping(restore_best_weights=True, patience=8)])
model.save('../saved_weights/CNN_LSTM_five_steps.h5')


#model = tensorflow.keras.models.load_model('../saved_weights_GPU/CNN_LSTM_bigger_more_data.h5')