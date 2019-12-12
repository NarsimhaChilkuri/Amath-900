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

#from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.layers import Dense, Input, GlobalAveragePooling2D, Conv2DTranspose, Reshape
#from keras.layers import TimeDistributed
#from keras.layers.recurrent import RNN
#from keras.models import Sequential, Model
#from keras.utils import multi_gpu_model, to_categorical

#import tensorflow
from keras.utils import multi_gpu_model
from keras.initializers import Constant
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Reshape, Activation, Conv2D, Conv2DTranspose, MaxPooling2D, RNN, LSTM, GRU, TimeDistributed
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.optimizers import SGD
import numpy as np

# Data
X = np.load('../data/X_5_40_16_4096.npy')
Y = np.load('../data/Y_5_40_16_4096.npy')
Y = np.reshape(Y, (4096, 16, 40, 40, 1))
Y = np.where(Y>1, 1, 0)
X_test = np.load('../data/X_5_40_16_100.npy')
# LMU layer
def lmu_layer(return_sequences=False,**kwargs):
    return RNN(LMUCell(units=800,
                       order=1200,
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
#model.add(TimeDistributed(MaxPooling2D((3, 3), strides=2)))

model.add(TimeDistributed(Conv2D(128, (3, 3))))
model.add(TimeDistributed(Conv2D(1, (3, 3))))
#model.add(TimeDistributed(MaxPooling2D((3, 3), strides=2)))

model.add(TimeDistributed(Flatten()))
model.add(LSTM(800, return_sequences=True))
model.add(lmu_layer(return_sequences=True))
#model.add(lmu_layer(return_sequences=True))

model.add(TimeDistributed(Dense(400)))
model.add(TimeDistributed(Dense(400)))

model.add(TimeDistributed(Reshape((20, 20, 1))))
model.add(TimeDistributed(Conv2DTranspose(1, (2, 2), strides=2)))
model.add(Activation('sigmoid'))
#model.add(TimeDistributed(Conv2DTranspose(1, (2, 2), strides=2)))

model.summary()
#model = multi_gpu_model(model, gpus=4)
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(X, Y, batch_size=32, epochs=100, validation_split=0.05, callbacks=[EarlyStopping(restore_best_weights=True, patience=40)])
p = model.predict(X_test)
print(p.shape)
np.save("pred_lmu", p) 
model.save('../saved_weights/CNN_LMU_mod.h5')
