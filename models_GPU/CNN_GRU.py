from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Reshape, Activation, Conv2D, Conv2DTranspose, MaxPooling2D, LSTM, GRU, TimeDistributed
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
import numpy as np
import tensorflow as tf
# Data
#X = np.load('../data/X_5_40_16_4096.npy')
#Y = np.load('../data/Y_5_40_16_4096.npy')
#Y = np.reshape(Y, (4096, 16, 40, 40, 1))
#Y = np.where(Y>1, 1, 0)

# Model
model = Sequential()
model.add(TimeDistributed(Conv2D(64, (3, 3)), input_shape=(16, 40, 40, 3)))
model.add(TimeDistributed(Conv2D(128, (3, 3))))
model.add(TimeDistributed(Conv2D(256, (3, 3))))
model.add(TimeDistributed(Conv2D(256, (3, 3))))
model.add(TimeDistributed(Conv2D(128, (3, 3))))
model.add(TimeDistributed(Conv2D(1, (1, 1))))
model.add(TimeDistributed(Flatten()))
model.add(GRU(800,return_sequences=True))
model.add(GRU(800,return_sequences=True))
model.add(TimeDistributed(Dense(400)))
model.add(TimeDistributed(Dense(400)))
model.add(TimeDistributed(Reshape((20, 20, 1))))
model.add(TimeDistributed(Conv2DTranspose(1, (2, 2), strides=2)))
model.add(Activation('sigmoid'))

model.summary()

#model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
#model.fit(X, Y, batch_size=32, epochs=30, validation_split=0.05, callbacks=[EarlyStopping(restore_best_weights=True, patience=8)])
#model.save('../saved_weights/CNN_LSTM_bigger_more_data.h5')

model = tf.keras.models.load_model('../saved_weights_GPU/CNN_LSTM_bigger_more_data.h5')







