import os
import numpy as np
from keras.models import load_model, Sequential
from keras.layers import Dense, LSTM, TimeDistributed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from myutilityfunctions import *

create_subfolder("models")

LENGTH = 13 * 17 * 3  # size of samples = 1547
TIME = 37  # length of each sample is short = 37
x1_train, x2_train, x3_train, x4_train, x1_test, x2_test, x3_test, x4_test, \
y1_train, y2_train, y3_train, y1_test, y2_test, y3_test = create_data_for_time_series(LENGTH, TIME)
#
m = 1
inputs = x1_train.reshape(LENGTH, TIME, m)
outputs = y1_train.reshape(LENGTH, TIME, m)
inputs_test = x1_test.reshape(LENGTH, TIME, m)
outputs_test = y1_test.reshape(LENGTH, TIME, m)
#
MODEL = Sequential()
dim_in = m
dim_out = m
nb_units = 10  # will also work with 2 units, but too long to train
MODEL.add(LSTM(input_shape=(None, dim_in), return_sequences=True, units=nb_units))
MODEL.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
MODEL.compile(loss='mse', optimizer='rmsprop')
#
np.random.seed(1337)
HISTORY = MODEL.fit(inputs, outputs, epochs=100, batch_size=32,
                    validation_data=(inputs_test, outputs_test))
plot_losses(HISTORY)
MODEL.save('models/4_A_y1_from_x1.h5')
load_model('models/4_A_y1_from_x1.h5')
# After 100 epochs: loss: 0.0048 / val_loss: 0.0047.
#
n = 0  # time series selected (between 0 and N-1)
idx = range(n, n+1)
x = inputs_test[idx].flatten()
y_hat = MODEL.predict(inputs_test[idx]).flatten()
y = outputs_test[idx].flatten()
plt.plot(range(TIME), y)
plt.plot(range(TIME), y_hat)
# plt.plot(range(T), x)
