"""
https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, TimeDistributed, Conv1D

import tensorflow as tf
#


def plot_losses(history):
    """ Plots the error history over epochs. """
    plt.plot(history.history['loss'], color="red")
    plt.plot(history.history['val_loss'], color="blue")
    red_patch = mpatches.Patch(color='red', label='Training')
    blue_patch = mpatches.Patch(color='blue', label='Test')
    plt.legend(handles=[red_patch, blue_patch])
    plt.xlabel('Epochs')
    plt.ylabel('MSE loss')
    plt.show()
#


def read_data():
    """ Imports the data and takes care of its peculiarities."""
    column_names = ["ID", "Activity", "Time", "x", "y", "z"]
    df = pd.read_csv("data/WISDM_ar_v1.1_raw.data", header=None, names=column_names)
    df["z"].replace(regex=True, inplace=True, to_replace=";", value="")
    df["z"] = pd.to_numeric(df["z"], errors="coerce")
    df.dropna(axis=0, how="any", inplace=True)
    return df
#

DATA = read_data()
# apply a convolution 1d of length 3 to a sequence with 10 timesteps, with 64 output filters

"""
MODEL = Sequential()
MODEL.add(Conv1D(64, 10, activation="tanh", input_shape=(DATA_LENGTH, DATA_FEATURES), padding="same")) # now model.output_shape == (None, 10, 64)
# add a new conv1d on top
MODEL.add(Conv1D(32, 5, activation="tanh", padding="same"))
# now model.output_shape == (None, 10, 32)
MODEL.add(TimeDistributed(Dense(activation='tanh', units=15)))
MODEL.add(Dense(activation="linear", units=1))
MODEL.compile(loss='mse', optimizer='adam')
#
x_train = DATA.loc[:2119, "Time"].values.reshape(1, 2120, 1)
r_train = DATA.loc[:2119, "Response(0)"].values.reshape(1, 2120, 1)
x_val = DATA.loc[2120:2820, "Time"].values.reshape(1, 700, 1)
r_val = DATA.loc[2120:2820, "Response(0)"].values.reshape(1, 700, 1)
#
HISTORY = MODEL.fit(x_train, r_train, batch_size=10, epochs=250, validation_data=(x_val, r_val))
plot_losses(HISTORY)
MODEL.save('models/1D-Conv.h5')
load_model('models/1D-Conv.h5')
#
"""
