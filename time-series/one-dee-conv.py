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
DATA = pd.read_csv("data/zuerich-monthly-sunspot-numbers.csv")


# apply a convolution 1d of length 3 to a sequence with 10 timesteps, with 64 output filters
MODEL = Sequential()
MODEL.add(Convolution1D(64, 3, padding='same', input_shape=(10, 32)))
# now model.output_shape == (None, 10, 64)

# add a new conv1d on top
MODEL.add(Convolution1D(32, 3, border_mode='same'))
# now model.output_shape == (None, 10, 32)