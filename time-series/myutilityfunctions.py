""" Utility functions """
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#


def create_subfolder(folder_name):
    """ Creates a subfolder where models will live. """
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
#


def create_data_for_time_series(length, times, seed=0):
    """ Dummy data. """
    np.random.seed(seed)
    #
    x1 = np.array(np.random.uniform(size=2*length*times)).reshape(2*length, times)
    x2 = np.array(np.random.uniform(size=2*length*times)).reshape(2*length, times)
    x3 = np.array(np.random.uniform(size=2*length*times)).reshape(2*length, times)
    x4 = np.array(np.random.uniform(size=2*length*times)).reshape(2*length, times)
    #
    y1 = np.roll(x1, 2)  # y1[t]=x1[t-2]
    y2 = np.roll(x2, 1) * np.roll(x3, 2)  # y2[t]=x2[t-1]*x3[t-2]
    y3 = np.roll(x4, 3)  # y3[t]=x4[t-3]
    #
    x1_train, x2_train, x3_train, x4_train = [x[0:length] for x in [x1, x2, x3, x4]]
    x1_test, x2_test, x3_test, x4_test = [x[length:2*length] for x in [x1, x2, x3, x4]]
    y1_train, y2_train, y3_train = [y[0:length] for y in [y1, y2, y3]]
    y1_test, y2_test, y3_test = [y[length:2*length] for y in [y1, y2, y3]]
    #
    return(x1_train, x2_train, x3_train, x4_train,
           x1_test, x2_test, x3_test, x4_test,
           y1_train, y2_train, y3_train,
           y1_test, y2_test, y3_test)
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
