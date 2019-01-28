import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, TimeDistributed
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


# DATA = pd.DataFrame({"Time": np.linspace(start=0.0, stop=30.0, num=1000, endpoint=False)})
# DATA["Response(0)"]=np.sin(DATA["Time"])+np.linspace(start=0.0, stop=5.0, num=1000, endpoint=False)+np.random.uniform(low=0.0, high=0.2, size=1000)
DATA = pd.read_csv("data/zuerich-monthly-sunspot-numbers.csv")
DATA["Response(-1)"] = DATA["Response(0)"].shift(-1)
DATA["Response(-2)"] = DATA["Response(0)"].shift(-2)
DATA["Response(-3)"] = DATA["Response(0)"].shift(-3)
DATA["Response(-4)"] = DATA["Response(0)"].shift(-4)
DATA["Response(-5)"] = DATA["Response(0)"].shift(-5)

# expected input data shape: (batch_size, timesteps, data_dim)
MODEL = Sequential()
MODEL.add(LSTM(50, return_sequences=True, input_shape=(None, 6)))  # returns a sequence of vectors of dimension 32
MODEL.add(LSTM(25, return_sequences=True))  # returns a sequence of vectors of dimension 32
#MODEL.add(LSTM(30, return_sequences=True))  # return a single vector of dimension 32
MODEL.add(TimeDistributed(Dense(activation='tanh', units=100)))
MODEL.add(Dense(activation="linear", units=1))
MODEL.compile(loss='mse', optimizer='adam')

# Use sunspot data
x_train = DATA.loc[:2114, "Time":"Response(-4)"].values.reshape(2115, 1, 6)
r_train = DATA.loc[:2114, "Response(-5)"].values.reshape(2115, 1, 1)
#
x_val = DATA.loc[2115:2814, "Time":"Response(-4)"].values.reshape(700, 1, 6)
r_val = DATA.loc[2115:2814, "Response(-5)"].values.reshape(700, 1, 1)
# Generate dummy training data
# x_train = DATA.loc[:499, "Time":"Response(-4)"].values.reshape(500, 1, 6)
# r_train = DATA.loc[:499, "Response(-5)"].values.reshape(500, 1, 1)
#
# x_val = DATA.loc[500:994, "Time":"Response(-4)"].values.reshape(495, 1, 6)
# r_val = DATA.loc[500:994, "Response(-5)"].values.reshape(495, 1, 1)


HISTORY = MODEL.fit(x_train, r_train, batch_size=10, epochs=250, validation_data=(x_val, r_val))
plot_losses(HISTORY)
MODEL.save('models/simpler.h5')
load_model('models/simpler.h5')

# x = DATA.loc[:, "Time":"Response(-4)"].values.reshape(1000, 1, 6)
# x = DATA["Time"].values.reshape(1000, 1, 1)
x = DATA.loc[:, "Time":"Response(-4)"].values.reshape(2820, 1, 6)
y_hat = MODEL.predict(x).flatten()
y = DATA["Response(0)"].values.flatten()
plt.plot(DATA["Time"], y)
plt.plot(DATA["Time"], y_hat)
plt.show()
