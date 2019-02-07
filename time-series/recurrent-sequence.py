import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
# from scipy.stats import mode
# from sklearn.metrics import confusion_matrix, classification_report

# from matplotlib import pyplot as plt
# import matplotlib.patches as mpatches
# from keras.models import Sequential, load_model
# from keras.layers import Dense, Conv1D, Dropout, GlobalAveragePooling1D, MaxPooling1D
# from keras.callbacks import ModelCheckpoint, EarlyStopping
# from keras.utils import np_utils


def prep_data():
    """ Loads the data and prepares it to be inputted to the network. """
    column_names = ["dew", "time", "m", "rpm"]
    df = pd.read_csv("data/WISDM_ar_v1.1_raw.data", header=0, names=column_names)
    df.dropna(axis=0, how="any", inplace=True)
    #
    scaler = RobustScaler()  # There are other options.
    df["time scaled"], df["m scaled"], df["rpm scaled"] = None, None, None
    df[["time scaled", "m scaled", "rpm scaled"]] = scaler.fit_transform(df[["time", "m", "rpm"]])
    #
    dews = df["dew"].unique()
    np.random.shuffle(dews)
    training_cutoff = np.rint(dews.size * 0.75).astype(int)
    training_dews = dews[:training_cutoff]
    df["Train/Test"] = "Test"
    df.loc[df["dew"].isin(training_dews), "Train/Test"] = "Train"
    #
    # DEFINE X, Y FOR TRAIN, TEST
    #
    return None
#