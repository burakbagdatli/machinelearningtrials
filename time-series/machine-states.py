"""
How about this:
https://mourafiq.com/2016/05/15/predicting-sequences-using-rnn-in-tensorflow.html
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
#from seaborn import heatmap
#from scipy.stats import mode
#from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
#from matplotlib import pyplot as plt
#import matplotlib.patches as mpatches
#from keras.models import Sequential, load_model
#from keras.layers import Dense, Conv1D, Dropout, GlobalAveragePooling1D, MaxPooling1D
#from keras.callbacks import ModelCheckpoint, EarlyStopping
#from keras.utils import np_utils


def create_subfolder(folder_name):
    """ Creates a subfolder where models will live. """
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
#


def start_tf_session():
    """ Starts a usable tensorflow session. """
    config = None
    if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
        # This may be unnecessary. On my computer cuDNN fails to initialize when TF asks for too much GPU memory.
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.85))
    return tf.Session(config=config)
#


def create_segments(data_df, seq_length):
    """
        I modified this only very slightly from the source.
        Source: https://github.com/ni79ls/har-keras-cnn
    """
    # df = data_df.loc[data_df["Train/Test"]==train_test, ["x scaled", "y scaled", "z scaled", "Encoded Activity"]]
    part_numbers = data_df["Part Number"].unique().tolist()
    segments = pd.DataFrame(columns=data_df.columns.values.tolist())
    segments["segment"] = None
    # labels = []
    for part in part_numbers:
        part_df = data_df.loc[data_df["Part Number"]==part]
        length = len(part_df) * seq_length - 2 * np.sum(np.arange(seq_length))
        part_segments = pd.DataFrame(index=pd.RangeIndex(length), columns=data_df.columns.values.tolist())
        part_segments["Part Number"] = part
        part_segments["segment"] = part_segments.index // seq_length
        for row in part_segments.itertuples():
            part_segments_index = row[0]
            part_df_index = part_df.index[row[7] + ( part_segments_index % seq_length )] # segment column
            part_segments.loc[part_segments_index, "dateTime"] = part_df.loc[part_df_index, "dateTime"]
            part_segments.loc[part_segments_index, "dataItemId"] = part_df.loc[part_df_index, "dataItemId"]
            part_segments.loc[part_segments_index, "value"] = part_df.loc[part_df_index, "value"]
            part_segments.loc[part_segments_index, "dataItemId_encoded"] = part_df.loc[part_df_index, "dataItemId_encoded"]
            part_segments.loc[part_segments_index, "Train/Test"] = part_df.loc[part_df_index, "Train/Test"]
        segments = pd.concat([segments, part_segments], axis=0, ignore_index=True)
    segments.reset_index(drop=True, inplace=True)
    segments.to_csv("data/ExperimentData_wrangled.data", index=False)
    return 1, 2
#


def prep_data(sequence_length):
    """ Imports the data and takes care of its peculiarities. """
    df = pd.read_csv("data/ExperimentData.data", header=0)
    df.dropna(axis=0, how="any", inplace=True)
    #
    # machine_states = df["dataItemId"].unique().tolist()
    encoder = LabelEncoder()
    df["dataItemId_encoded"] = encoder.fit_transform(df["dataItemId"].values.ravel())
    #
    part_ids = df["Part Number"].unique()
    np.random.shuffle(part_ids)
    training_cutoff = np.rint(part_ids.size * 0.75).astype(int)
    training_part_ids = part_ids[:training_cutoff]
    df["Train/Test"] = "Test"
    df.loc[df["Part Number"].isin(training_part_ids), "Train/Test"] = "Train"
    #
    x_train, y_train = create_segments(df.loc[df["Train/Test"]=="Train"], sequence_length)
    #num_features = x_train.shape[2]
    #num_classes = encoder.classes_.size
    #y_train = np_utils.to_categorical(y_train, num_classes)
    #x_test, y_test = create_segments_and_labels(df, time_periods, step_distance, "Test")
    #y_test = np_utils.to_categorical(y_test, num_classes)
    #
    #return x_train, y_train, x_test, y_test, num_features, num_classes, labels
    return None
#

prep_data(5)
