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


def load_data():
    """ Loads the initial data in a Pandas DataFrame. """
    # df = pd.read_csv("data/ExperimentData.data", header=0)
    df = pd.read_csv("data/ExperimentData_abr.data", header=0) # Smaller dataset for debugging
    # Remove rows with missing data
    # df.dropna(axis=0, how="any", inplace=True)
    # Turn date times to values that can be used for calculations and calculate durations
    df["dateTime_value"] = pd.to_datetime(df["dateTime"])
    df["Duration"] = df["dateTime_value"].shift(-1) - df["dateTime_value"]
    df["Different Part?"] = df["Part Number"].shift(-1) - df["Part Number"]
    df.loc[df["Different Part?"] != 0, "Duration"] = pd.to_timedelta('0')
    df.loc[df.index[-1], "Duration"] = pd.to_timedelta('0')
    df.drop(columns="Different Part?", inplace=True)
    # Create training/validation/test labels
    part_ids = df["Part Number"].unique()
    np.random.shuffle(part_ids)
    training_cutoff = np.rint(part_ids.size * 0.75).astype(int)
    training_part_ids = part_ids[:training_cutoff]
    df["Train/Test"] = "Test"
    df.loc[df["Part Number"].isin(training_part_ids), "Train/Test"] = "Train"
    # Label encode dataItemIds
    data_item_id_encoder = LabelEncoder()
    df["dataItemId_encoded"] = data_item_id_encoder.fit_transform(df["dataItemId"].values.ravel())
    process_encoder = LabelEncoder()
    df["Process_encoded"] = process_encoder.fit_transform(df["Process"].values.ravel())
    # Save and return
    df.to_csv("data/ExperimentData_processed.data", index=False)
    return df
#


def create_segments(data_df, seq_length):
    """
        I modified this only very slightly from the source.
        Source: https://github.com/ni79ls/har-keras-cnn
    """
    part_numbers = data_df["Part Number"].unique().tolist()
    segments = pd.DataFrame(columns=data_df.columns.values.tolist())
    segments["Segment"] = None
    segments["Label"] = None
    segment_column_location_in_tuple = segments.columns.get_loc("Segment") + 1
    segment_base = 0
    print(segment_base)
    # labels = []
    for part in part_numbers:
        part_df = data_df.loc[data_df["Part Number"]==part]
        length = len(part_df) * seq_length - 2 * np.sum(np.arange(seq_length))
        part_segments = pd.DataFrame(index=pd.RangeIndex(length), columns=data_df.columns.values.tolist())
        part_segments["Part Number"] = part
        part_segments["Segment"] = part_segments.index // seq_length
        for row in part_segments.itertuples():
            part_segments_index = row[0]
            part_df_index = part_df.index[row[segment_column_location_in_tuple] + ( part_segments_index % seq_length )]
            part_segments.loc[part_segments_index, "dateTime"] = part_df.loc[part_df_index, "dateTime"]
            part_segments.loc[part_segments_index, "dataItemId"] = part_df.loc[part_df_index, "dataItemId"]
            part_segments.loc[part_segments_index, "value"] = part_df.loc[part_df_index, "value"]
            # Part Number taken care of earlier
            part_segments.loc[part_segments_index, "Process"] = part_df.loc[part_df_index, "Process"]
            part_segments.loc[part_segments_index, "dateTime_value"] = part_df.loc[part_df_index, "dateTime_value"]
            part_segments.loc[part_segments_index, "Duration"] = part_df.loc[part_df_index, "Duration"]
            part_segments.loc[part_segments_index, "dataItemId_encoded"] = part_df.loc[part_df_index, "dataItemId_encoded"]
            part_segments.loc[part_segments_index, "Process_encoded"] = part_df.loc[part_df_index, "Process_encoded"]
            part_segments.loc[part_segments_index, "Train/Test"] = part_df.loc[part_df_index, "Train/Test"]
        part_segments["Segment"] = np.repeat(segment_base, length) + part_segments["Segment"] # shift them for uniqueness
        segments = pd.concat([segments, part_segments], axis=0, ignore_index=True)
        segment_base = part_segments["Segment"].max() + 1 # calculate the new base
    segments.reset_index(drop=True, inplace=True)
    for segment in np.nditer(segments["Segment"].unique()):
        this_segment = segments.loc[segments["Segment"]==segment]
        if this_segment.loc["Process_encoded"].sum() % seq_length == 0:
            this_segment.loc["Label"] = this_segment.loc["Process_encoded"]
        elif this_segment.iloc[1, this_segment.columns.get_loc("Process_encoded")] == 0:
            this_segment.loc["Label"] = np.repeat(1, seq_length)
        else:
            this_segment.loc["Label"] = np.repeat(0, seq_length)
    segments.to_csv("data/ExperimentData_segments_"+str(int(seq_length))+".data", index=False)
    return segments
#


def prep_data(sequence_length):
    """ Imports the data and takes care of its peculiarities. """
    print("Processing raw data...")
    try:
        data = pd.read_csv("data/ExperimentData_processed.data", header=0)
        data["dateTime_value"] = pd.to_datetime(data["dateTime"]) # Cannot be recovered once saved to csv
        data["Duration"] = pd.to_timedelta(data["Duration"]) # Cannot be recovered once saved to csv
    except FileNotFoundError:
        data = load_data()
    print("Segmenting processed data...")
    try:
        segments = pd.read_csv("data/ExperimentData_segments_"+str(int(sequence_length))+".data", header=0)
        # Check to make sure segmentation length is valid
        if segments.loc[segments["Segment"]==0].shape[0] != sequence_length:
            raise ValueError
        segments["dateTime_value"] = pd.to_datetime(segments["dateTime"]) # Cannot be recovered once saved to csv
        segments["Duration"] = pd.to_timedelta(segments["Duration"]) # Cannot be recovered once saved to csv
        # encoding loses bitdepth. Is this a problem?
    except (FileNotFoundError, ValueError, KeyError):
        segments = create_segments(data.loc[data["Train/Test"]=="Train"], sequence_length)
    #num_features = x_train.shape[2]
    #num_classes = encoder.classes_.size
    #y_train = np_utils.to_categorical(y_train, num_classes)
    #x_test, y_test = create_segments_and_labels(df, time_periods, step_distance, "Test")
    #y_test = np_utils.to_categorical(y_test, num_classes)
    #
    #return x_train, y_train, x_test, y_test, num_features, num_classes, labels
    return None
#

prep_data(3)
#prep_data(4)
#prep_data(5)
#prep_data(6)
#prep_data(7)
#prep_data(8)
#prep_data(9)
#prep_data(10)
