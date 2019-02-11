"""
How about this:
https://mourafiq.com/2016/05/15/predicting-sequences-using-rnn-in-tensorflow.html
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from seaborn import heatmap
#from scipy.stats import mode
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
#import matplotlib.patches as mpatches
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, TimeDistributed
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils

SEQUENCE_LENGTH = 5
REFIT = True


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
    df = pd.read_csv("data/ExperimentData.data", header=0)
    # df = pd.read_csv("data/ExperimentData_abr.data", header=0) # Smaller dataset for debugging
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
    return df, {"dataItemId":data_item_id_encoder, "Process":process_encoder}
#


def create_segments(data_df, seq_length):
    """
        Creates overlapping segments for learning
    """
    part_numbers = data_df["Part Number"].unique().tolist()
    segments = pd.DataFrame(columns=data_df.columns.values.tolist())
    segments["Segment"] = None
    segments["Label"] = None
    segment_column_location_in_tuple = segments.columns.get_loc("Segment") + 1
    segment_base = 0
    # labels = []
    for part in part_numbers:
        part_df = data_df.loc[data_df["Part Number"]==part]
        length = len(part_df) * seq_length - 2 * np.sum(np.arange(seq_length))
        part_segments = pd.DataFrame(index=pd.RangeIndex(length), columns=segments.columns.values.tolist())
        part_segments["Part Number"] = part
        part_segments["Segment"] = part_segments.index // seq_length
        for part_segments_index in range(part_segments.shape[0]):
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
    for segment in list(segments["Segment"].unique()):
        this_segment = segments.loc[segments["Segment"]==segment]
        indices = this_segment.index
        if this_segment["Process_encoded"].sum() == 0:
            segments.loc[indices, "Label"] = np.repeat(0, seq_length)
        elif this_segment["Process_encoded"].sum() == seq_length:
            segments.loc[indices, "Label"] = np.repeat(1, seq_length)
        elif this_segment.iloc[1, this_segment.columns.get_loc("Process_encoded")] == 0:
            segments.loc[indices, "Label"] = np.repeat(1, seq_length)
        else:
            segments.loc[indices, "Label"] = np.repeat(0, seq_length)
    segments.to_csv("data/ExperimentData_segments_"+str(int(seq_length))+".data", index=False)
    return segments
#


def create_segments_without_parts(data_df, seq_length):
    """
        Creates overlapping segments for learning
    """
    length = len(data_df) * seq_length - 2 * np.sum(np.arange(seq_length))
    segments = pd.DataFrame(index=pd.RangeIndex(length), columns=data_df.columns.values.tolist())
    segments["Segment"] = segments.index // seq_length
    segments["Label"] = None
    segment_column_location_in_tuple = segments.columns.get_loc("Segment") + 1
    for segments_index in range(segments.shape[0]):
        data_df_index = data_df.index[row[segment_column_location_in_tuple] + ( segments_index % seq_length )]
        segments.loc[segments_index, "dateTime"] = data_df.loc[data_df_index, "dateTime"]
        segments.loc[segments_index, "dataItemId"] = data_df.loc[data_df_index, "dataItemId"]
        segments.loc[segments_index, "value"] = data_df.loc[data_df_index, "value"]
        segments.loc[segments_index, "Part Number"] = data_df.loc[data_df_index, "Part Number"]
        segments.loc[segments_index, "Process"] = data_df.loc[data_df_index, "Process"]
        segments.loc[segments_index, "dateTime_value"] = data_df.loc[data_df_index, "dateTime_value"]
        segments.loc[segments_index, "Duration"] = data_df.loc[data_df_index, "Duration"]
        segments.loc[segments_index, "dataItemId_encoded"] = data_df.loc[data_df_index, "dataItemId_encoded"]
        segments.loc[segments_index, "Process_encoded"] = data_df.loc[data_df_index, "Process_encoded"]
        segments.loc[segments_index, "Train/Test"] = data_df.loc[data_df_index, "Train/Test"]
    segments.reset_index(drop=True, inplace=True)
    for segment in list(segments["Segment"].unique()):
        this_segment = segments.loc[segments["Segment"]==segment]
        indices = this_segment.index
        if this_segment["Process_encoded"].sum() == 0:
            segments.loc[indices, "Label"] = np.repeat(0, seq_length)
        elif this_segment["Process_encoded"].sum() == seq_length:
            segments.loc[indices, "Label"] = np.repeat(1, seq_length)
        elif this_segment.iloc[1, this_segment.columns.get_loc("Process_encoded")] == 0:
            segments.loc[indices, "Label"] = np.repeat(1, seq_length)
        else:
            segments.loc[indices, "Label"] = np.repeat(0, seq_length)
    segments.to_csv("data/ExperimentData_segments_withoutparts_"+str(int(seq_length))+".data", index=False)
    return segments
#


def create_arrayed_segments(seg_df, seq_length, num_classes):
    """ Slices the dataframes and returns an array of sequences for learning. """
    trn_seg_df = seg_df.loc[seg_df["Train/Test"]=="Train"]
    tst_seg_df = seg_df.loc[seg_df["Train/Test"]=="Test"]
    #
    trn_segments = list(trn_seg_df["Segment"].unique())
    tst_segments = list(tst_seg_df["Segment"].unique())
    #
    x_trn = trn_seg_df["dataItemId_encoded"].to_numpy().reshape(len(trn_segments), 1, seq_length)
    x_tst = tst_seg_df["dataItemId_encoded"].to_numpy().reshape(len(tst_segments), 1, seq_length)
    #
    y_trn = np_utils.to_categorical(trn_seg_df["Process_encoded"].to_numpy()[::seq_length].reshape(len(trn_segments), 1, 1), num_classes)
    y_tst = np_utils.to_categorical(tst_seg_df["Process_encoded"].to_numpy()[::seq_length].reshape(len(tst_segments), 1, 1), num_classes)
    #
    return x_trn, y_trn, x_tst, y_tst
#


def prep_data(sequence_length):
    """ Imports the data and takes care of its peculiarities. """
    print("Processing raw data...")
    data, encoders = load_data()
    print("Segmenting processed data...")
    try:
        #this takes too long so if it's been done before, I'm trying to skip it and just load the file
        segments = pd.read_csv("data/ExperimentData_segments_withoutparts_"+str(int(sequence_length))+".data", header=0)
        print("Previously segmented sequences found. Loading them instead or resegmenting...")
        # Check to make sure segmentation length is valid
        if segments.loc[segments["Segment"]==0].shape[0] != sequence_length:
            print("Sequence length is wrong. Re-segmenting...")
            raise ValueError
        segments["dateTime_value"] = pd.to_datetime(segments["dateTime"]) # Cannot be recovered once saved to csv
        segments["Duration"] = pd.to_timedelta(segments["Duration"]) # Cannot be recovered once saved to csv
        # encoding loses bitdepth. Is this a problem?
    except (FileNotFoundError, ValueError, KeyError):
        segments = create_segments_without_parts(data.loc[data["Train/Test"]=="Train"], sequence_length)
    num_classes = encoders["Process"].classes_.size
    x_trn, y_trn, x_tst, y_tst = create_arrayed_segments(segments, sequence_length, num_classes)
    num_features = x_trn.shape[1]
    labels = data["Process"].unique().tolist()
    #
    return x_trn, y_trn, x_tst, y_tst, num_features, num_classes, labels
#


def create_model(input_shape, num_classes):
    """ Creates the TF model """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(units=25, return_sequences=True))  # returns a sequence of vectors of dimension 32
    #MODEL.add(LSTM(30, return_sequences=True))  # return a single vector of dimension 32
    model.add(TimeDistributed(Dense(activation='tanh', units=100)))
    # model.add(Dense(units=100, activation="tanh"))
    model.add(Dropout(0.4))
    model.add(Dense(units=num_classes, activation='softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
#


def fit_model(model, x, y):
    """ Fits the model """
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    if not os.path.exists("models"):
        os.makedirs("models")
    callbacks = [
        ModelCheckpoint(filepath='models/machine_states_best_model.h5', monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='val_acc', patience=1) # if accuracy doesn't improve in 2 epochs, stop.
    ]
    return model.fit(x, y, batch_size=50, epochs=50, verbose=1, validation_split=0.2, callbacks=callbacks)
#


def plot_error_history(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['acc'], "g--", label="Accuracy of training data")
    plt.plot(history.history['val_acc'], "g", label="Accuracy of validation data")
    plt.plot(history.history['loss'], "r--", label="Loss of training data")
    plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
    plt.title('Model Accuracy and Loss')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.legend()
    plt.show()
#


def plot_confusion_matrix(matrix, labels):
    """ Plots a confusion matrix in the form of a heatmap """
    plt.figure(figsize=(6, 4))
    heatmap(matrix, cmap="coolwarm", linecolor='white', linewidths=1,
            xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()
#


def evaluate_model(model, labels, x, y):
    """ Evaluates model with inputs. Use it with test data. """
    # Calculate accuracy and loss:
    score = model.evaluate(x, y, verbose=0)
    print(f"Accuracy on the test data: {score[1]:0.2}. \nLoss on the test data: {score[0]:0.2}.")
    # Create confusion matrix:
    y_pred_max = np.argmax(model.predict(x), axis=1)
    y_max = np.argmax(y, axis=1)
    matrix = confusion_matrix(y_max, y_pred_max)
    plot_confusion_matrix(matrix, labels)
    # Create classifcation report:
    print("=====================\nClassification report:")
    print(classification_report(y_max, y_pred_max))
#


X_TRN, Y_TRN, X_TST, Y_TST, NUM_FEATURES, NUM_CLASSES, LABELS = prep_data(SEQUENCE_LENGTH)
TF_SESSION = start_tf_session()
if REFIT:
    MODEL = create_model(input_shape=(NUM_FEATURES, SEQUENCE_LENGTH), num_classes=NUM_CLASSES)
    HISTORY = fit_model(MODEL, X_TRN, Y_TRN)
    print("\nTraining finished.\n")
    plot_error_history(HISTORY)
else:
    try:
        MODEL = load_model("machine_states_best_model.h5")
    except FileNotFoundError:
        MODEL = create_model(input_shape=(NUM_FEATURES, SEQUENCE_LENGTH), num_classes=NUM_CLASSES)
        HISTORY = fit_model(MODEL, X_TRN, Y_TRN)
        print("\nTraining finished.\n")
        plot_error_history(HISTORY)
    #
evaluate_model(MODEL, LABELS, X_TST, Y_TST)