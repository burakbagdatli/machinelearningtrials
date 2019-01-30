"""
https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
"""
from __future__ import print_function
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, Dropout, GlobalAveragePooling1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
#


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu)/sigma
#


def prep_data():
    """ Imports the data and takes care of its peculiarities."""
    column_names = ["Signal ID", "Activity", "Time", "x", "y", "z"]
    df = pd.read_csv("data/WISDM_ar_v1.1_raw.data", header=None, names=column_names)
    df["z"].replace(regex=True, inplace=True, to_replace=";", value="")
    df["z"] = pd.to_numeric(df["z"], errors="coerce")
    df.dropna(axis=0, how="any", inplace=True)
    #
    df["x norm"] = feature_normalize(df["x"])
    df["y norm"] = feature_normalize(df["y"])
    df["z norm"] = feature_normalize(df["z"])
    #
    signal_ids = df["Signal ID"].unique()
    np.random.shuffle(signal_ids)
    training_cutoff = np.rint(signal_ids.size * 0.75).astype(int)
    training_signal_ids = signal_ids[:training_cutoff]
    df["Train/Test"] = "Test"
    df.loc[df["Signal ID"].isin(training_signal_ids), "Train/Test"] = "Train"
    #
    labels = df["Activity"].unique().tolist()
    encoder = preprocessing.LabelEncoder()
    df["Encoded Activity"] = encoder.fit_transform(df["Activity"].values.ravel())
    return df, labels, encoder
#


def create_model(input_shape):
    """ Creates the TF model """
    model = Sequential()
    model.add(Conv1D(100, 10, activation='tanh', input_shape=input_shape))
    model.add(Conv1D(100, 5, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(200, 10, activation='tanh'))
    model.add(Conv1D(200, 5, activation='tanh'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.1))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_segments_and_labels(data_df, time_steps, step, train_test):
    """
    This function receives a dataframe and returns the reshaped segments
    of x, y, z acceleration as well as the corresponding labels
    Args:
        df: Dataframe in the expected format
        time_steps: Integer value of the length of a segment that is created
    Returns:
        reshaped_segmentS
        labels:
    """
    df = data_df.loc[DATA["Train/Test"]==train_test, ["x norm", "y norm", "z norm", "Encoded Activity"]]
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['x norm'].values[i: i + time_steps]
        ys = df['y norm'].values[i: i + time_steps]
        zs = df['z norm'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment.
        # This is questionable
        label = stats.mode(df["Encoded Activity"][i: i + time_steps])[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)
    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype=np.float16).reshape(-1, time_steps, 3)
    labels = np.asarray(labels)
    return reshaped_segments, labels
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


def show_confusion_matrix(validations, predictions):
    """ Plots a confusion matrix in the form of a heatmap """
    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, cmap="coolwarm", linecolor='white', linewidths=1, 
                xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()
#

DATA, LABELS, ENCODER = prep_data()
# LABELS = ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"]
TIME_PERIODS = 100
STEP_DISTANCE = 50
#
X_TRAIN, Y_TRAIN = create_segments_and_labels(DATA, TIME_PERIODS, STEP_DISTANCE, "Train")
NUM_TIME_PERIODS, NUM_FEATURES = X_TRAIN.shape[1], X_TRAIN.shape[2]
NUM_CLASSES = ENCODER.classes_.size
Y_TRAIN = np_utils.to_categorical(Y_TRAIN, NUM_CLASSES)
#
MODEL = create_model(input_shape=(TIME_PERIODS, NUM_FEATURES))
with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.85))):
    CALLBACKS = [
        ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='acc', patience=1) # if accuracy doesn't improve in 2 epochs, stop.
    ]
    HISTORY = MODEL.fit(X_TRAIN, Y_TRAIN, batch_size=100, epochs=50, verbose=1, validation_split=0.2,
                        callbacks=CALLBACKS)
plot_error_history(HISTORY)
#
MODEL = load_model("best_model.h5")
#
X_TEST, Y_TEST = create_segments_and_labels(DATA, TIME_PERIODS, STEP_DISTANCE, "Test")
Y_TEST = np_utils.to_categorical(Y_TEST, NUM_CLASSES)
SCORE = MODEL.evaluate(X_TEST, Y_TEST, verbose=0)
print("\nAccuracy on test data: %0.2f" % SCORE[1])
print("\nLoss on test data: %0.2f" % SCORE[0])
Y_PRED_TEST = MODEL.predict(X_TEST)
# Take the class with the highest probability from the test predictions
MAX_Y_PRED_TEST = np.argmax(Y_PRED_TEST, axis=1)
MAX_Y_TEST = np.argmax(Y_TEST, axis=1)
show_confusion_matrix(MAX_Y_TEST, MAX_Y_PRED_TEST)
print(classification_report(MAX_Y_TEST, MAX_Y_PRED_TEST))
