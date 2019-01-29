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
from keras.layers import LSTM, Dense, TimeDistributed, Conv1D, Dropout, GlobalAveragePooling1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
#
#from keras.layers import Dense, Dropout, Flatten, Reshape, 
#from keras.layers import Conv2D, MaxPooling2D, Conv1D, 
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


def create_segments_and_labels(df, time_steps, step, label_name):
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
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['x'].values[i: i + time_steps]
        ys = df['y'].values[i: i + time_steps]
        zs = df['z'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = stats.mode(df[label_name][i: i + time_steps])[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)
    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, 3)
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

DATA = read_data()
LABELS = ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"]
TIME_PERIODS = 100
STEP_DISTANCE = 50
LABEL = "EncodedActivity"
LABEL_ENCODER = preprocessing.LabelEncoder()
DATA[LABEL] = LABEL_ENCODER.fit_transform(DATA["Activity"].values.ravel())
TRAIN_DATA = DATA[DATA['ID'] <= 28]
TEST_DATA = DATA[DATA['ID'] > 28]
# Normalize features for training data set
# df_train['x-axis'] = feature_normalize(df['x-axis'])
# df_train['y-axis'] = feature_normalize(df['y-axis'])
# df_train['z-axis'] = feature_normalize(df['z-axis'])

# Reshape the training data into segments
# so that they can be processed by the network
X_TRAIN, Y_TRAIN = create_segments_and_labels(TRAIN_DATA, TIME_PERIODS, STEP_DISTANCE, LABEL)
NUM_TIME_PERIODS, NUM_FEATURES = X_TRAIN.shape[1], X_TRAIN.shape[2]
NUM_CLASSES = LABEL_ENCODER.classes_.size
Y_TRAIN = np_utils.to_categorical(Y_TRAIN, NUM_CLASSES)
# Convert type for Keras otherwise Keras cannot process the data
# x_train = x_train.astype("float32")
# y_train = y_train.astype("float32")
MODEL = Sequential()
# MODEL.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
MODEL.add(Conv1D(100, 10, activation='tanh', input_shape=(TIME_PERIODS, NUM_FEATURES)))
MODEL.add(Conv1D(100, 5, activation='tanh'))
MODEL.add(MaxPooling1D(3))
MODEL.add(Conv1D(200, 10, activation='tanh'))
MODEL.add(Conv1D(200, 5, activation='tanh'))
MODEL.add(GlobalAveragePooling1D())
MODEL.add(Dropout(0.1))
MODEL.add(Dense(NUM_CLASSES, activation='softmax'))
print(MODEL.summary())
CALLBACKS_LIST = [
    ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True),
    # ModelCheckpoint(filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=True),
    EarlyStopping(monitor='acc', patience=1) # if accuracy doesn't improve in 2 epochs, stop.
]
MODEL.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.85))):
    HISTORY = MODEL.fit(X_TRAIN, Y_TRAIN, batch_size=25, epochs=50, callbacks=CALLBACKS_LIST, validation_split=0.2, verbose=1)
plot_error_history(HISTORY)
#
MODEL = load_model("best_model.h5")
X_TEST, Y_TEST = create_segments_and_labels(TEST_DATA, TIME_PERIODS, STEP_DISTANCE, LABEL)

Y_TEST = np_utils.to_categorical(Y_TEST, NUM_CLASSES)
SCORE = MODEL.evaluate(X_TEST, Y_TEST, verbose=1)
print("\nAccuracy on test data: %0.2f" % SCORE[1])
print("\nLoss on test data: %0.2f" % SCORE[0])
Y_PRED_TEST = MODEL.predict(X_TEST)
# Take the class with the highest probability from the test predictions
MAX_Y_PRED_TEST = np.argmax(Y_PRED_TEST, axis=1)
MAX_Y_TEST = np.argmax(Y_TEST, axis=1)
show_confusion_matrix(MAX_Y_TEST, MAX_Y_PRED_TEST)
print(classification_report(MAX_Y_TEST, MAX_Y_PRED_TEST))
