"""
Lots of inspiration was taken from here:
https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from seaborn import heatmap
from scipy.stats import mode
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import RobustScaler, LabelEncoder
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, Dropout, GlobalAveragePooling1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
#


def start_tf_session():
    """ Starts a usable tensorflow session. """
    config = None
    if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
        # This may be unnecessary. On my computer cuDNN fails to initialize when TF asks for too much GPU memory.
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.85))
    return tf.Session(config=config)
#


def create_segments_and_labels(data_df, time_steps, step, train_test):
    """
        I modified this only very slightly from the source.
        Source: https://github.com/ni79ls/har-keras-cnn
    """
    df = data_df.loc[data_df["Train/Test"]==train_test, ["x scaled", "y scaled", "z scaled", "Encoded Activity"]]
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['x scaled'].values[i: i+time_steps]
        ys = df['y scaled'].values[i: i+time_steps]
        zs = df['z scaled'].values[i: i+time_steps]
        # Retrieve the most often used label in this segment.
        # This is questionable but it makes some sense
        # If we happen to sample the signal when the user is switching from walking to running,
        # we still want a valid label: the majority.
        label = mode(df["Encoded Activity"][i: i + time_steps])[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)
    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype=np.float16).reshape(-1, time_steps, 3)
    labels = np.asarray(labels)
    return reshaped_segments, labels
#


def prep_data(time_periods, step_distance):
    """ Imports the data and takes care of its peculiarities. """
    column_names = ["Signal ID", "Activity", "Time", "x", "y", "z"]
    df = pd.read_csv("data/WISDM_ar_v1.1_raw.data", header=None, names=column_names)
    df["z"].replace(regex=True, inplace=True, to_replace=";", value="")
    df["z"] = pd.to_numeric(df["z"], errors="coerce")
    df.dropna(axis=0, how="any", inplace=True)
    #
    scaler = RobustScaler()  # There are other options.
    df["x scaled"], df["y scaled"], df["z scaled"] = None, None, None
    df[["x scaled", "y scaled", "z scaled"]] = scaler.fit_transform(df[["x", "y", "z"]])
    #
    labels = df["Activity"].unique().tolist()
    encoder = LabelEncoder()
    df["Encoded Activity"] = encoder.fit_transform(df["Activity"].values.ravel())
    #
    signal_ids = df["Signal ID"].unique()
    np.random.shuffle(signal_ids)
    training_cutoff = np.rint(signal_ids.size * 0.75).astype(int)
    training_signal_ids = signal_ids[:training_cutoff]
    df["Train/Test"] = "Test"
    df.loc[df["Signal ID"].isin(training_signal_ids), "Train/Test"] = "Train"
    #
    x_train, y_train = create_segments_and_labels(df, time_periods, step_distance, "Train")
    num_features = x_train.shape[2]
    num_classes = encoder.classes_.size
    y_train = np_utils.to_categorical(y_train, num_classes)
    x_test, y_test = create_segments_and_labels(df, time_periods, step_distance, "Test")
    y_test = np_utils.to_categorical(y_test, num_classes)
    #
    return x_train, y_train, x_test, y_test, num_features, num_classes, labels
#


def create_model(input_shape, num_classes):
    """ Creates the TF model """
    model = Sequential()
    model.add(Conv1D(50, 16, activation='tanh', input_shape=input_shape))  # 100 in 85 out
    model.add(Conv1D(50, 8, activation='tanh'))  # 85 in 78 out
    model.add(MaxPooling1D(3))  # 78 in 26 out
    model.add(Conv1D(25, 8, activation='tanh'))  # 26 in 19 out
    model.add(Conv1D(25, 4, activation='tanh'))  # 19 in 16 out
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
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
        ModelCheckpoint(filepath='models/best_model.h5', monitor='val_loss', save_best_only=True),
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
    score, accuracy = model.evaluate(x, y, verbose=0)
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

# Define execution settings:
REFIT = True
TIME_PERIODS = 100
STEP_DISTANCE = 50
# Prepare data:
X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, NUM_FEATURES, NUM_CLASSES, LABELS = prep_data(TIME_PERIODS, STEP_DISTANCE)
# Create/fit/load model:
TF_SESSION = start_tf_session()
if REFIT:
    MODEL = create_model(input_shape=(TIME_PERIODS, NUM_FEATURES), num_classes=NUM_CLASSES)
    HISTORY = fit_model(MODEL, X_TRAIN, Y_TRAIN)
    print("\nTraining finished.\n")
    plot_error_history(HISTORY)
else:
    MODEL = load_model("best_model.h5")
# Evaluate model:
evaluate_model(MODEL, LABELS, X_TEST, Y_TEST)
