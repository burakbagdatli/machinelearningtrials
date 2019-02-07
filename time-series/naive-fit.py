import os
import numpy as np
import pandas as pd  # Requires pandas 0.24
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from matplotlib import pyplot as plt
from seaborn import lmplot
#


def start_tf_session():
    """ Starts a usable tensorflow session. """
    config = None
    if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
        # This may be unnecessary. On my computer cuDNN fails to initialize when TF asks for too much GPU memory.
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.85))
    return tf.Session(config=config)
#


def prep_data():
    """ Loads the data and transforms it into a form that the model can ingest. """
    column_names = ["dew", "time", "m", "rpm"]
    df = pd.read_csv("data/DEW.data", header=0, names=column_names)
    df.dropna(axis=0, how="any", inplace=True)
    #
    scaler = RobustScaler()  # There are other options.
    df["dew scaled"], df["time scaled"], df["m scaled"] = None, None, None
    df[["dew scaled", "time scaled", "m scaled"]] = scaler.fit_transform(df.loc[:,["dew", "time", "m"]])
    # Here's how I'm splitting the data:
    # The data is quite linear wrt dew, but very nonlinear in time
    # So, I'm taking entire time sequences for learning
    # And, that means I need some entire sequences for validation
    dews = df["dew"].unique()
    np.random.shuffle(dews)
    training_cutoff = np.rint(dews.size * 0.75).astype(int)
    training_dews = dews[:training_cutoff]
    df["Training/Validation"] = "Validation"
    df.loc[df["dew"].isin(training_dews), "Training/Validation"] = "Training"
    #
    x_train = df.loc[df["Training/Validation"]=="Training", ["dew scaled", "time scaled"]].to_numpy(copy=True)
    y_train = df.loc[df["Training/Validation"]=="Training", ["m scaled"]].to_numpy(copy=True)
    x_val = df.loc[df["Training/Validation"]=="Validation", ["dew scaled", "time scaled"]].to_numpy(copy=True)
    y_val = df.loc[df["Training/Validation"]=="Validation", ["m scaled"]].to_numpy(copy=True)
    input_shape = x_train.shape[1]
    #
    return x_train, y_train, x_val, y_val, (input_shape,), scaler
#


def create_model(input_shape):
    """ Creates the TF model """
    model = Sequential()
    model.add(Dense(200, activation="tanh", input_shape=input_shape))
    model.add(Dense(100, activation="tanh"))
    model.add(Dense(50, activation="tanh"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    print(model.summary())
    model.compile(loss='mse', optimizer='adam', metrics=["mae"])
    return model
#


def fit_model(model, x_trn, y_trn, x_val, y_val):
    """ Trains the model weights to fit the data. """
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    if not os.path.exists("models"):
        os.makedirs("models")
    callbacks = [
        ModelCheckpoint(filepath='models/naive-fit-best_model.h5', monitor='loss', save_best_only=True),
        EarlyStopping(monitor='loss', patience=14) # if loss doesn't improve in 15 epochs, stop.
    ]
    return model.fit(x=x_trn, y=y_trn, validation_data=(x_val, y_val), epochs=1000, verbose=1, callbacks=callbacks)
#


def plot_error_history(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['mean_absolute_error'], "g--", label="Mean absolute error of training data")
    plt.plot(history.history['val_mean_absolute_error'], "g", label="Mean absolute error of validation data")
    plt.plot(history.history['loss'], "r--", label="Mean squared error of training data")
    plt.plot(history.history['val_loss'], "r", label="Mean squared error of validation data")
    plt.title('Model Error')
    plt.ylabel('Errors')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.legend()
    plt.show()
#


def evaluate_model(model, x_trn, y_trn, x_val, y_val, scaler):
    """ Evaluates model with inputs. Use it with test data. """
    # Calculate accuracy and loss:
    score_trn = model.evaluate(x_trn, y_trn, verbose=0)
    print(f"Mean absolute error on the train data: {score_trn[1]:0.2}. \nMean squared error on the train data: {score_trn[0]:0.2}.")
    score_val = model.evaluate(x_val, y_val, verbose=0)
    print(f"Mean absolute error on the validation data: {score_val[1]:0.2}. \nMean square error on the validation data: {score_val[0]:0.2}.")
    predictions_trn = model.predict(x_trn, verbose=0)
    predictions_val = model.predict(x_val, verbose=0)
    # Collect data for rescaling:
    df_trn = pd.DataFrame(data={"dew":x_trn[:,0], "time":x_trn[:,1], "m":y_trn[:,0]})
    df_val = pd.DataFrame(data={"dew":x_val[:,0], "time":x_val[:,1], "m":y_val[:,0]})
    df_trn_pred = pd.DataFrame(data={"dew":x_trn[:,0], "time":x_trn[:,1], "m_pred":predictions_trn[:,0]})
    df_val_pred = pd.DataFrame(data={"dew":x_val[:,0], "time":x_val[:,1], "m_pred":predictions_val[:,0]})
    # Inverse transform:
    df_trn = pd.DataFrame(scaler.inverse_transform(df_trn), columns=["dew", "time", "m"])
    df_val = pd.DataFrame(scaler.inverse_transform(df_val), columns=["dew", "time", "m"])
    df_trn_pred = pd.DataFrame(scaler.inverse_transform(df_trn_pred), columns=["dew", "time", "m_pred"])
    df_val_pred = pd.DataFrame(scaler.inverse_transform(df_val_pred), columns=["dew", "time", "m_pred"])
    #
    df_trn = pd.concat([df_trn, df_trn_pred["m_pred"]], axis=1)
    df_val = pd.concat([df_val, df_val_pred["m_pred"]], axis=1)
    df_trn["Validation"] = "Training"
    df_val["Validation"] = "Validation"
    df = pd.concat([df_trn, df_val], axis=0, ignore_index=True)
    #
    lmplot(x="m", y="m_pred", hue="Validation", data=df)
    plt.show()
    return None
#


# Define execution settings:
REFIT = True
# Prepare data:
X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, INPUT_SHAPE, SCALER = prep_data()
# Create/fit/load model:
TF_SESSION = start_tf_session()
if REFIT:
    MODEL = create_model(input_shape=INPUT_SHAPE)
    HISTORY = fit_model(MODEL, X_TRAIN, Y_TRAIN, X_VAL, Y_VAL)
    print("\nTraining finished.\n")
    plot_error_history(HISTORY)
else:
    MODEL = load_model("models/naive-fit-best_model.h5")
# Evaluate model:
evaluate_model(MODEL, X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, SCALER)
