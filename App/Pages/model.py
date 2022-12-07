import streamlit as st
from streamlit_tensorboard import st_tensorboard
import tensorflow as tf

import datetime
import random

import numpy as np
import pandas as pd
import tensorflow as tf

import autokeras as ak
from tensorflow import keras

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

logdir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback_train = keras.callbacks.TensorBoard(log_dir=logdir)
tensorboard_callback_test = keras.callbacks.TensorBoard(log_dir=logdir)
# x_train as pandas.DataFrame, y_train as pandas.Series
x_train = pd.read_csv(train_file_path)
print(type(x_train))  # pandas.DataFrame
y_train = x_train.pop("survived")
print(type(y_train))  # pandas.Series

# You can also use pandas.DataFrame for y_train.
y_train = pd.DataFrame(y_train)
print(type(y_train))  # pandas.DataFrame

# You can also use numpy.ndarray for x_train and y_train.
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
print(type(x_train))  # numpy.ndarray
print(type(y_train))  # numpy.ndarray

# Preparing testing data.
x_test = pd.read_csv(test_file_path)
y_test = x_test.pop("survived")

# It tries 10 different models.
clf = ak.StructuredDataClassifier(overwrite=True, max_trials=5)
# Feed the structured data classifier with training data.
clf.fit(x_train, y_train, epochs=10, callbacks=[tensorboard_callback_train])
clf.fit(x_test, y_test, epochs=10, callbacks=[tensorboard_callback_test])
# Predict with the best model.
#predicted_y = clf.predict(test_file_path)
# Evaluate the best model with testing data.
#print(clf.evaluate(test_file_path, "survived"))


# Start TensorBoard
st_tensorboard(logdir=logdir, port=6006, width=1080)

