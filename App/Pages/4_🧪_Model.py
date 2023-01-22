# import streamlit as st
# from streamlit_tensorboard import st_tensorboard
# import tensorflow as tf
# import shutil
# import datetime
# import random

# import numpy as np
# import pandas as pd
# import tensorflow as tf

# import autokeras as ak
# from tensorflow import keras

# TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
# TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

# train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
# test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

# #logdir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# logdir = "logs/fit/" + time

# tensorboard_callback_train = keras.callbacks.TensorBoard(log_dir=logdir)
# tensorboard_callback_test = keras.callbacks.TensorBoard(log_dir=logdir)
# # x_train as pandas.DataFrame, y_train as pandas.Series
# x_train = pd.read_csv(train_file_path)
# print(type(x_train))  # pandas.DataFrame
# y_train = x_train.pop("survived")
# print(type(y_train))  # pandas.Series

# # You can also use pandas.DataFrame for y_train.
# y_train = pd.DataFrame(y_train)
# print(type(y_train))  # pandas.DataFrame

# # You can also use numpy.ndarray for x_train and y_train.
# x_train = x_train.to_numpy()
# y_train = y_train.to_numpy()
# print(type(x_train))  # numpy.ndarray
# print(type(y_train))  # numpy.ndarray

# # Preparing testing data.
# x_test = pd.read_csv(test_file_path)
# y_test = x_test.pop("survived")

# # It tries 10 different models.
# clf = ak.StructuredDataClassifier(overwrite=True, max_trials=5)
# # Feed the structured data classifier with training data.
# clf.fit(x_train, y_train, epochs=10, callbacks=[tensorboard_callback_train])
# clf.fit(x_test, y_test, epochs=10, callbacks=[tensorboard_callback_test])
# # Predict with the best model.
# #predicted_y = clf.predict(test_file_path)
# # Evaluate the best model with testing data.
# #print(clf.evaluate(test_file_path, "survived"))

# # Save Logs as zip file
# output_filename = 'logs/{}_log_output'.format(time)
# shutil.make_archive(output_filename, 'zip', logdir)


# # Start TensorBoard
# st_tensorboard(logdir=logdir, port=6006, width=1080)

##-------------------------------------------------------------------------------------------------##


import streamlit as st
import pandas as pd
import io
import os
from io import StringIO
from zipfile import ZipFile
from zipfile import is_zipfile
from google.oauth2 import service_account
from google.cloud import storage
from streamlit_tensorboard import st_tensorboard

st.set_page_config(layout="wide")

title_text = 'AutoHealthX: AutoML Model'
st.markdown(f"<h2 style='text-align: center;'><b>{title_text}</b></h2>", unsafe_allow_html=True)

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)




@st.experimental_memo(ttl=600)
def read_file(bucket_name, file_path):
    bucket = client.bucket(bucket_name)
    content = bucket.blob(file_path).download_as_string().decode("utf-8")
    return content

@st.experimental_memo(ttl=600)
def zipextract(bucketname, zipfilename, destination):
    if not os.path.exists(destination):
        os.mkdir(destination)
    bucket = client.get_bucket(bucketname)
    
    blob = bucket.blob(zipfilename)
    zipbytes = io.BytesIO(blob.download_as_string())

    if is_zipfile(zipbytes):
        with ZipFile(zipbytes, 'r') as zip_ref:
             zip_ref.extractall(destination)
    else:
        raise Exception("Not a zipfile.")

bucket_name = 'model_tf_logs'
file_name = 'wisc_bc_data_log_output.zip'
original_filename = file_name.replace('_data_model.zip', '')
destination_file = os.path.join('logs/', original_filename)

zipextract(bucket_name, file_name,destination_file)

st_tensorboard(logdir=destination_file, port=6006, width=1080)



