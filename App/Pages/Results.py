import streamlit as st
import pandas as pd
from io import StringIO
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
from google.oauth2 import service_account
from google.cloud import storage
import streamlit.components.v1 as components
from sklearn.preprocessing import OneHotEncoder
import mpld3
import keras
import autokeras as ak
import os
import io
from zipfile import ZipFile
from zipfile import is_zipfile

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


bucket_name = 'autolm_model_data'
file_name = 'heart_cleveland_upload_data_model.zip'
original_filename = file_name.replace('_data_model.zip', '')
destination_file = os.path.join('models/', original_filename)



#Downloading and unzipping the Model file

zipextract(bucket_name, file_name,destination_file)
model = keras.models.load_model(destination_file,custom_objects=ak.CUSTOM_OBJECTS)

# Getting testing dataset for prediction

x_test_data = StringIO(read_file('testing_data_automl', '{}_x_test.csv'.format(original_filename)))
x_test = pd.read_csv(x_test_data)

y_test_data = StringIO(read_file('testing_data_automl', '{}_y_test.csv'.format(original_filename)))
y_test = pd.read_csv(y_test_data)
# Prediction of the model results
y_pred = model.predict(x_test)

for i in range(len(y_pred)):
    if(y_pred[i]>0.5):
        y_pred[i] = 1
    else:
        y_pred[i] = 0

import scikitplot as skplt


#fig = plt.figure(figsize=(10,6))
y_pred = OneHotEncoder().fit_transform(pd.DataFrame(y_pred)).toarray()
fig = (y_test, y_pred)
st.pyplot(fig)
