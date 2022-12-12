import streamlit as st
import pandas as pd
from io import StringIO
import keras
import autokeras as ak
import os
from google.oauth2 import service_account
from google.cloud import storage
import streamlit.components.v1 as components
import numpy as np
import lime
import lime.lime_tabular
from lime import submodular_pick


file_name = 'wisc_bc_data_data_model.zip'
original_filename = file_name.replace('_data_model.zip', '')
destination_file = os.path.join('models/', original_filename)

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)

@st.experimental_memo(ttl=600)
def read_file(bucket_name, file_path):
    bucket = client.bucket(bucket_name)
    content = bucket.blob(file_path).download_as_string().decode("utf-8")
    return content


model = keras.models.load_model(destination_file,compile = False,custom_objects=ak.CUSTOM_OBJECTS)

x_train_data = StringIO(read_file('training_data_automl', '{}_x_train.csv'.format(original_filename)))
x_train = pd.read_csv(x_train_data)

x_test_data = StringIO(read_file('testing_data_automl', '{}_x_test.csv'.format(original_filename)))
x_test = pd.read_csv(x_test_data)

y_train_data = StringIO(read_file('training_data_automl', '{}_y_train.csv'.format(original_filename)))
y_train = pd.read_csv(y_train_data)

y_test_data = StringIO(read_file('testing_data_automl', '{}_y_test.csv'.format(original_filename)))
y_test = pd.read_csv(y_test_data)
# Output class names
class_names = list(set(y_train))

# Creating lime tabular and SP-LIME objects
# SP-LIME returns exaplanations on a sample set to provide a non redundant global decision boundary of original model
explainer = lime.lime_tabular.LimeTabularExplainer(np.array(x_train), feature_names=list(x_train), class_names= class_names, mode='classification')

# Build app
title_text = 'AutoHealthX: LIME Explainer'
subheader_text = '''1: Strongly Negative &nbsp 2: Weakly Negative &nbsp  3: Neutral &nbsp  4: Weakly Positive &nbsp  5: Strongly Positive'''

st.markdown(f"<h2 style='text-align: center;'><b>{title_text}</b></h2>", unsafe_allow_html=True)


sample = st.selectbox(
    'Choose test sample',
    x_test.index,
)

exp = explainer.explain_instance(np.array(x_test)[sample], model.predict, top_labels=1)
components.html(exp.as_html(), height=800)

# st.markdown(f"<h5 style='text-align: center;'>{subheader_text}</h5>", unsafe_allow_html=True)

# lime_data = StringIO(read_file('model_explanations', '{}_lime_final.html'.format(original_filename)))
# components.html(lime_data, height=800)


