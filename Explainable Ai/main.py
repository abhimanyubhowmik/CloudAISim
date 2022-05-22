import os
import lime
import lime.lime_tabular
from lime import submodular_pick
import numpy as np
import pandas as pd
import autokeras as ak
from tensorflow import keras
from google.cloud import storage

storage_client = storage.Client(project='auto-ml-and-explainable-ai')

def upload_blob_txt(bucket_name, source_data, destination_blob_name):
    """Uploads a text file to the bucket."""    
    print('function upload_blob_txt called')     
    bucket = storage_client.get_bucket(bucket_name)    
    blob = bucket.blob(destination_blob_name)    
    blob.upload_from_string(source_data)  
    print('File {} uploaded to {}.'.format(destination_blob_name, bucket_name))


def explainable_ai(event, context):
    """Triggered by a change to a Cloud Storage bucket.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    file = event
    filename = file['name']
    bucket = file['bucket']
    print(f"Processing file: {file['name']}.")

    original_filename = filename.replace('_data_model.h5', '')

    # Getting the ML Model
    model_file = 'gs://{}/{}'.format(bucket,filename)
    model = keras.models.load_model(model_file,custom_objects=ak.CUSTOM_OBJECTS)

    # Detting training and testing dataset for prediction
    x_train_data = 'gs://{}/{}'.format('training_data_automl','{}_x_train.csv'.format(original_filename))
    x_train = pd.read_csv(x_train_data)

    x_test_data = 'gs://{}/{}'.format('testing_data_automl','{}_x_test.csv'.format(original_filename))
    x_test = pd.read_csv(x_test_data)

    y_train_data = 'gs://{}/{}'.format('training_data_automl','{}_y_train.csv'.format(original_filename))
    y_train = pd.read_csv(y_train_data)

    # Output class names
    class_names = list(set(y_train))

    # Creating lime tabular and SP-LIME objects
    # SP-LIME returns exaplanations on a sample set to provide a non redundant global decision boundary of original model
    explainer = lime.lime_tabular.LimeTabularExplainer(np.array(x_train), feature_names=list(x_train), class_names= class_names, mode='classification')
    sp_obj = submodular_pick.SubmodularPick(explainer, np.array(x_test), model.predict, sample_size=20, num_exps_desired=5)

    # Saving the HTML file to local storage
    os.chdir('/tmp')
    os.mkdir('lime')
    for i in range(len(sp_obj.sp_explanations)):

        sp_obj.sp_explanations[i].save_to_file('lime/lime{}.html'.format(i))

    tot_data = ''

    for filename in os.listdir('lime'):
        with open(os.path.join('lime', filename), 'r') as f:
            data = f.read()
            tot_data += data + '\n'

    # Upload model explanations html file to cloud storage
    upload_blob_txt('model_explainations',tot_data,'{}_lime_final.html'.format(original_filename))


