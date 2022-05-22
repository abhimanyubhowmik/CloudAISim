import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from autokeras import StructuredDataClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


storage_client = storage.Client(project='auto-ml-and-explainable-ai')

def upload_blob(bucket_name, file, destination_blob_name):
    """Uploads any file to the bucket."""    
    print('function upload_blob called')     
    bucket = storage_client.get_bucket(bucket_name)    
    blob = bucket.blob(destination_blob_name)    
    blob.upload_from_filename(file)    
    print('File {} uploaded to {}.'.format(destination_blob_name, bucket_name))

def upload_blob_txt(bucket_name, source_data, destination_blob_name):
    """Uploads a text file to the bucket."""    
    print('function upload_blob_txt called')     
    bucket = storage_client.get_bucket(bucket_name)    
    blob = bucket.blob(destination_blob_name)    
    blob.upload_from_string(source_data)  
    print('File {} uploaded to {}.'.format(destination_blob_name, bucket_name))


def auto_ml(event, context):
    """Triggered by a change to a Cloud Storage bucket.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    file = event
    filename = file['name']
    bucket = file['bucket']
    filename_without_extension = filename.replace('.csv', '')
    print(f"Processing file: {file['name']}.")

    csv_file = 'gs://{}/{}'.format(bucket,filename)

    df = pd.read_csv(csv_file)

    y = df.iloc[:,-1]
    x = df.iloc[:,:-1]

    # Train - Test Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    os.chdir('/tmp')

    # Using AutoML to generate Machine Learning model
    search=StructuredDataClassifier(max_trials=3, loss='accuracy')
    search.fit(x=x_train,y=y_train,epochs=10)

    # Final Model
    model = search.export_model()

    # Prediction of the model results
    y_pred = model.predict(x_test)

    for i in range(len(y_pred)):
        if(y_pred[i]>0.5):
            y_pred[i] = 1
        else:
            y_pred[i] = 0

    # Classification report and confusion_matrix
    class_rep = pd.DataFrame.from_dict(classification_report(y_pred, y_test,output_dict=True)).to_csv()
    cm = pd.DataFrame(confusion_matrix(y_pred, y_test)).to_csv()

    # Uploading of classification report and confusion matrix to cloud storage
    upload_blob_txt('classification_reports',class_rep,'{}_classification_report.csv'.format(filename_without_extension))
    upload_blob_txt('confusion_matrices',cm,'{}_confusion_matrix.csv'.format(filename_without_extension))
    
    # Training datasets uploading to cloud storage
    xtrain_csv = x_train.to_csv(index=False)
    upload_blob_txt('training_data_automl',xtrain_csv,'{}_x_train.csv'.format(filename_without_extension))

    ytrain_csv = y_train.to_csv(index=False)
    upload_blob_txt('training_data_automl',ytrain_csv,'{}_y_train.csv'.format(filename_without_extension))

    # Testing datasets uploading to cloud storage
    xtest_csv = x_test.to_csv(index=False)
    upload_blob_txt('testing_data_automl',xtest_csv,'{}_x_test.csv'.format(filename_without_extension))

    ytest_csv = y_test.to_csv(index=False)
    upload_blob_txt('testing_data_automl',ytest_csv,'{}_y_test.csv'.format(filename_without_extension))

    # Save Model to local file system
    model.save('model/{}_data_model.h5'.format(filename_without_extension))

    # Upload model to google cloud storage
    upload_blob('autolm_model_data','model/{}_data_model.h5'.format(filename_without_extension),'{}_data_model.h5'.format(filename_without_extension))

