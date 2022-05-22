import pandas as pd
from flask import jsonify
from google.cloud import storage
from sklearn.model_selection import train_test_split
from autokeras import StructuredDataClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


storage_client = storage.Client(project='auto-ml-and-explainable-ai')

def upload_blob(bucket_name, source_data, destination_blob_name):
    """Uploads a file to the bucket."""    
    print('function upload_blob called')     
    bucket = storage_client.get_bucket(bucket_name)    
    blob = bucket.blob(destination_blob_name)    
    blob.upload_from_string(source_data)    
    print('File {} uploaded to {}.'.format(destination_blob_name, bucket_name))

def auto_ml(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """

    request_json = request.get_json()

    # Getting data from feature_engineering_dataset bucket
    bucket_name = 'feature_engineering_dataset'
    bucket = storage_client.get_bucket(bucket_name)


    if request_json and 'file' in request_json:
        filename =  request_json['file']
        # Filename without extension
        file = filename.replace('.csv','')

        # Get CSV file
        csv_file = 'gs://{}/{}'.format(bucket,filename)
        
        # Convert to CSV to Dataframe
        df = pd.read_csv(csv_file)

        if 'target' in request_json:
            # Gettihg the target variable and generating X & Y Dataframe form it
            target = request_json['target']
            y = df[target]
            x = df.drop([target],axis= 1)

            # Train - Test Split
            x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

            # Using AutoML to generate Machine Learning model
            search=StructuredDataClassifier(max_trials=3, loss='accuracy')
            search.fit(x=x_train,y=y_train)

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
            class_rep = classification_report(y_pred, y_test)
            cm = str(confusion_matrix(y_pred, y_test))

            # Save Model to local file system
            model.save('model/{}_data_model.h5'.format(file))

            # Upload model to google cloud storage
            with open('model/{}_data_model.h5'.format(file)) as model_data:
                upload_blob('autolm_model_data',model_data,'{}_data_model.h5'.format(file))

            # Training datasets uploading to cloud storage
            xtrain_csv = x_train.to_csv()
            upload_blob('training_data_automl',xtrain_csv,'{}_x_train.csv'.format(file))

            ytrain_csv = y_train.to_csv()
            upload_blob('training_data_automl',ytrain_csv,'{}_y_train.csv'.format(file))

            # Testing datasets uploading to cloud storage
            xtest_csv = x_test.to_csv()
            upload_blob('testing_data_automl',xtest_csv,'{}_x_test.csv'.format(file))

            ytest_csv = y_test.to_csv()
            upload_blob('testing_data_automl',ytest_csv,'{}_y_test.csv'.format(file))

            return_dict = {
                'classification_report' : str(class_rep),
                'confusion_matrix' : str(cm)
            }

            return jsonify(return_dict)

        else:
            return f'Target Variable not given.'

    else:
        return f'File Name not found.'

    

