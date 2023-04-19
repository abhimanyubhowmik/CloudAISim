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
from sklearn.metrics import roc_curve, auc
import plotly.graph_objs as go
import scikitplot as skplt

import numpy as np
from itertools import cycle

from sklearn.metrics import roc_curve, auc
from scipy import interp

# Plot functions Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.metrics import precision_score, roc_auc_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import warnings
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

st.set_page_config(layout="wide")


credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)

@st.cache_data(ttl=600)
def read_file(bucket_name, file_path):
    bucket = client.bucket(bucket_name)
    content = bucket.blob(file_path).download_as_string().decode("utf-8")
    return content

@st.cache_data(ttl=600)
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
file_name = 'wisc_bc_data_data_model.zip'
original_filename = file_name.replace('_data_model.zip', '')
destination_file = os.path.join('models/', original_filename)



#Downloading and unzipping the Model file

zipextract(bucket_name, file_name,destination_file)
model = keras.models.load_model(destination_file,compile = False,custom_objects=ak.CUSTOM_OBJECTS)

# Getting testing dataset for prediction

x_test_data = StringIO(read_file('testing_data_automl', '{}_x_test.csv'.format(original_filename)))
x_test = pd.read_csv(x_test_data)

y_test_data = StringIO(read_file('testing_data_automl', '{}_y_test.csv'.format(original_filename)))
y_test = pd.read_csv(y_test_data)

##-----------------------------------------------------------------------------------------------------------------------##
# Prediction of the model results
# y_pred = model.predict(x_test)

# for i in range(len(y_pred)):
#     if(y_pred[i]>0.5):
#         y_pred[i] = 1
#     else:
#         y_pred[i] = 0

# y_test = OneHotEncoder().fit_transform(pd.DataFrame(y_test)).toarray()
# y_pred = OneHotEncoder().fit_transform(pd.DataFrame(y_pred)).toarray()

# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# n_classes = y_test.shape[1]
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# lw = 2

# # Compute macro-average ROC curve and ROC area

# # First aggregate all false positive rates
# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# # Then interpolate all ROC curves at this points
# mean_tpr = np.zeros_like(all_fpr)
# for i in range(n_classes):
#     mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# # Finally average it and compute AUC
# mean_tpr /= n_classes

# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# # Plot all ROC curves
# data = []
# trace1 = go.Scatter(x=fpr["micro"], y=tpr["micro"],
#                     mode='lines', 
#                     line=dict(color='deeppink', width=lw, dash='dot'),
#                     name='micro-average ROC curve (area = {0:0.2f})'
#                            ''.format(roc_auc["micro"]))
# data.append(trace1)

# trace2 = go.Scatter(x=fpr["macro"], y=tpr["macro"],
#                     mode='lines', 
#                     line=dict(color='navy', width=lw, dash='dot'),
#                     name='macro-average ROC curve (area = {0:0.2f})'
#                           ''.format(roc_auc["macro"]))
# data.append(trace2)

# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# for i, color in zip(range(n_classes), colors):
#     trace3 = go.Scatter(x=fpr[i], y=tpr[i],
#                         mode='lines', 
#                         line=dict(color=color, width=lw),
#                         name='ROC curve of class {0} (area = {1:0.2f})'
#                         ''.format(i, roc_auc[i]))
#     data.append(trace3)

# trace4 = go.Scatter(x=[0, 1], y=[0, 1], 
#                     mode='lines', 
#                     line=dict(color='black', width=lw, dash='dash'),
#                     showlegend=False)

# layout = go.Layout(title='Receiver operating characteristic Plot (ROC-AUC)',
#                    xaxis=dict(title='False Positive Rate'),
#                    yaxis=dict(title='True Positive Rate'))

# fig = go.Figure(data=data, layout=layout)

# st.plotly_chart(fig, use_container_width=True)

##--------------------------------------------------------------------------------------------------------------------------##

# New Function


title_text = 'AutoHealthX: Results'
st.markdown(f"<h2 style='text-align: center;'><b>{title_text}</b></h2>", unsafe_allow_html=True)


def model_performance(model,y_pred,y_test,y_score) : 
    #Conf matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    trace1 = go.Heatmap(z = conf_matrix  ,x = ["0 (pred)","1 (pred)"],
                        y = ["0 (true)","1 (true)"],xgap = 2, ygap = 2, 
                        colorscale = 'blues', showscale  = False,showlegend = True,text = conf_matrix,
                        texttemplate="%{text}",textfont={"size": 10},hoverongaps = False)

    #Show metrics
    tp = conf_matrix[1,1]
    fn = conf_matrix[1,0]
    fp = conf_matrix[0,1]
    tn = conf_matrix[0,0]
    Accuracy  =  ((tp+tn)/(tp+tn+fp+fn))
    Precision =  (tp/(tp+fp))
    Recall    =  (tp/(tp+fn))
    F1_score  =  (2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn)))))

    show_metrics = pd.DataFrame(data=[[Accuracy , Precision, Recall, F1_score]])
    show_metrics = show_metrics.T

    colors = ['gold', 'lightgreen', 'lightcoral', 'lightskyblue']
    trace2 = go.Bar(x = (show_metrics[0].values), 
                   y = ['Accuracy', 'Precision', 'Recall', 'F1_score'], text = np.round_(show_metrics[0].values,4),
                    textposition = 'auto',
                   orientation = 'h', opacity = 0.8,marker=dict(
            color=colors,
            line=dict(color='#000000',width=1.5)))
    
    #Roc curve
    model_roc_auc = round(roc_auc_score(y_test, y_score) , 3)
    fpr, tpr, t = roc_curve(y_test, y_score)
    trace3 = go.Scatter(x = fpr,y = tpr,
                        name = "Roc : " + str(model_roc_auc),
                        line = dict(color = ('rgb(22, 96, 167)'),width = 2), fill='tozeroy')
    trace4 = go.Scatter(x = [0,1],y = [0,1],
                        line = dict(color = ('black'),width = 1.5,
                        dash = 'dot'))
    
    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    trace5 = go.Scatter(x = recall, y = precision,
                        name = "Precision" + str(precision),
                        line = dict(color = ('lightcoral'),width = 2), fill='tozeroy')
    
    # #Feature importance
    # coefficients  = pd.DataFrame(eval(model).feature_importances_)
    # column_data   = pd.DataFrame(list(data))
    # coef_sumry    = (pd.merge(coefficients,column_data,left_index= True,
    #                           right_index= True, how = "left"))
    # coef_sumry.columns = ["coefficients","features"]
    # coef_sumry    = coef_sumry.sort_values(by = "coefficients",ascending = False)
    # coef_sumry = coef_sumry[coef_sumry["coefficients"] !=0]
    # trace6 = go.Bar(x = coef_sumry["features"],y = coef_sumry["coefficients"],
    #                 name = "coefficients",
    #                 marker = dict(color = coef_sumry["coefficients"],
    #                               colorscale = "Viridis",
    #                               line = dict(width = .6,color = "black")))
    
    #Cumulative gain

    pos = pd.get_dummies(y_test).to_numpy()
    #pos = pos[:,1] 
    npos = np.sum(pos)
    index = np.argsort(y_score) 
    index = index[::-1] 
    sort_pos = pos[index]
    #cumulative sum
    cpos = np.cumsum(sort_pos) 
    #recall
    recall = cpos/npos 
    #size obs test
    n = y_test.shape[0] 
    size = np.arange(start=1,stop=369,step=1) 
    #proportion
    size = size / n 
    #plots
    model = model
    trace7 = go.Scatter(x = size,y = recall,
                        name = "Lift curve",
                        line = dict(color = ('gold'),width = 2), fill='tozeroy') 
    
    #Subplots
    fig = tls.make_subplots(rows=3, cols=2, print_grid=False, 
                          specs=[[{}, {}], 
                                 [{}, {}],
                                 [{'colspan': 2}, None]],
                          subplot_titles=('Confusion Matrix',
                                        'Metrics',
                                        'ROC curve'+" "+ '('+ str(model_roc_auc)+')',
                                        'Precision - Recall curve',
                                        'Cumulative gains curve'
                                        ))
    
    fig.append_trace(trace1,1,1)
    fig.append_trace(trace2,1,2)
    fig.append_trace(trace3,2,1)
    fig.append_trace(trace4,2,1)
    fig.append_trace(trace5,2,2)
   # fig.append_trace(trace6,4,1)
    fig.append_trace(trace7,3,1)
    
    fig['layout'].update(showlegend = False, title = '<b>Model performance report</b><br>',
                        autosize = False, height = 1500,width = 830,
                        plot_bgcolor = 'rgba(240,240,240, 0.95)',
                        paper_bgcolor = 'rgba(240,240,240, 0.95)',
                        margin = dict(b = 195))
    fig["layout"]["xaxis2"].update((dict(range=[0, 1])))
    fig["layout"]["xaxis3"].update(dict(title = "false positive rate"))
    fig["layout"]["yaxis3"].update(dict(title = "true positive rate"))
    fig["layout"]["xaxis4"].update(dict(title = "recall"), range = [0,1.05])
    fig["layout"]["yaxis4"].update(dict(title = "precision"), range = [0,1.05])
    fig["layout"]["xaxis5"].update(dict(title = "Percentage contacted"))
    fig["layout"]["yaxis5"].update(dict(title = "Percentage positive targeted"))
    fig.layout.titlefont.size = 14
    
    st.plotly_chart(fig, use_container_width=True)


x_test_data = StringIO(read_file('testing_data_automl', '{}_x_test.csv'.format(original_filename)))
x_test = pd.read_csv(x_test_data)

y_test_data = StringIO(read_file('testing_data_automl', '{}_y_test.csv'.format(original_filename)))
y_test = pd.read_csv(y_test_data)

y_pred = model.predict(x_test)
for i in range(len(y_pred)):
    if(y_pred[i]>0.5):
        y_pred[i] = 1
    else:
        y_pred[i] = 0

y_score = OneHotEncoder().fit_transform(pd.DataFrame(y_pred)).toarray()[:,1]
model_performance('model',y_pred,y_test,y_score)

