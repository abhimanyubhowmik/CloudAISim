import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PowerTransformer
import mpld3
import streamlit.components.v1 as components
from google.oauth2 import service_account
from google.cloud import storage

st.set_page_config(layout="wide")

title_text = 'AutoHealthX: Feture Engineering'
st.markdown(f"<h2 style='text-align: center;'><b>{title_text}</b></h2>", unsafe_allow_html=True)

df = st.session_state['df']
attribute = st.selectbox(
'Select attribute for Visualization',
df.columns)

fig = plt.figure(figsize=(10,6))
plt.subplot(221)
sns.distplot(df[attribute])
plt.title('Probability Density Funation')
plt.subplot(223)
stats.probplot(df[attribute], dist="norm", plot=plt)
plt.title('QQ Plot')
plt.subplot(122)
plt.boxplot(df[attribute])
plt.tight_layout()
plt.title('Box Plot')
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=610)

st.write('Sample Data')
st.write(df)

with st.form("fe_form_1"):
    st.write('Feature Selection')
    options = st.multiselect(
    'Select Features',
    df.columns,default= list(df.columns))

    st.write('Missing Values Imputation')
    c1, c2, c3 = st.columns((1, 1, 1))
    method = c1.selectbox('Select Methods for missing values interpolation',
    ('linear', 'time','index','pad','nearest','zero', 'slinear', 'quadratic', 'cubic', 'spline', 'barycentric', 'polynomial')) 
    order = c2.number_input('Enter order of polynomial/spline',min_value=1)

    limits = c3.number_input('Maximum number of consecutive NaNs to fill',min_value=1)

    submitted_1 = st.form_submit_button("Submit")

with st.form("fe_form_2"):
    select = st.multiselect(
    'Select attributes for analysis',
    options)

    st.write('Feature Transformation')

    trans = st.radio(
    "Select transformation method",
    ('Log Transform', 'Power Transform','Square-Root Transform','Reciprocal Transformation'))
    
    st.write('Outlier Detectation')

    detect = st.selectbox('Select outlier capping method',
    ('Z-score', 'Interquartile Range','Percentile' ))

    submitted_2 = st.form_submit_button("Submit")

if 'new_df' not in st.session_state:
        st.session_state['new_df'] = df
        dataframe = df
else:

    dataframe = st.session_state['new_df']


def form_1():

    if submitted_1:
        new_df = df[options]
        if method == 'polynomial' or 'spline': 
            new_df = new_df.interpolate(method = method, limit = limits,order=order)
        else:
            new_df = new_df.interpolate(method = method, limit = limits)
        return new_df
    else:
        return -1

def form_2(data):

    if type(form_1()) == int :
        new_df = data
    else:
        new_df = form_1()


    if submitted_2:
        
        if trans == 'Log Transform':
            trf = FunctionTransformer(func=np.log1p)
            new_df[select] = trf.fit_transform(new_df[select])
        elif trans == 'Power Transform':
            pt = PowerTransformer()
            new_df[select] = pt.fit_transform(new_df[select])
        elif trans == 'Square-Root Transform':
            new_df[select] = new_df[select].abs()**(1/2)
        else:
            try:
                new_df[select] = 1/new_df[select]
            except:
                new_df[select] = 1/(new_df[select]+0.0001)
        if detect == 'Z-score':
            upper_limit = new_df[select].mean() + 3*new_df[select].std()
            lower_limit = new_df[select].mean() - 3*new_df[select].std()
        elif detect == 'Interquartile Range':
            percentile25 = new_df[select].quantile(0.25)
            percentile75 = new_df[select].quantile(0.75)
            iqr = percentile75 - percentile25
            upper_limit = percentile75 + 1.5 * iqr
            lower_limit = percentile25 - 1.5 * iqr
        else:
            upper_limit = new_df[select].quantile(0.99)
            lower_limit = new_df[select].quantile(0.01)
        
        new_df[select] = np.where(
        new_df[select]>upper_limit,
        upper_limit,
        np.where(
            new_df[select]<lower_limit,
            lower_limit,
            new_df[select]
            )
        )
        return new_df

    else:
        return new_df

@st.experimental_memo(ttl=600)
def upload_blob(bucket_name, _storage_client, source_data, destination_blob_name):
    """Uploads a file to the bucket."""    
    print('function upload_blob called')     
    try:
        bucket = _storage_client.get_bucket(bucket_name)    
        blob = bucket.blob(destination_blob_name)    
        blob.upload_from_string(source_data)    
        print('File {} uploaded to {}.'.format(destination_blob_name, bucket_name))
        return 0
    except Exception as e:
        print(e)
        return 1


if submitted_1 or submitted_2:
    new_df = form_2(dataframe)
    st.session_state['new_df'] = new_df
    dataframe = new_df
    st.write(new_df)

if st.button('Upload Data'):
    if 'new_df' not in st.session_state:
        new_df = st.session_state['df']
    else:
        new_df = st.session_state['new_df']
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    client = storage.Client(credentials=credentials)

    data = new_df.to_csv(index=False)

    filename = "wisc_bc_data_new.csv"

    bucket_name = "testing_bucket_automlxai"

    uploaded = upload_blob(bucket_name,client,data,filename)

    if uploaded == 0:
        st.write('Successfully Uploaded')
    else:
        st.write('Failed to Upload')



     









