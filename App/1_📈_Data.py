import streamlit as st
from google.oauth2 import service_account
from google.cloud import storage
import pandas as pd
import numpy as np
from io import StringIO
import plotly.graph_objects as go

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

title_text = 'AutoHealthX: Brest Cancer Dataset'
st.markdown(f"<h2 style='text-align: center;'><b>{title_text}</b></h2>", unsafe_allow_html=True)

# Space out the maps so the first one is 2x the size of the other three
c1, c2, c3 = st.columns((2, 1, 1))

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)

# Retrieve file contents.
# Uses st.experimental_memo to only rerun when the query changes or after 10 min.
@st.experimental_memo(ttl=600)
def read_file(bucket_name, file_path):
    bucket = client.bucket(bucket_name)
    content = bucket.blob(file_path).download_as_string().decode("utf-8")
    return content

bucket_name = "tabular_dataset" #"testing_bucket_automlxai"
file_path = "wisc_bc_data.csv"

content = read_file(bucket_name, file_path)

data = StringIO(content)
df = pd.read_csv(data)
c1.markdown('Sample Data')
c1.write(df)

# Writting dataframe to session_state
if 'df' not in st.session_state:
    st.session_state['df'] = df


### DataSet INFO
c2.write("  ")
c2.markdown('Dataset Info')

num_var = len(df.iloc[0,:]) - 1
num_obs = len(df)
num_class = len(np.unique(df.iloc[:,-1]))
num_na = df.isna().sum().sum()

df_info = pd.DataFrame([num_var,num_obs,num_class,num_na],
                index = ['Number of variables', 'Number of observations', 'Number of classes','Missing cells'],columns = [''])
style = df_info.style.hide_columns()
c2.write(style.to_html(),unsafe_allow_html=True)

### Dataset TYPE
c2.write("  ")
c2.markdown('Data Type')

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = df.select_dtypes(include=numerics)
num_numeric = numeric.count().sum()

num_object = df.select_dtypes(include=['O']).count().sum()

num_bool = df.select_dtypes(include=['bool']).count().sum()

date_time = ['datetime64[ns]', 'datetime64[ns, UTC]','datetime64[ns, US/Pacific]']
date_time_col = df.select_dtypes(include=date_time)
num_date_time = date_time_col.count().sum()

df_type = pd.DataFrame([num_numeric,num_object,num_bool,num_date_time],
                index = ['Numeric', 'Categorical', 'Boolean','Date'],columns = [''])
style = df_type.style.hide_columns()
c2.write(style.to_html(),unsafe_allow_html=True)

### Class Distribution

c3.markdown('Class Distribution')

classes = df.iloc[:,-1].groupby(df.iloc[:,-1]).count()
fig = go.Figure(go.Pie(values = classes.values,labels=classes.index))
fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
)
c3.plotly_chart(fig,use_container_width=True)

### Dataset Corellation Plots

st.markdown('Data Corellation')

options = st.multiselect(
    'Select attributes',
    df.columns,default= list(df.columns[:4]))

index_vals = df.iloc[:,-1].astype('category').cat.codes

dimensions = []
for option in options:
    dimensions.append(dict(label = option, values = df[option]))

fig = go.Figure(data=go.Splom(
                dimensions= dimensions,
                text=df.iloc[:,-1],
                marker=dict(color=index_vals,
                            showscale=False, # colors encode categorical variables
                            line_color='white', line_width=0.5)
                ))

fig.update_layout(title= 'Scatterplot matrix of selected attributes',
                    autosize=True,
                    height=1000,
                  dragmode='select',
                  hovermode='closest')


st.plotly_chart(fig,use_container_width=True)



