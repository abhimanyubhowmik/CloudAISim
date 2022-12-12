import streamlit as st
import pandas_profiling
import pandas as pd
import numpy as np
from streamlit_pandas_profiling import st_profile_report



title_text = 'AutoHealthX: Exploratory Data Analysis'
st.markdown(f"<h2 style='text-align: center;'><b>{title_text}</b></h2>", unsafe_allow_html=True)
df = st.session_state['df']



with st.form("eda_form"):
   value = st.slider('Percentage of Data for Visualization', 0, 100, 10)
   options = st.multiselect(
    'Select attributes for analysis',
    df.columns,default= list(df.columns[:3]))
   submitted = st.form_submit_button("Submit")

# After Submission
if submitted:
    total_len = len(df.index)
    val_len = int(np.round(value*total_len/100))
    new_df = df.iloc[:val_len,:][options]
    profile_df = new_df.profile_report()
    st_profile_report(profile_df)

    # Download EDA
    export=profile_df.to_html()
    c1, c2, c3,c4 = st.columns((1, 1, 1, 1))
    c4.download_button(label="Download Full Report", data=export, file_name='report.html')







