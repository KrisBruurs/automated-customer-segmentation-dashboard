import pandas as pd
import streamlit as st

st.title('Automated Customer Segmentation Dashboard')
st.write('Upload your customer data to get started with segmentation analysis.')

df = st.file_uploader('Upload your customer data (CSV)', type=['csv'])




