import pandas as pd
import streamlit as st

st.title('Automated Customer Segmentation Dashboard')
st.write('Upload your customer data to get started with segmentation analysis.')

upload_file = st.file_uploader('Upload your customer data (CSV)', type=['csv'])

if upload_file is not None:
    df = pd.read_csv(upload_file)

    st.write('Select your columns below to build an RFM table grouped by customer.')

    columns = df.columns.tolist()
    st.write('Overview of all columns')
    st.write(columns)

    st.subheader('How to choose your RFM columns')
    st.markdown(
        """
        If you are new to RFM analysis, choose columns using this guide:

        - **Recency**: How recently a customer made a purchase.
        - **Frequency**: Calculated automatically by counting each customer's transactions.
        - **Monetary**: How much money a customer spends.
        """
    )

    with st.expander('Need help matching your CSV columns?'):
        st.markdown(
            """
            Look for column names like these:

            - **Recency**: `days_since_last_purchase`, `last_order_days`, `recency`
            - **Customer ID**: `customer_id`, `user_id`, `client_id`
            - **Monetary**: `total_spent`, `revenue`, `amount`, `monetary`

            Tips:
            - Recency is usually measured in **days** (smaller number = more recent).
            - Frequency is computed after grouping by customer ID.
            - Monetary should be a **currency amount** (total spend).
            """
        )


    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<p style='font-size:14px; margin-bottom:6px;'>Please select the Customer ID column</p>", unsafe_allow_html=True)
        customer_id_col = st.selectbox('Customer ID Column', options=columns)

    with col2:
        st.markdown("<p style='font-size:14px; margin-bottom:6px;'>Please select the Recency column</p>", unsafe_allow_html=True)
        recency_col = st.selectbox('Recency Column', options=columns)   
    
    with col3:
        st.markdown("<p style='font-size:14px; margin-bottom:6px;'>Please select the Monetary column</p>", unsafe_allow_html=True)
        monetary_col = st.selectbox('Monetary Column', options=columns)



elif upload_file is None:
    st.write('Please upload a CSV file to proceed with segmentation analysis.')

 