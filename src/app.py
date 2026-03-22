import pandas as pd
import streamlit as st

st.title('Automated Customer Segmentation Dashboard')
st.write('Upload your customer data to get started with segmentation analysis.')

upload_file = st.file_uploader('Upload your customer data (CSV or Excel)', type=['csv', 'xlsx'])

if upload_file is not None:
    df = pd.read_csv(upload_file)
    rfm_source_df = df.copy()
    
    with st.expander('Preview your data'):
        st.dataframe(df.head())

    columns = df.columns.tolist()

    possible_total_spend_cols = []
    total_spend_keywords = [
        'total_spend', 'totalspent', 'spend', 'amount',
        'revenue', 'sales', 'order_total', 'total_amount', 'monetary'
    ]

    for col in columns:
        normalized_col = col.lower().strip().replace(' ', '_')
        keyword_match = any(keyword in normalized_col for keyword in total_spend_keywords)
        numeric_col = pd.api.types.is_numeric_dtype(df[col])
        if keyword_match and numeric_col:
            possible_total_spend_cols.append(col)

    suggested_total_spend_col = possible_total_spend_cols[0] if possible_total_spend_cols else None

    with st.expander('Overview of your columns'):
        st.write('Overview of all columns')
        for col in columns:
            st.markdown(f'- **{col}**')

    st.write('---')

    st.subheader('Checks before starting segmentation')

    st.markdown(
        """
        Before we can perform RFM segmentation, we need to ensure your dataset contains the necessary columns.
        """
        )

    if suggested_total_spend_col:
        st.info(f"Possible total spend column detected: **{suggested_total_spend_col}**")
    else:
        st.warning('No likely total spend column was detected automatically. You can still select one manually below.')

    choice_ts = st.radio('Does your dataset contain a total spend column?', 
                        options=['Yes', 'No'],
                        index=None)

    order_id_col = None
    quantity_col = None
    price_col = None

    if choice_ts == 'No':
        st.markdown(
            """
            No worries! We will proceed to calculate the total spend for each customer order.

            Please fill in:
            - Order identifier: Usually `order_id`, `transaction_id`, or similar.
            - Quantity identifier: Usually `quantity`, `qty`, or similar.
                            If your dataset does not have a quantity column, select `None`.
            - Price identifier: Usually `price`, `unit_price`, `amount`, or similar.
            
            This will allow us to compute the total spend for each transaction and perform RFM analysis based on that.
            """
        )

        st.write('---')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<p style='font-size:14px; margin-bottom:6px;'>Please select the Order ID column</p>", unsafe_allow_html=True)
            order_id_col = st.selectbox('Order ID Column', 
                                        options=columns,
                                        index=None)
        with col2:
            st.markdown("<p style='font-size:14px; margin-bottom:6px;'>Please select the Quantity column</p>", unsafe_allow_html=True)
            quantity_col = st.selectbox('Quantity Column', 
                                        options=[None] + columns,
                                        index=None)
        with col3:
            st.markdown("<p style='font-size:14px; margin-bottom:6px;'>Please select the Price column</p>", unsafe_allow_html=True)
            price_col = st.selectbox('Price Column', 
                                     options=columns,
                                     index=None)

        if order_id_col is not None and price_col is not None:
            price_series = pd.to_numeric(df[price_col], errors='coerce').fillna(0)

            if quantity_col is None:
                quantity_series = pd.Series(1, index=df.index, dtype='float64')
            else:
                quantity_series = pd.to_numeric(df[quantity_col], errors='coerce').fillna(1)

            line_spend = price_series * quantity_series
            df['computed_total_spend'] = line_spend.groupby(df[order_id_col]).transform('sum')

            rfm_source_df = df.drop_duplicates(subset=[order_id_col]).copy()
            st.session_state['rfm_source_df'] = rfm_source_df

            st.write('Total spend has been computed at order level by grouping line items using the selected Order ID. You can select computed_total_spend as your Monetary value in the next step.')
            columns = df.columns.tolist()
        elif order_id_col is None and price_col is not None:
            st.warning('Please select an Order ID column to compute total spend at order level.')

    elif choice_ts == 'Yes':
        st.write('Great! You will be able to select your total spend column in the next step.')

    if choice_ts == 'Yes' or price_col != None:
        st.write('---')
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

                - **Recency**: `order_date`, `last_order_days`, `recency`
                - **Customer ID**: `customer_id`, `user_id`, `client_id`
                - **Monetary**: `total_spent`, `revenue`, `amount`, `monetary`

                Tips:
                - Recency is usually a date of purchase.
                - Frequency is computed after grouping by customer identifier.
                - Monetary should be a **currency amount** (total spend).
                """
            )


        col1, col2, col3 = st.columns(3)
        monetary_default_index = columns.index(suggested_total_spend_col) if suggested_total_spend_col else 0

        with col1:
            st.markdown("<p style='font-size:14px; margin-bottom:6px;'>Please select the Customer ID column</p>", unsafe_allow_html=True)
            customer_id_col = st.selectbox('Customer ID Column', 
                                           options=columns,
                                           index=None)

        with col2:
            st.markdown("<p style='font-size:14px; margin-bottom:6px;'>Please select the Recency column</p>", unsafe_allow_html=True)
            recency_col = st.selectbox('Recency Column', 
                                       options=columns, 
                                       index=None)   
        
        with col3:
            st.markdown("<p style='font-size:14px; margin-bottom:6px;'>Please select the Monetary column</p>", unsafe_allow_html=True)
            monetary_col = st.selectbox('Monetary Column', 
                                        options=columns, 
                                        index=None)



elif upload_file is None:
    st.write('Please upload your file to proceed with segmentation analysis.')

 