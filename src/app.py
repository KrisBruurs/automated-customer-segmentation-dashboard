import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import streamlit as st


@st.cache_data(show_spinner=False)
def compute_silhouette_scores(cluster_values, min_k, max_k):
    scores = []
    tested_k = []
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(cluster_values)
        if len(set(labels)) < 2:
            continue
        scores.append(silhouette_score(cluster_values, labels))
        tested_k.append(k)
    return tested_k, scores


def _normalized_col_name(col_name):
    return str(col_name).lower().strip().replace(' ', '_')


def parse_dates_flexible(series):
    # Try standard parsing first, then common day-first and Excel serial-date fallbacks.
    parsed_default = pd.to_datetime(series, errors='coerce')
    best_parsed = parsed_default
    best_rate = parsed_default.notna().mean()

    parsed_dayfirst = pd.to_datetime(series, errors='coerce', dayfirst=True)
    dayfirst_rate = parsed_dayfirst.notna().mean()
    if dayfirst_rate > best_rate:
        best_parsed = parsed_dayfirst
        best_rate = dayfirst_rate

    if pd.api.types.is_numeric_dtype(series):
        parsed_excel_serial = pd.to_datetime(series, errors='coerce', unit='D', origin='1899-12-30')
        excel_rate = parsed_excel_serial.notna().mean()
        if excel_rate > best_rate:
            best_parsed = parsed_excel_serial
            best_rate = excel_rate

    return best_parsed, best_rate


def get_column_selection_warnings(df, customer_id_col, order_id_col, recency_col, monetary_col):
    warnings = []

    selected_cols = [
        ('Customer ID', customer_id_col),
        ('Order ID', order_id_col),
        ('Recency', recency_col),
        ('Monetary', monetary_col),
    ]

    chosen = [col for _, col in selected_cols if col is not None]
    if len(chosen) != len(set(chosen)):
        warnings.append('You selected the same column for multiple roles. Each role should usually have its own column.')

    if customer_id_col is not None:
        cid_name = _normalized_col_name(customer_id_col)
        cid_keywords = ['customer', 'client', 'user', 'member', 'account', 'cust']
        if not any(k in cid_name for k in cid_keywords):
            warnings.append('Customer ID column name does not look like a customer identifier (for example: customer_id, client_id, user_id).')

        cid_series = df[customer_id_col]
        cid_non_null = cid_series.dropna()
        if not cid_non_null.empty:
            customer_uniqueness = cid_non_null.nunique() / len(cid_non_null)
            if customer_uniqueness > 0.995:
                is_likely_aggregated = False
                if order_id_col is not None and order_id_col in df.columns:
                    customer_order_counts = (
                        df[[customer_id_col, order_id_col]]
                        .dropna()
                        .groupby(customer_id_col)[order_id_col]
                        .nunique()
                    )
                    if not customer_order_counts.empty and customer_order_counts.median() <= 1:
                        is_likely_aggregated = True

                if not is_likely_aggregated:
                    warnings.append('Customer ID is almost unique per row. This can be valid, but please double-check that you did not select an order/transaction ID by mistake.')
            if customer_uniqueness < 0.01:
                warnings.append('Customer ID has very low variety. Please check if this is the correct customer identifier column.')

    if order_id_col is not None:
        oid_name = _normalized_col_name(order_id_col)
        oid_keywords = ['order', 'invoice', 'transaction', 'purchase', 'receipt']
        if not any(k in oid_name for k in oid_keywords):
            warnings.append('Order ID column name does not look like an order identifier (for example: order_id, invoice_id, transaction_id).')

        oid_series = df[order_id_col].dropna()
        if not oid_series.empty:
            order_uniqueness = oid_series.nunique() / len(oid_series)
            if order_uniqueness < 0.2:
                warnings.append('Order ID has low uniqueness. Frequency may be underestimated if this is not a true order-level identifier.')

    if recency_col is not None:
        rec_name = _normalized_col_name(recency_col)
        rec_keywords = ['date', 'time', 'order_date', 'purchase_date', 'recency']
        if not any(k in rec_name for k in rec_keywords):
            warnings.append('Recency column name does not look date-related. Recency should usually be a purchase date column.')

        parsed_dates, parse_rate = parse_dates_flexible(df[recency_col])
        if parse_rate < 0.5:
            warnings.append('A large part of the Recency column cannot be read as dates. Please choose another date column.')
        elif parsed_dates.notna().any() and parsed_dates.nunique(dropna=True) <= 1:
            warnings.append('Recency dates have little to no variation. Segmentation quality may be weak.')

    if monetary_col is not None:
        mon_name = _normalized_col_name(monetary_col)
        mon_keywords = ['amount', 'spend', 'price', 'sales', 'revenue', 'total', 'monetary']
        if not any(k in mon_name for k in mon_keywords):
            warnings.append('Monetary column name does not look like a spend/amount field.')

        mon_values = pd.to_numeric(df[monetary_col], errors='coerce')
        valid_money_rate = mon_values.notna().mean()
        if valid_money_rate < 0.7:
            warnings.append('A large part of the Monetary column is not numeric. Please choose a numeric spend column.')
        else:
            non_null_monetary = mon_values.dropna()
            if not non_null_monetary.empty:
                negative_ratio = (non_null_monetary < 0).mean()
                if negative_ratio > 0.2:
                    warnings.append('Many monetary values are negative. This can indicate returns/refunds and may affect clustering.')
                if non_null_monetary.nunique() <= 3:
                    warnings.append('Monetary values have very low variation. Segments may be less meaningful.')

    return warnings


def get_total_spend_selection_warnings(df, order_id_col, quantity_col, price_col):
    warnings = []

    selected = [col for col in [order_id_col, quantity_col, price_col] if col is not None]
    if len(selected) != len(set(selected)):
        warnings.append('You selected the same column for multiple roles. Order ID, Quantity, and Price should usually be different columns.')

    if order_id_col is not None:
        oid_name = _normalized_col_name(order_id_col)
        oid_keywords = ['order', 'invoice', 'transaction', 'purchase', 'receipt']
        if not any(k in oid_name for k in oid_keywords):
            warnings.append('Order ID column name does not look like an order identifier (for example: order_id or transaction_id).')

        oid_series = df[order_id_col].dropna()
        if not oid_series.empty and (oid_series.nunique() / len(oid_series)) < 0.2:
            warnings.append('Order ID has low uniqueness. Please check if this is really an order-level identifier.')

    if quantity_col is not None:
        qty_name = _normalized_col_name(quantity_col)
        qty_keywords = ['qty', 'quantity', 'units', 'count', 'pieces']
        if not any(k in qty_name for k in qty_keywords):
            warnings.append('Quantity column name does not look like a quantity field.')

        qty_values = pd.to_numeric(df[quantity_col], errors='coerce')
        qty_parse_rate = qty_values.notna().mean()
        if qty_parse_rate < 0.7:
            warnings.append('A large part of the Quantity column is not numeric.')
        elif qty_values.dropna().le(0).mean() > 0.5:
            warnings.append('Many quantity values are zero or negative. Please verify that this is the correct quantity column.')

    if price_col is not None:
        price_name = _normalized_col_name(price_col)
        price_keywords = ['price', 'amount', 'revenue', 'sales', 'spend', 'total', 'cost', 'value']
        if not any(k in price_name for k in price_keywords):
            warnings.append('Price column name does not look like a price/amount field.')

        price_values = pd.to_numeric(df[price_col], errors='coerce')
        price_parse_rate = price_values.notna().mean()
        if price_parse_rate < 0.7:
            warnings.append('A large part of the Price column is not numeric.')
        else:
            non_null_price = price_values.dropna()
            if not non_null_price.empty and (non_null_price < 0).mean() > 0.2:
                warnings.append('Many price values are negative. This may indicate returns/credits instead of normal sales amounts.')

    return warnings

st.title('Customer Segmentation Assistant')
st.write('This tool helps you group customers into clear segments. Follow the steps below.')
st.caption('You do not need technical knowledge. Select the columns that best match each question.')

upload_file = st.file_uploader('Step 1: Upload your data file (CSV or Excel)', type=['csv', 'xlsx', 'xls'])

if upload_file is not None:
    file_name = upload_file.name.lower()
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(upload_file)
        elif file_name.endswith(('.xlsx', '.xls')):
            excel_file = pd.ExcelFile(upload_file)
            if len(excel_file.sheet_names) > 1:
                selected_sheet = st.selectbox('Choose the sheet you want to use', options=excel_file.sheet_names)
            else:
                selected_sheet = excel_file.sheet_names[0]
            df = excel_file.parse(selected_sheet)
        else:
            st.error('This file type is not supported. Please upload a CSV or Excel file (.xlsx or .xls).')
            st.stop()
    except UnicodeDecodeError:
        st.error('We could not read this CSV file. Please save it with UTF-8 encoding and try again.')
        st.stop()
    except pd.errors.ParserError:
        st.error('We could not read this file structure. Please check the delimiter and column layout.')
        st.stop()
    except ValueError as exc:
        st.error(f'File format issue: {exc}')
        st.stop()
        
    rfm_source_df = df.copy()
    
    with st.expander('Preview: first rows of your data'):
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

    with st.expander('Preview: all column names'):
        st.write('Columns found in your file:')
        for col in columns:
            st.markdown(f'- **{col}**')

    st.write('---')

    st.subheader('Step 2: Tell us about your spend column')

    st.markdown(
        """
        We first need to know if your file already has a total spend column.
        """
        )

    if suggested_total_spend_col:
        st.info(f"Good news: a likely total spend column was found: **{suggested_total_spend_col}**")
    else:
        st.warning('No clear total spend column was found automatically. You can still continue.')

    choice_ts = st.radio('Do you already have a total spend column in your file?', 
                        options=['Yes', 'No'],
                        index=None)

    order_id_col = None
    quantity_col = None
    price_col = None

    if choice_ts == 'No':
        st.markdown(
            """
            No problem. We can calculate total spend for each order.

            Please select:
            - Order identifier: Usually `order_id`, `transaction_id`, or similar.
            - Quantity identifier: Usually `quantity`, `qty`, or similar.
                            If your dataset does not have a quantity column, select `None`.
            - Price identifier: Usually `price`, `unit_price`, `amount`, or similar.
            
            After this, the app will create a spend column for you.
            """
        )

        st.write('---')

        col1, col2, col3 = st.columns(3)
        with col1:
            order_id_col = st.selectbox('Order ID column', 
                                        options=columns,
                                        index=None)
        with col2:
            quantity_col = st.selectbox('Quantity column', 
                                        options=[None] + columns,
                                        index=None)
        with col3:
            price_col = st.selectbox('Price column', 
                                     options=columns,
                                     index=None)

        total_spend_warnings = get_total_spend_selection_warnings(
            df,
            order_id_col,
            quantity_col,
            price_col,
        )
        if total_spend_warnings:
            st.subheader('Check your column choices')
            for warning_message in total_spend_warnings:
                st.warning(warning_message)

        if order_id_col is not None and price_col is not None:
            price_series = pd.to_numeric(df[price_col], errors='coerce').fillna(0)

            if quantity_col is None:
                quantity_series = pd.Series(1, index=df.index, dtype='float64')
            else:
                quantity_series = pd.to_numeric(df[quantity_col], errors='coerce').fillna(1)

            line_spend = price_series * quantity_series
            df['computed_total_spend'] = line_spend.groupby(df[order_id_col]).transform('sum')

            rfm_source_df = df.copy()
            st.session_state['rfm_source_df'] = rfm_source_df

            st.success('Total spend was calculated successfully. In the next step, choose computed_total_spend as your spend column.')
            columns = df.columns.tolist()
        elif order_id_col is None and price_col is not None:
            st.warning('Please select an Order ID column so total spend can be calculated.')

    elif choice_ts == 'Yes':
        st.write('Great. You will pick that spend column in the next step.')

    can_continue_to_rfm = choice_ts == 'Yes' or (choice_ts == 'No' and order_id_col is not None and price_col is not None)
    if can_continue_to_rfm:
        st.write('---')
        st.subheader('Step 3: Match your columns')
        st.markdown(
            """
            Quick guide:

            - **Customer ID**: Unique value for each customer.
            - **Order ID**: Unique value for each order.
            - **Recency**: Purchase date column.
            - **Monetary**: Spend amount column.
            """
        )

        with st.expander('Need help choosing columns?'):
            st.markdown(
                """
                Common examples:

                - **Recency**: `order_date`, `last_order_days`, `recency`
                - **Customer ID**: `customer_id`, `user_id`, `client_id`
                - **Monetary**: `total_spent`, `revenue`, `amount`, `monetary`

                Tips:
                - Recency should be a date column.
                - Frequency is calculated automatically from orders.
                - Monetary should be a currency/spend amount.
                """
            )


        col1, col2 = st.columns(2)

        with col1:
            customer_id_col = st.selectbox('Customer ID column', 
                                           options=columns,
                                           index=None)

            if choice_ts == 'No' and order_id_col is not None:
                st.info(f'Using Order ID column from the previous step: {order_id_col}')
            else:
                order_id_col = st.selectbox('Order ID column', 
                                            options=columns, 
                                            index=None)

        with col2:
            recency_col = st.selectbox('Recency (date) column', 
                                       options=columns, 
                                       index=None)   
            
            monetary_col = st.selectbox(
                'Monetary (spend) column',
                                        options=columns, 
                                        index=(columns.index(suggested_total_spend_col) if suggested_total_spend_col else None))

        selection_warnings = get_column_selection_warnings(
            rfm_source_df,
            customer_id_col,
            order_id_col,
            recency_col,
            monetary_col,
        )
        if selection_warnings:
            st.subheader('Check your column choices')
            for warning_message in selection_warnings:
                st.warning(warning_message)

        reference_date = None
        recency_dates = None
        if recency_col is not None:
            recency_dates, _ = parse_dates_flexible(rfm_source_df[recency_col])
            valid_recency_dates = recency_dates.dropna()

            if valid_recency_dates.empty:
                st.warning('This recency column does not contain valid dates. Please choose another date column.')
            else:
                st.subheader('Step 4: Choose your reference date')
                st.caption('Recency is calculated as: reference date minus each customer\'s latest purchase date.')
                reference_date_option = st.radio(
                    'How should we pick the reference date?',
                    options=['Use the latest date in my data', 'I want to choose a custom date'],
                    index=0
                )

                dataset_latest_date = valid_recency_dates.max().date()
                dataset_earliest_date = valid_recency_dates.min().date()

                if reference_date_option == 'Use the latest date in my data':
                    reference_date = pd.Timestamp(dataset_latest_date)
                else:
                    selected_reference_date = st.date_input(
                        'Choose custom reference date',
                        value=dataset_latest_date,
                        min_value=dataset_earliest_date,
                        max_value=dataset_latest_date
                    )
                    reference_date = pd.Timestamp(selected_reference_date)

        run_segmentation = st.button('Step 5: Create customer segments')

        if run_segmentation:
            if customer_id_col is None or order_id_col is None or recency_col is None or monetary_col is None or reference_date is None or recency_dates is None:
                st.warning('Please complete all required selections before creating segments.')
                st.stop()

            rfm_source_df = rfm_source_df[[customer_id_col, order_id_col, recency_col, monetary_col]].copy()
            rfm_source_df[recency_col] = recency_dates
            rfm_source_df[monetary_col] = pd.to_numeric(rfm_source_df[monetary_col], errors='coerce').fillna(0)
            rfm_source_df = rfm_source_df.dropna(subset=[recency_col])
            st.session_state['rfm_source_df'] = rfm_source_df

            rfm_df = rfm_source_df.groupby(customer_id_col).agg({
                recency_col: lambda v: (reference_date - v.max()).days,
                order_id_col: 'nunique',
                monetary_col: 'sum'
            })
            rfm_df.columns = ['Recency', 'Frequency', 'Monetary']

            scale = StandardScaler()
            rfm_cluster = pd.DataFrame(
                scale.fit_transform(rfm_df),
                index=rfm_df.index,
                columns=rfm_df.columns
            )

            n_samples = len(rfm_cluster)
            if n_samples < 3:
                st.warning('Not enough customers to run silhouette analysis. At least 3 customers are required.')
                st.session_state['rfm_df'] = rfm_df
                st.dataframe(rfm_df.head())
            else:
                max_k = min(10, n_samples - 1)
                tested_k, scores = compute_silhouette_scores(rfm_cluster.values, 2, max_k)

                if not scores:
                    st.warning('We could not evaluate segment quality for this data. Please review your selected columns and values.')
                    st.session_state['rfm_df'] = rfm_df
                    st.dataframe(rfm_df.head())
                else:
                    optimal_k = tested_k[scores.index(max(scores))]
                    st.success(f'Suggested number of customer segments: **{optimal_k}**')

                    k = st.slider(
                        'Choose number of customer segments',
                        min_value=2,
                        max_value=max_k,
                        value=optimal_k
                    )

                    final_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    rfm_df['Cluster'] = final_kmeans.fit_predict(rfm_cluster)
                    st.session_state['rfm_df'] = rfm_df

                    with st.expander('Technical preview: scaled values'):
                        st.dataframe(rfm_cluster.head())
                    st.subheader('Your customer segments (preview)')
                    st.dataframe(rfm_df.head())

elif upload_file is None:
    st.info('Upload a file to begin.')

 