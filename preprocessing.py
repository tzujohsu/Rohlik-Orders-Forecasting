from datetime import datetime
import pandas as pd
import string
import re
import numpy as np

czech_holiday = [ # Prague
    (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),#loss
    (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother Day"), #loss
]
brno_holiday = [ # Brno
    (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),#loss
    (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother Day"), #loss
]
munich_holidays = [ # Bavaria - Munich
    (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),#loss
    (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),#loss
]
frank_holidays = [ # Hesse - Frankfurt
    (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),#loss
    (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),#loss
]

# Holiday and warehouse mapping
warehouse_holiday_mapping = [
    (['Prague_1', 'Prague_2', 'Prague_3'], czech_holiday),
    (['Brno_1'], brno_holiday),
    (['Munich_1'], munich_holidays),
    (['Frankfurt_1'], frank_holidays),
    (['Budapest_1'], [])
]

# Generate dates for fixed-date holidays
def generate_dates_for_years(month_day, start_year=2020, end_year=2024):
    return [f"{year}-{month_day}" for year in range(start_year, end_year + 1)]

def fill_loss_holidays(df, warehouse_holiday_mapping):
    df = df.copy()
    for warehouses, holidays in warehouse_holiday_mapping:
        for item in holidays:
            dates, holiday_name = item

            # Generate dates for fixed-date holidays or reformat provided dates
            if isinstance(dates, str):
                month_day = datetime.strptime(dates, '%m/%d/%Y').strftime('%m-%d')
                generated_dates = generate_dates_for_years(month_day)
            else:
                generated_dates = [datetime.strptime(date, '%m/%d/%Y').strftime('%Y-%m-%d') for date in dates]

            # Update DataFrame for generated dates
            mask = (df['warehouse'].isin(warehouses)) & (df['date'].isin(generated_dates))
            df.loc[mask, ['holiday', 'holiday_name']] = [1, holiday_name]
    return df


def process_text_column(df, text_columns):
    """Processes text columns by lowercasing, removing punctuation, and replacing numbers."""
    table = str.maketrans('', '', string.punctuation)
    for col in text_columns:
        df[col] = df[col].str.lower().str.translate(table).str.replace(r'\d+', 'num', regex=True)
    return df

def add_date_features(df, date_columns):
    """Adds date-related features and removes original date columns."""
    df['day_before_holiday'] = df['holiday'].shift(-1).fillna(0)
    df['day_after_holiday'] = df['holiday'].shift().fillna(0)
    df['day_before_holiday'] = df['day_before_holiday'].astype(int)
    df['day_after_holiday'] = df['day_after_holiday'].astype(int)

    for col in date_columns:
        date_col = pd.to_datetime(df[col], errors='coerce')
        df[f"{col}_year"] = date_col.dt.year.fillna(-1)
        df[f"{col}_month"] = date_col.dt.month.fillna(-1)
        # df[f"{col}_week"] = date_col.dt.week.fillna(-1) #
        df[f"{col}_day"] = date_col.dt.day.fillna(-1)
        df[f"{col}_day_of_week"] = date_col.dt.dayofweek.fillna(-1)
        
        # df['sin_month']=np.sin(2*np.pi*df[f"{col}_month"]/12) #
        # df['sin_week']=np.sin(2*np.pi*df[f"{col}_week"]/53) #
        # df['sin_day']=np.sin(2*np.pi*df[f"{col}_day_of_week"]/7) #
        
        df.drop(col, axis=1, inplace=True)
    return df

def encode_categorical_columns(df, columns, encoder, fit=True):
    """Encodes categorical columns using a one-hot encoder."""
    if fit:
        encoded = encoder.fit_transform(df[columns])
    else:
        encoded = encoder.transform(df[columns])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(), index=df.index)
    return pd.concat([df, encoded_df], axis=1).drop(columns, axis=1)

def encode_text_columns(df, text_columns, vectorizer, fit=True):
    """Encodes text columns using a TF-IDF vectorizer."""
    for col in text_columns:
        if fit:
            vectors = vectorizer.fit_transform(df[col])
        else:
            vectors = vectorizer.transform(df[col])
        vector_df = pd.DataFrame.sparse.from_spmatrix(
            vectors, columns=[f"{col}_{feat}" for feat in vectorizer.get_feature_names_out()], index=df.index
        )
        df = pd.concat([df.drop(col, axis=1), vector_df], axis=1)
    return df

def fill_calendar2df(df, df_cld):
    
    # Manual fill all nan holiday name to "Not"
    df = df.fillna('Not')
    df['holiday'] = df['holiday'].astype(float)
    # Convert holiday to 1 if it is None
    df['holiday'] = df['holiday'].fillna(0)
    hl_count = 0 # count total number of holidays
    spec = 0 # count number of special holidays
    
    # Iterate through each holiday in df_cld
    for _, row in df_cld.iterrows():
        warehouse = row['warehouse']
        holiday_date = row['date']
        holiday_name = row['holiday_name']

        is_spec = False
        hl_count += 1

        # Calculate the date range: 2 days before, the holiday itself, and 1 day after
        # we fill this date range as holiday
        if (holiday_name in ['Labour Day','Easter Monday']): # special holidays
            date_range = pd.date_range(start=holiday_date - pd.Timedelta(days=2), end=holiday_date + pd.Timedelta(days=1))
            is_spec = True
        else: # with other holidays: 1 days before, the holiday itself, and 1 day after
            date_range = pd.date_range(start=holiday_date - pd.Timedelta(days=1), end=holiday_date + pd.Timedelta(days=1))

        # Update df for the corresponding warehouse and date range
        for i, date in enumerate(date_range):
            mask = (df['warehouse'] == warehouse) & (df['date'] == date)
            if is_spec and i==0:
                spec += 1
                df.loc[mask, 'holiday'] = 1.3 # Use higher weight for 2 special holidays
            else:
                df.loc[mask, 'holiday'] = 1

            # not fill holiday name for day after holiday
            if i+1!=len(date_range):
                df.loc[mask, 'holiday_name'] = holiday_name

    print("Total holidays:", hl_count)
    print("Total special holidays:", spec)
    return df

def cld_process(
        df: pd.DataFrame
    ):
    # Fill loss holidays
    df = fill_loss_holidays(df, warehouse_holiday_mapping)
    # Filter for rows with holidays and reset index
    df = df[df['holiday'] == 1].reset_index(drop=True)
    # Fill missing holiday names with "Easter Monday"
    df['holiday_name'].fillna('Easter Monday', inplace=True)
    # Sort DataFrames
    df = df.sort_values(by=['warehouse', 'date'])
    return df

def data_process(
        df: pd.DataFrame,
        create_test_mode: bool,
        target_cl: list,
        test_target_cl: list = None,
        onehot_encoder=None,
        tfidfvectorizer=None,
    ):
    if create_test_mode:
        test_ids = df['id']
    # Filter for target warehouses
    test_target_cl = test_target_cl or target_cl
    df = df[df['warehouse'].isin(target_cl)]
    
    # Drop unnecessary columns
    ignore_columns = [
        'id', 'shutdown', 'mini_shutdown', 'blackout', 'mov_change', 
        'frankfurt_shutdown', 'precipitation', 'snow', 'user_activity_1', 'user_activity_2'
    ]
    df = df.drop(ignore_columns, axis=1, errors="ignore")


    
    # Dictionary for renaming specific holiday names
    rename_dict = {
        "Memorial Day for the Victims of the Holocaust": "Victims of the Holocaust",
        "Memorial Day for the Victims of the Communist Dictatorships": "Victims of the Communist",
        "Den vzniku samostatneho ceskoslovenskeho statu": "Den vzniku"
    }

    # Replace the values in the 'holiday_name' column based on the dictionary
    df['holiday_name'] = df['holiday_name'].replace(rename_dict)

    # Add date and holiday features
    df = add_date_features(df, date_columns=['date'])

    # Process text columns
    text_columns = ['holiday_name']
    df = process_text_column(df, text_columns)

    # Separate target and features
    target_columns = ['orders']
    if set(target_columns).issubset(df.columns):
        feature_train = df.drop(target_columns, axis=1)
        target_train = df[target_columns].copy()
    else:
        feature_train = df
        target_train = None

    # One-hot encode categorical columns
    categorical_columns = ['warehouse']
    if len(target_cl) != 1:
        feature_train = encode_categorical_columns(
            feature_train, categorical_columns, onehot_encoder, fit=not create_test_mode
        )

    # TF-IDF encode text columns
    feature_train = encode_text_columns(
        feature_train, text_columns, tfidfvectorizer, fit=not create_test_mode
    )

    # Convert remaining features to sparse
    feature_train = feature_train.astype(pd.SparseDtype("float64", 0))

    if create_test_mode:
        return feature_train, test_ids
    else:
        # print("Feature names:", list(feature_train.columns))
        return feature_train, target_train, onehot_encoder, tfidfvectorizer