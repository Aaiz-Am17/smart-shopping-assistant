import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def TV_preprocess_data(TV_df):
    TV_df['TV_OS_Category'] = TV_df['Operating_system'].apply(
        lambda x: 'Android' if 'Android' in x else
                  'Linux' if 'Linux' in x else
                  'Google TV' if 'Google TV' in x else 'Other'
    )
    TV_df['TV_Picture_Quality_Category'] = TV_df['Picture_quality'].apply(
        lambda x: '4K' if '4K' in x else
                  'Full HD' if 'Full HD' in x else
                  'HD Ready' if 'HD Ready' in x else 'Other'
    )
    TV_df['TV_Speaker_Output_Category'] = TV_df['Speaker'].str.extract(r'(\d+)')[0].astype(int).apply(
        lambda x: '10-30W' if 10 <= x <= 30 else
                  '30-60W' if 30 < x <= 60 else
                  '60-90W' if 60 < x <= 90 else '90+W'
    )
    return TV_df

def get_features_and_target(TV_df):
    TV_X = TV_df[['TV_OS_Category', 'TV_Picture_Quality_Category', 'TV_Speaker_Output_Category', 'Frequency', 'channel']]
    TV_y = TV_df['current_price']
    return TV_X, TV_y

def get_column_transformer():
    TV_categorical_features = ['TV_OS_Category', 'TV_Picture_Quality_Category', 'TV_Speaker_Output_Category', 'Frequency', 'channel']
    TV_column_transformer = ColumnTransformer(
        transformers=[('onehot', OneHotEncoder(handle_unknown='ignore'), TV_categorical_features)],
        remainder='passthrough'
    )
    return TV_column_transformer
