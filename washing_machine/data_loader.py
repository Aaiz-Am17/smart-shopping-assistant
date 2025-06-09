import pandas as pd

# Path to your dataset (update path as needed)
AC_FILE_PATH = r"Data/Washingmachine.csv

def load_ac_dataset(file_path=AC_FILE_PATH):
    """
    Load the Air Conditioner dataset from CSV file.
    Returns:
        DataFrame, str (status message)
    Raises:
        ValueError if loading fails.
    """
    try:
        df = pd.read_csv(file_path)
        return df, "Dataset loaded successfully!"
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")

def preprocess_ac_data(df, for_prediction=False):
    """
    Preprocess AC dataset:
    - Clean Power_Consumption column (extract numeric values)
    - Extract numeric Noise_level
    - Normalize Refrigerant categories
    - Fill missing values
    Args:
        df: pandas DataFrame
        for_prediction: bool, if True, skip checking Price column
    Returns:
        Processed DataFrame
    Raises:
        ValueError if required columns are missing.
    """
    required_cols = ['Power_Consumption', 'Noise_level', 'Refrigerant', 'Condenser_Coil']
    if not for_prediction:
        required_cols.append('Price')
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Clean Power_Consumption: remove non-numeric characters
    df['Power_Consumption'] = df['Power_Consumption'].astype(str).str.replace(r'[^\d.]+', '', regex=True)
    df['Power_Consumption'] = pd.to_numeric(df['Power_Consumption'], errors='coerce')
    df['Power_Consumption'].fillna(df['Power_Consumption'].median(), inplace=True)

    # Extract numeric Noise_level
    df['Noise_level'] = df['Noise_level'].astype(str).str.extract(r'(\d+)').astype(float)
    df['Noise_level'].fillna(df['Noise_level'].median(), inplace=True)

    # Normalize Refrigerant column values
    df['Refrigerant'] = df['Refrigerant'].apply(lambda x: 'R-32' if 'R-32' in str(x) else
                                                'R410a' if 'R410a' in str(x) else 'Other')
    df['Refrigerant'].fillna(df['Refrigerant'].mode()[0], inplace=True)

    # Fill missing values in Condenser_Coil with mode
    df['Condenser_Coil'].fillna(df['Condenser_Coil'].mode()[0], inplace=True)

    return df

