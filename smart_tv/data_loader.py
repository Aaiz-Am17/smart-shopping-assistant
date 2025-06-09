import pandas as pd

TV_file_path = r"C:\WOLF\Private\GUNI YEAR 3\ML\TELEVISION.csv"

def TV_load_dataset():
    TV_encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for TV_encoding in TV_encodings:
        try:
            TV_df = pd.read_csv(TV_file_path, encoding=TV_encoding)
            return TV_df, f"File successfully read with encoding: {TV_encoding}"
        except UnicodeDecodeError:
            pass
    raise ValueError("Unable to decode the file with the specified encodings.")

