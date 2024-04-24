import pandas as pd

"""
=====================================================================
---------------------- Code Section --------------------------------
=====================================================================
"""


def clean_and_save_dataset(root_path, output_path, csv_name):
    """Loads, cleans, and saves the dataset."""
    # Load the dataset
    df = pd.read_csv(root_path + csv_name, encoding='ISO-8859-1')

    # Check for missing values
    if df.isnull().values.any():
        print("Warning: Missing values found. Rows with missing values will be dropped.")
        df = df.dropna()

    # Check data types and correct if necessary
    if not pd.api.types.is_string_dtype(df['text']):
        print("Warning: Text column data type is not string. Attempting to convert.")
        df['text'] = df['text'].astype(str)

    if not pd.api.types.is_object_dtype(df['sentiment']):
        print("Warning: Sentiment column data type is not object. Attempting to convert.")
        df['sentiment'] = df['sentiment'].astype(str)

    # Validate sentiment values
    valid_sentiments = ['neutral', 'negative', 'positive']
    if not df['sentiment'].isin(valid_sentiments).all():
        print("Warning: Invalid sentiment values detected. These entries will be removed.")
        df = df[df['sentiment'].isin(valid_sentiments)]

    # Save the cleaned dataset with index
    output_csv_name = csv_name  # Use the same file name
    df = df[['text', 'sentiment']]  # Keep only the necessary columns
    df.to_csv(output_path + output_csv_name, index=True,
              index_label='Index', quoting=1, sep=',', encoding='utf-8')

    print(
        f"Dataset has been cleaned and saved to '{output_path + output_csv_name}' with index.")


"""
=====================================================================
---------------------- Settings Section -----------------------------
=====================================================================
"""
if __name__ == "__main__":
    # Path where the CSV file is stored
    ROOT_PATH = 'sentimaent_sample/data/Raw/'
    # Path where the cleaned CSV will be saved
    OUTPUT_PATH = 'sentimaent_sample/data/Processed/'
    # Name of the CSV file containing the training data
    CSV_NAME = 'test.csv'

    clean_and_save_dataset(ROOT_PATH, OUTPUT_PATH, CSV_NAME)
