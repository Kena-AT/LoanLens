import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    # Example preprocessing steps (customize based on your notebook)
    df = df.dropna()  # drop missing values
    # Add feature engineering or encoding here if needed
    return df

def split_data(df, target_col='SeriousDlqin2yrs', test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    df = load_data("../data/credit_data.csv")
    df_clean = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_clean)

    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
