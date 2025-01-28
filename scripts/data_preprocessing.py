import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Print column names to verify
    print("Columns in the dataset:", df.columns)

    # Handle missing values
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df = df.dropna()

    # Encode categorical features
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Split into train-test
    X = df.drop('churn', axis=1)  # Update this column name after verifying
    y = df['churn']              # Update this column name after verifying
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
