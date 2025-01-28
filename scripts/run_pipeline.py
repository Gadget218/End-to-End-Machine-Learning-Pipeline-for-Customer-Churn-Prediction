from data_preprocessing import preprocess_data
from model_training import train_and_compare_models
import os
import pandas as pd

def main():
    # Get the absolute directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the dataset
    data_path = os.path.join(script_dir, "Telecom_customer_churn.csv")

    # Ensure the dataset exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please provide the dataset.")

    # Step 1: Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(data_path)
    print("Data preprocessing complete.")

    # Step 2: Train the model
    print("Training the model...")
    train_and_compare_models(X_train, y_train, X_test, y_test)
    print("Model training complete.")

if __name__ == "__main__":
    main()
