import sys
import os


from data_preprocessing import preprocess_data
from drift_monitoring import check_data_drift
from model_training import train_and_compare_models

def retrain_model():
    try:
        print("Starting preprocessing...")
        X_train, X_test, y_train, y_test = preprocess_data("/app/scripts/Telecom_customer_churn.csv")
        print("Preprocessing complete. Checking for drift...")
        drift_detected = check_data_drift("/app/scripts/Telecom_customer_churn.csv", "/app/scripts/reference_data.csv")
        if drift_detected:
            print("Drift detected! Retraining the model...")
            train_and_compare_models(X_train, y_train, X_test, y_test)
        else:
            print("No drift detected. Skipping retraining.")
    except Exception as e:
        print(f"Error during retraining pipeline execution: {e}")
        raise

if __name__ == "__main__":
    retrain_model()
