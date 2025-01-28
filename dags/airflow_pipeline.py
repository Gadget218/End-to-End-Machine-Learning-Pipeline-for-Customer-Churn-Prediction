import sys
import os

# Ensure the scripts folder is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts")))

from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator

from data_preprocessing import preprocess_data
from drift_monitoring import check_data_drift
from model_training import train_and_compare_models

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
}


# Define the DAG
with DAG(
    'data_pipeline',
    default_args=default_args,
    schedule=timedelta(minutes=2),  # Trigger every 2 minutes
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    def preprocess_and_check_drift():
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
            print(f"Error during pipeline execution: {e}")
            raise

    # Define tasks
    preprocessing_task = PythonOperator(
        task_id='preprocess_and_check_drift',
        python_callable=preprocess_and_check_drift
    )
    preprocessing_task
