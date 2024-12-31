# Install required packages
# !pip install apache-airflow pandas numpy scikit-learn

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add the parent directory of `scripts` to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))

from enhanced_mlflow_tracking import train_and_log_model


# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define the DAG
with DAG(
    dag_id="customer_churn_pipeline",
    default_args=default_args,
    description="Airflow pipeline for Customer Churn Prediction",
    schedule="*/10 * * * *",  # Adjust as needed
    start_date=datetime(2024, 12, 28),
    catchup=False,
) as dag:

    # Task: Train and log the model using MLflow
    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_and_log_model,
    )

    # Define the task sequence
    train_model_task
