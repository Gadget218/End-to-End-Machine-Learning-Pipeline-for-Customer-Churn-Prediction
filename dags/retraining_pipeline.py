import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts")))

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from data_preprocessing import preprocess_data
from model_training import train_and_compare_models

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'retraining_pipeline',
    default_args=default_args,
    schedule=timedelta(minutes=5),  # Run every 2 minutes
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    def retrain_model():
        X_train, X_test, y_train, y_test = preprocess_data("/app/scripts/Telecom_customer_churn.csv")
        train_and_compare_models(X_train, y_train, X_test, y_test)

    retrain_task = PythonOperator(
        task_id='retrain_model',
        python_callable=retrain_model
    )

    retrain_task
