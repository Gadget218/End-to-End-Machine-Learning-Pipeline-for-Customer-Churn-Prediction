from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from datetime import datetime, timedelta
import mlflow
import sys
from pathlib import Path

# Add the parent directory of `scripts` to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))

from enhanced_mlflow_tracking import train_and_log_model
from improved_drift_monitoring import check_data_drift

def retrain_model_with_drift_check():
    """
    Comprehensive model retraining workflow
    1. Check for data drift
    2. If drift detected, retrain model
    3. Compare and potentially promote new model
    """
    # Load new inference data
    new_data = pd.read_csv('recent_customer_data.csv')

    # Check data drift
    drift_detected = check_data_drift(new_data)

    if drift_detected:
        # Retrain model with previous model as baseline
        latest_model_uri = mlflow.get_latest_model_version('customer_churn_model')
        new_model = train_and_log_model(baseline_model_path=latest_model_uri)

        return True
    return False

def send_drift_notification(context):
    """Send Slack notification about model retraining"""
    slack_webhook_token = '{{ var.value.slack_webhook }}'
    slack_msg = f"""
    🚨 *Model Drift Detected* 🚨
    Timestamp: {datetime.now()}
    Action: Model Retraining Triggered
    """

    slack_alert = SlackWebhookOperator(
        task_id='slack_alert',
        webhook_token=slack_webhook_token,
        message=slack_msg,
        channel='#ml-ops-alerts'
    )
    slack_alert.execute(context)

# Default arguments for the retraining DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=15),
}

# Define the retraining DAG
with DAG(
    dag_id="customer_churn_model_maintenance",
    default_args=default_args,
    description="Advanced Model Maintenance Pipeline",
    schedule_interval="*/10 * * * *",
    start_date=datetime(2024, 12, 28),
    catchup=False,
) as dag:

    # Drift Detection and Retraining Task
    drift_and_retrain_task = PythonOperator(
        task_id="drift_detection_and_retraining",
        python_callable=retrain_model_with_drift_check,
        provide_context=True,
        on_success_callback=send_drift_notification,
    )

    # Define the task sequence
    drift_and_retrain_task
