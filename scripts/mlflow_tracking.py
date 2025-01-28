import mlflow

def start_mlflow_tracking():
    mlflow.set_tracking_uri("http://host.docker.internal:5000")
    mlflow.set_experiment("Customer Churn")
