# mlflow_tracking.py
import os
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
from pathlib import Path

def setup_mlflow():
    """Configure MLflow tracking URI and create necessary directories"""
    mlflow_tracking_dir = Path("mlruns")
    mlflow_tracking_dir.mkdir(exist_ok=True)

    # Set the tracking URI to the local directory
    mlflow.set_tracking_uri(f"file:{mlflow_tracking_dir.absolute()}")

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    return str(artifacts_dir)

def register_model_to_production(model, run_id):
    """Register the model to MLflow registry and transition it to Production"""
    try:
        # Register the model
        model_version = mlflow.register_model(
            f"runs:/{run_id}/sklearn-model",
            "customer_churn_model"
        )
        print(f"Model registered as version {model_version.version}")

        # Transition the model to Production stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="customer_churn_model",
            version=model_version.version,
            stage="Production"
        )
        print("Model transitioned to Production stage")

        return model_version
    except Exception as e:
        print(f"Error in model registration: {e}")
        return None

def train_and_log_model(data_path='Telecom_customer_churn.csv'):
    """Train and log model with enhanced error handling and artifact management"""
    artifacts_dir = setup_mlflow()

    # Ensure the correct path is used inside the Docker container
    if not os.path.isabs(data_path):
        data_path = os.path.join('/app/scripts', data_path)

    try:
        data = pd.read_csv(data_path)
        print(f"Successfully loaded data from {data_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file '{data_path}' not found. Please check the file path.")

    # Preprocess data
    categorical_columns = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    if 'churn' not in data.columns:
        raise KeyError("The 'churn' column is missing in the dataset.")

    X = data.drop('churn', axis=1)
    y = data['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set up MLflow experiment
    experiment_name = "Customer_Churn_Prediction"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="random_forest_run") as run:
        print(f"Started MLflow run with ID: {run.info.run_id}")

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        print("Model training completed.")

        # Save model locally
        model_path = os.path.join(artifacts_dir, 'rf_model.pkl')
        joblib.dump(rf_model, model_path)
        print(f"Model saved locally at: {model_path}")

        # Calculate and log metrics
        y_pred = rf_model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log model to MLflow
        mlflow.sklearn.log_model(rf_model, "sklearn-model")

        # Register model to production
        model_version = register_model_to_production(rf_model, run.info.run_id)

        print(f"\nMLflow run completed. Run ID: {run.info.run_id}")
        return rf_model, run.info.run_id, model_path

if __name__ == "__main__":
    try:
        model, run_id, model_path = train_and_log_model()
        print(f"\nSuccess! Model trained and logged with run ID: {run_id}")
        print(f"Model saved locally at: {model_path}")
        print("To view results, run 'mlflow ui' in your terminal and open http://localhost:5000")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
