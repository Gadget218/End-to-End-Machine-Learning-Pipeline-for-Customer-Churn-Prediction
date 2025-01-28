import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn

# Set up MLflow tracking URI
mlflow.set_tracking_uri("http://host.docker.internal:5000")  # Ensure this matches your MLflow server URI
mlflow.set_experiment("Customer Churn")  # Use an appropriate experiment name

def train_and_compare_models(X_train, y_train, X_test, y_test):
    best_model = None
    best_roc_auc = 0.0
    best_model_name = ""

    # Define models to compare
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"),
    }

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            # Train the model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]  # For ROC AUC, we need probabilities

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)

            # Log metrics to MLflow
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("roc_auc", roc_auc)

            # Log the model to MLflow
            mlflow.sklearn.log_model(model, artifact_path=f"{model_name}_model")

            # Update the best model if this model has a better ROC AUC
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_model = model
                best_model_name = model_name

            print(f"Model: {model_name}, Accuracy: {accuracy}, Recall: {recall}, ROC AUC: {roc_auc}")

    # Save the best model locally
    if best_model is not None:
        print(f"Best Model: {best_model_name} with ROC AUC: {best_roc_auc}")
        joblib.dump(best_model, "best_model.pkl")
        with mlflow.start_run(run_name='best_model'):
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("roc_auc", best_roc_auc)
            mlflow.log_artifact("best_model.pkl")
            mlflow.sklearn.log_model(best_model, artifact_path="best_model")
