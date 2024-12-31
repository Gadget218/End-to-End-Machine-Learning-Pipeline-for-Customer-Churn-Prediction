from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from fastapi.responses import JSONResponse, Response
import joblib
import pandas as pd
import mlflow
import logging
from prometheus_client import Counter, Summary, generate_latest
from typing import List
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    try:
        # First try to load from local file
        model_path = os.path.join('artifacts', 'rf_model.pkl')
        if os.path.exists(model_path):
            logger.info(f"Loading model from local file: {model_path}")
            return joblib.load(model_path)
        else:
            logger.info("Local model not found, attempting to load from MLflow")
            mlflow.set_tracking_uri("file:mlruns")
            return mlflow.sklearn.load_model(f"models:/customer_churn_model/Production")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError("Failed to load model")

# Initialize model
model = load_model()

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-time churn prediction with model monitoring",
    version="1.0.0"
)

# Input validation model
class CustomerData(BaseModel):
    features: List[float]

    @field_validator('features')
    @classmethod
    def validate_features(cls, features):
        if not hasattr(model, 'n_features_in_'):
            raise ValueError("Model not properly initialized")
        if len(features) != model.n_features_in_:
            raise ValueError(f"Expected {model.n_features_in_} features, got {len(features)}")
        return features

# Prometheus Metrics
predictions_total = Counter('predictions_total', 'Total number of predictions')
churn_predictions = Counter('churn_predictions', 'Total churn predictions')
prediction_latency = Summary('prediction_latency_seconds', 'Prediction time')

@app.get("/")
async def root():
    """
    Root endpoint that provides basic API information and available endpoints.
    """
    return {
        "message": "Welcome to the Customer Churn Prediction API",
        "available_endpoints": {
            "POST /predict": "Make churn predictions",
            "GET /metrics": "Get Prometheus metrics",
            "GET /model_info": "Get model information",
            "GET /docs": "OpenAPI documentation",
            "GET /redoc": "ReDoc documentation"
        }
    }

@app.post("/predict/")
@prediction_latency.time()
def predict_churn(data: CustomerData):
    """
    Make predictions for customer churn.
    """
    try:
        predictions_total.inc()

        # Predict
        input_data = pd.DataFrame([data.features])
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1]

        # Update metrics
        if prediction[0] == 1:
            churn_predictions.inc()

        # Log prediction
        logger.info(f"Prediction: {prediction[0]}, Probability: {probability[0]}")

        return {
            "prediction": "Churned" if prediction[0] == 1 else "Not Churned",
            "churn_probability": float(probability[0])
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    """
    Get Prometheus metrics for monitoring.
    """
    return Response(generate_latest(), media_type="text/plain")

@app.get("/model_info")
def get_model_info():
    """
    Get information about the loaded model.
    """
    return {
        "model_name": "Customer Churn Predictor",
        "features_count": model.n_features_in_,
        "model_type": type(model).__name__
    }

@app.exception_handler(404)
async def custom_404_handler(request, exc):
    """
    Custom handler for 404 Not Found errors.
    """
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Not Found",
            "available_endpoints": {
                "GET /": "API information",
                "POST /predict": "Make predictions",
                "GET /metrics": "Prometheus metrics",
                "GET /model_info": "Model information",
                "GET /docs": "API documentation"
            }
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
