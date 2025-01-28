from fastapi import FastAPI
import joblib
import pandas as pd
import os

app = FastAPI()

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../best_model.pkl"))

print(f"Looking for model at: {model_path}")

try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    else:
        model = None
        print(f"Model file not found at {model_path}.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.post("/predict/")
def predict(features: dict):
    if not model:
        return {"error": "Model not loaded properly. Please check the model path."}

    try:
        # Convert input to DataFrame
        df = pd.DataFrame([features])
        print(f"Received input: {df}")

        # Make prediction
        prediction = model.predict(df)
        print(f"Prediction result: {prediction}")

        return {"prediction": int(prediction[0])}
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}
