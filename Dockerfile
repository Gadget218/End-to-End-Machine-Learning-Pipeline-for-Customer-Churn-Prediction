# Use an official Python image as the base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Upgrade pip, setuptools, and wheel to ensure compatibility
RUN pip install --upgrade pip setuptools wheel

# Copy requirements file and install dependencies
COPY requirements.txt .
# Install dependencies in smaller batches for reliability
RUN pip install pandas numpy scikit-learn
RUN pip install mlflow fastapi uvicorn
RUN pip install apache-airflow great-expectations==0.15.39
RUN pip install xgboost

# Copy the entire project into the container
COPY . /app

# Expose ports (Airflow: 8080, FastAPI: 8000, MLflow: 5000)
EXPOSE 8000 8080 5000

# Set environment variables for Airflow
ENV AIRFLOW_HOME=/app/dags

# Set entrypoint for the container to start the FastAPI app
CMD ["uvicorn", "app.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
