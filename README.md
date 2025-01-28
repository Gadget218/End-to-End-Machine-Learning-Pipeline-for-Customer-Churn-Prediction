# Customer Churn Prediction

## Instructions

1. Clone the repository.
2. Build the Docker image:
docker build -t customer_churn_project .
3. Run the Docker container:
docker run -p 8000:8000 -p 8080:8080 -p 5000:5000 --name customer_churn_project customer_churn_project
4. Access FastAPI at `http://localhost:8000`.
5. Set up Airflow for retraining:
airflow webserver -p 8080
airflow scheduler
7. Access MLflow at `http://localhost:5000`.
