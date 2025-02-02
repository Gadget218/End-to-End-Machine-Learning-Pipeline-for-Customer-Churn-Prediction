# Customer Churn Prediction

## Instructions

### 1. Clone the Repository
```bash
git clone <repository_link>
cd <repository_directory>
```

### 2. Build the Docker Image
```bash
docker build -t customer_churn_project .
```

### 3. Run the Docker Container
```bash
docker run -p 8000:8000 -p 8080:8080 -p 5000:5000 --name customer_churn_project customer_churn_project
```

### 4. Access FastAPI
Go to:
```
http://localhost:8000
```
(Alternatively, use `127.0.0.1:8000`)

### 5. Set Up Airflow for Retraining

1. **Enter the Docker container:**
```bash
docker exec -it customer_churn_project bash
```

2. **Initialize the Airflow database:**
```bash
airflow db init
```

3. **Create an Airflow user:**
```bash
airflow users create \
    --username airflow \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email example@mail.com \
    --password airflow
```

4. **Install `nano` text editor in the container:**
```bash
apt-get update && apt-get install -y nano
```

5. **Modify the DAGs folder path:**
```bash
nano /app/dags/airflow.cfg
```
Change the `dags_folder` to:
```
dags_folder = /app/dags/pipeline_dags
```
**Note:** To save changes, press `Ctrl+O`, then `Enter`. To exit, press `Ctrl+X`.

6. **Move the DAG files to the new path:**
```bash
mkdir -p /app/dags/pipeline_dags
mv /app/dags/*.py /app/dags/pipeline_dags/
```

7. **Set up Airflow logs ignoring:**
```bash
nano /app/dags/.airflowignore
```
Add the following line:
```
logs/*
```

8. **Run Airflow webserver and scheduler:**

- **Webserver (in one terminal):**
```bash
airflow webserver -p 8080
```

- **Scheduler (in a different terminal):**
```bash
docker exec -it customer_churn_project bash
airflow scheduler
```

9. **Access Airflow:**
```
http://127.0.0.1:8080
```

### 6. Access MLflow

1. **Enter the Docker container:**
```bash
docker exec -it customer_churn_project bash
```

2. **Run MLflow:**
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

3. **Access MLflow:**
```
http://localhost:5000
```

---

You can now monitor retraining workflows in Airflow and track experiments in MLflow!

