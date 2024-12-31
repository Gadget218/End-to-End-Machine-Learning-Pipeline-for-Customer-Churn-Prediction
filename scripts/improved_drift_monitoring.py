import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metrics import (
    DataDriftPreset,
    TargetDriftPreset,
    DataQualityPreset
)
import logging
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def comprehensive_drift_detection(
    baseline_data: pd.DataFrame,
    current_data: pd.DataFrame,
    target_column: str = 'churn'
):
    """
    Comprehensive data and target drift detection

    Args:
        baseline_data (pd.DataFrame): Original training dataset
        current_data (pd.DataFrame): New inference dataset
        target_column (str): Name of the target variable

    Returns:
        dict: Drift detection results
    """
    try:
        # Preprocess data to ensure consistent columns
        baseline_data = baseline_data.dropna()
        current_data = current_data.dropna()

        # Ensure same columns are present
        common_columns = list(set(baseline_data.columns) & set(current_data.columns))
        baseline_data = baseline_data[common_columns]
        current_data = current_data[common_columns]

        # Create comprehensive drift report
        report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
            DataQualityPreset()
        ])

        report.run(
            reference_data=baseline_data,
            current_data=current_data
        )

        # Convert report to dictionary
        drift_report = report.as_dict()

        # Analyze drift results
        data_drift = drift_report['metrics'][0]['result']['dataset_drift']
        target_drift = drift_report['metrics'][1]['result']['target_drift']

        drift_detected = data_drift or target_drift

        # Log drift metrics to MLflow
        with mlflow.start_run():
            mlflow.log_metrics({
                'data_drift': data_drift,
                'target_drift': target_drift
            })

        logger.info(f"Drift Detection: Data Drift={data_drift}, Target Drift={target_drift}")

        return {
            'drift_detected': drift_detected,
            'data_drift': data_drift,
            'target_drift': target_drift,
            'full_report': drift_report
        }

    except Exception as e:
        logger.error(f"Drift detection error: {e}")
        return {
            'drift_detected': False,
            'error': str(e)
        }

def check_data_drift(current_data: pd.DataFrame):
    """
    Simplified drift detection interface

    Args:
        current_data (pd.DataFrame): New inference dataset

    Returns:
        bool: Whether drift was detected
    """
    baseline_data = pd.read_csv("baseline_data.csv")
    drift_results = comprehensive_drift_detection(baseline_data, current_data)
    return drift_results['drift_detected']
