import pandas as pd
from great_expectations.dataset.pandas_dataset import PandasDataset

def check_data_drift(new_data_path, reference_data_path):
    try:
        # Load new and reference datasets
        new_data = pd.read_csv(new_data_path)
        reference_data = pd.read_csv(reference_data_path)

        # Verify if the required column exists in the dataset
        column_to_check = "rev_Mean"  
        if column_to_check not in new_data.columns or column_to_check not in reference_data.columns:
            raise KeyError(f"Column '{column_to_check}' not found in one of the datasets.")

        # Initialize Great Expectations datasets
        ge_new_data = PandasDataset(new_data)
        ge_reference_data = PandasDataset(reference_data)

        # Check for drift in mean of the specified column
        drift_detected = not ge_new_data.expect_column_mean_to_be_between(
            column_to_check,
            ge_reference_data[column_to_check].mean() - 100,  
            ge_reference_data[column_to_check].mean() + 100
        )["success"]

        return drift_detected
    except KeyError as key_error:
        print(f"KeyError: {key_error}")
        raise
    except Exception as e:
        print(f"Error during drift detection: {e}")
        raise
