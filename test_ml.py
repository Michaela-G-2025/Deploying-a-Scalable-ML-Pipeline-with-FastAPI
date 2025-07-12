import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

# Import functions to test
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
LABEL_COLUMN = 'salary'

@pytest.fixture(scope="module")
def real_data_sample():
    """
    Fixture to load a small, representative sample of the real census.csv data
    for testing.
    """
    project_path = "/home/missm/Deploying-a-Scalable-ML-Pipeline-with-FastAPI"
    data_path = os.path.join(project_path, "data", "census.csv")
    full_data = pd.read_csv(data_path)
    sample_size = 75
    sample_df = full_data.sample(n=sample_size)
    return sample_df

@pytest.fixture(scope="module")
def processed_data_and_model(real_data_sample): # <-- FIX: Corrected fixture name
    """
    Processes the sampled real data and trains a model once for all tests in this module.
    Returns X_processed, y_processed, encoder, lb, and the trained model.
    """
    X_processed, y_processed, encoder, lb = process_data(
        real_data_sample.copy(), # Pass a copy to avoid modifying the original fixture data
        categorical_features=CAT_FEATURES,
        label=LABEL_COLUMN,
        training=True
    )
    model = train_model(X_processed, y_processed) # <-- FIX: X_processed and y_processed are now correctly scoped
    # Return all necessary components for various tests
    return X_processed, y_processed, encoder, lb, model

### Begin Tests Here ###

def test_processed_data_return_types(processed_data_and_model):
    """
    Test to test the expected returned values from the process_data function
    """
    X_processed, y_processed, encoder, lb, _ = processed_data_and_model

    # Test process_data return types
    assert isinstance(X_processed, np.ndarray), "X_processed should be a numpy array"
    assert isinstance(y_processed, np.ndarray), "y_processed should be a numpy array"
    assert isinstance(encoder, OneHotEncoder), "Encoder should be a OneHotEncoder"
    assert isinstance(lb, LabelBinarizer), "LabelBinarizer should be a LabelBinarizer"



def test_inference_data_return_types(processed_data_and_model):
    """
    Test to test the expected returned values from the inference function
    """
    X_processed_for_inference, _, _, _, model = processed_data_and_model
    # Use a small slice of processed data for inference
    predictions = inference(model, X_processed_for_inference[:5])  # Take first 5 for test

    assert isinstance(predictions, np.ndarray), "X_inference should be a numpy array"
    assert np.all(np.isin(predictions, [0, 1])), "Model predictions should be strictly 0 or 1"
    assert predictions.dtype == np.int64, "Prediction dtype should be integer"

def test_ml_model_uses_expected_algorithm(processed_data_and_model):
    """
    Test if the ML model uses the expected algorithm (LogisticRegression).
    """
    _, _, _, _, model = processed_data_and_model
    assert isinstance(model, LogisticRegression), "The trained model should be a LogisticRegression instance"
    assert hasattr(model, 'coef_'), "LogisticRegression model should have 'coef_' attribute after training"
    assert hasattr(model, 'intercept_'), "LogisticRegression model should have 'intercept_' attribute after training"

def test_compute_metrics_returns_expected_value():
    """
    Test if the computing metrics functions return the expected value with known inputs.
    """
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert precision == pytest.approx(0.8), "Precision calculation is incorrect"
    assert recall == pytest.approx(0.8), "Recall calculation is incorrect"
    assert fbeta == pytest.approx(0.8), "F1 Score calculation is incorrect"

def test_processed_data_shapes(processed_data_and_model, real_data_sample):
    """
    Test that training and test datasets have the expected size.
    Specifically checks processed data shapes and consistency with original.
    """
    X_processed, y_processed, _, _, _ = processed_data_and_model
    n_samples = real_data_sample.shape[0]

    assert X_processed.shape[0] == n_samples, "Processed X should have same number of rows as input"
    assert y_processed.shape[0] == n_samples, "Processed y should have same number of rows as input"
