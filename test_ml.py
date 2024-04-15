import pytest
import numpy as np
import pandas as pd
from ml.model import train_model, compute_model_metrics
from ml.data import apply_label
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, fbeta_score

# TODO: add necessary import
X_train = np.array([[1,4,8],[5,8,8],[6,7,4]])
y_train = np.array([0,1,0])
X = pd.DataFrame(X_train)
X['y'] = y_train
# TODO: implement the first test. Change the function name and input as needed
def test_model_type():
    """
    confirms our model is a LogisticRegression model
    """
    model = train_model(X_train, y_train)
    assert isinstance(model, LogisticRegression), "The model should be a Logistic Regression model"

# Define test cases
@pytest.mark.parametrize("inference, expected", [
    ([1], ">50K"),
    ([0], "<=50K"),
    ([1, 0], ">50K"),  # Additional test case to check only the first element of inference is considered
    ([0, 1], "<=50K"),  # Additional test case to check only the first element of inference is considered
    ([1, 1], ">50K"),  # Test case for unexpected input
    ([0, 0], "<=50K")  # Test case for unexpected input
])
def test_apply_label(inference, expected):
    """
    Test the apply_label function
    """
    assert apply_label(inference) == expected

# Define test cases
@pytest.mark.parametrize("y, preds, expected_precision, expected_recall, expected_fbeta", [
    (np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]), 0.5, 0.5, 0.5),  # Balanced case
    (np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]), 1.0, 1.0, 1.0),  # Perfect prediction
    (np.array([1, 0, 1, 0]), np.array([0, 1, 0, 1]), 0.0, 0.0, 0.0),  # All wrong predictions
    (np.array([]), np.array([]), 1.0, 1.0, 1.0),  # Empty input arrays
])

def test_compute_model_metrics(y, preds, expected_precision, expected_recall, expected_fbeta):
    """
    Test the compute_model_metrics function
    """
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(expected_precision, abs=1e-5)
    assert recall == pytest.approx(expected_recall, abs=1e-5)
    assert fbeta == pytest.approx(expected_fbeta, abs=1e-5)

