import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
# TODO: add necessary import

# TODO: implement the first test. Change the function name and input as needed
def test_model_type():
    """
    confirms our model is a LogisticRegression model
    """
    model = LogisticRegression()
    assert isinstance(model, LogisticRegression), "The model should be a Logistic Regression model"


def test_encoder():
    """
    confirms our encoder is OneHotEncoder()
    """
    encoder= OneHotEncoder()

    assert isinstance(encoder, OneHotEncoder), "Encoder needs to be OneHotEncoder instance"



def test_label():
    """
    Confirms our labeler is an instance of LabelBinarizer()
    """
    lb = LabelBinarizer()

    assert isinstance(lb, LabelBinarizer), "Labeler needs to be BiaryLabelizer() instance"

