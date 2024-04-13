import logging
import pandas as pd 
 
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig


@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
)->  RegressorMixin:
    print("working 03")
    """
    Trains the model on ingested data
    Args:
        df is ingested data
    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            ''' model = LinearRegressionModel().train(X_train, y_train) '''
            model = LinearRegressionModel()
            train_model = model.train(X_train, y_train)
            return train_model
        else:
            raise ValueError("Error {} not supported".format(config.model_name)) 
    except Exception as e:
        logging.error("Error in training model {}".format(e))
        raise e
