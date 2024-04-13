import logging
import pandas as pd 
 
from src.evaluation import MSE, R2, RMSE
from sklearn.base import RegressorMixin
from zenml import step
from typing import Annotated
from typing_extensions import Tuple

@step
def evaluate( model: RegressorMixin, 
              X_test: pd.DataFrame,
              y_test: pd.DataFrame
            )-> Tuple[
                Annotated[float, "r2_score"],
                Annotated[float, "rmse"]
            ]:
    print("working 04")

    """
    Evaluate the model on ingested data
    Args:
        df is ingested data
    """

    try:
        prediction = model.predict(X_test) # prediction:  model prediction on train data
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)

        r2_class = R2()
        r2_score = r2_class.calculate_scores(y_test, prediction)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)

        return r2_score, rmse
    except Exception as e:
        logging.error("Error in evaluating model {}".format(e))
        raise e