from zenml import logging
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class Evaluated(ABC):
    """ Abstract class definging for evaluating our model """
    """ That's where we write about our Matrics """

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """ calculate the score for the model 
            Args: y_ture
                  y_pred
            Return: None
        """
        pass

class MSE(Evaluated):
    """ Evaluation strategy that uses MSE """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE score {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE : {}".format(e))
            raise e
        
class R2(Evaluated):
    """ Evaluation strategy that uses R2 score """


    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):


        try:
            logging.info("Calculating R2 score")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 score {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating r2 score {}".format(e))
            raise e
        
class RMSE(Evaluated):
    """ Evaluation strategy that uses RMSE (Root mean squared error) """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info("MSE score {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating RMSE : {}".format(e))
            raise e