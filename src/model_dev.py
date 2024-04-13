import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression


class Model(ABC):

    @abstractmethod
    def train(self, X_train, y_train):


        """ Trains the model
            Agrs: X_train > Training data
                y_train > Training label
            Return: None
        """
        pass

class LinearRegressionModel(Model):
    def train(self, X_train, y_train, **kwargs):
        
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Training model completed")
        except Exception as e:
            logging.error(f"Error while training model {e}")
            raise e
        