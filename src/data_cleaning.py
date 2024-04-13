import logging 
import pandas as pd
import numpy as np
from typing import Union
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """ Abstract class defining strategy for handling data """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreProcessStrategy(DataStrategy):

    """ Data Preprocessing. """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        
        """ Drop columns """
        try:
            drop = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivered_date",
                    "order_purchase_timestamp"
                ],
                axis= 1),
            """ filling the null values """
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True),
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True),
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True),
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True),
            # write "No review" in review_comment_message column
            data["review_comment_message"].fillna("No review", inplace=True) , 
            data = data.select_dtypes(include=(np.number)),

            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            return data
        except Exception as e:
            logging.error(f"Error in preporecessing data {e}")
            raise e

class DataDivideStrategy(DataStrategy):

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:

        try:
            X = data.drop(["review_score"], axis=1 ),
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error while Dividing the data {e}")
            raise e

""" Utilaizing both the classes """
class DataCleaning:
    def __init__ (self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.data = strategy

    def handle_data(self)-> Union[pd.DataFrame, pd.Series]:

        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data {e}")
            raise e
        