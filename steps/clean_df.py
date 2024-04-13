import logging
import pandas as pd 

from zenml import step
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    print("working 02")
    """ 
        Clean the data and divide into train and test 
        Args:   df: raw data
        Return: X_train: train data
                X_test: test data
                y_train: train lable
                y_test: test lable
    """
    
    try:
        process_strategy = DataDivideStrategy
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        process_strategy = DataPreProcessStrategy
        data_cleaning = DataCleaning(processed_data, process_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning complete")
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logging.error(f"Error in cleaning data {e}")
        raise e
