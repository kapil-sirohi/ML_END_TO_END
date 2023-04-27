import os, sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)

from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_mode_file_path = os.path.join('artifacts',"model.pkl")


class ModelTrainer:

    def __init__(self) :
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("splitting training and test array")
            logging.info(train_array)
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {

                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbour Regressor": KNeighborsRegressor(),
                "XGBoost Regressor" : XGBRegressor(),
                "CatBoost Regressor" : CatBoostRegressor(verbose=False),
                "Adaboost Regressor" : AdaBoostRegressor()

            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)

            best_model_score = max(sorted(model_report.values()))

            best_mode_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model= models[best_mode_name]
            
            logging.info(f"Best model found {best_mode_name} with accuracy of {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_mode_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2score = r2_score(y_test,predicted)

            return r2score

        except Exception as e:
            raise CustomException(e,sys)
            
