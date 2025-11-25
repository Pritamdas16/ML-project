
            

import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing datasets")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "KNN": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False)
            }

            params = {
                "LinearRegression": {},
                "DecisionTree": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "max_depth": [None, 5, 10],
                },
                "RandomForest": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 5, 10],
                },
                "GradientBoosting": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [50, 100],
                },
                "AdaBoost": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [50, 100],
                },
                "KNN": {
                    "n_neighbors": [3, 5, 7],
                },
                "XGBRegressor": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [50, 100],
                },
                "CatBoostRegressor": {
                    "depth": [4, 6, 10],
                    "learning_rate": [0.01, 0.05],
                    "iterations": [50, 100],
                }
            }

            # Run model evaluation from utils.py
            model_report: dict = evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )

            # Get best model name & score
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            best_model = models[best_model_name]

            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")

            # Save final model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
