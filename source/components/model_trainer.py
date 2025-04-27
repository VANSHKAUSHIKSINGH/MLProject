import os
import sys
from dataclasses import dataclass 

from catboost import CatBoostRegressor
from sklearn.ensemble import  (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# r2_score = None
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from source.exception import CustomException
from source.logger import logging   
from source.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config  = ModelTrainerConfig()
        
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('splitting training and test data')
            X_train, y_train, X_test, y_test =(
                train_array[:,:-1], 
                train_array[:,-1],
                test_array[:,:-1], 
                test_array[:,-1]

            )
            models = {
                'RandomForestRegressor': RandomForestRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'XGBRegressor': XGBRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose=0),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'LinearRegression': LinearRegression()
            }
        #     params = {
        #         "Descision Tree": {
        #             'criterion': ['squared_error', 'freidman_mse', 'absolute_error', 'poisson'],
        #             # 'splitter': ['best', 'random'],
        #             # 'max_features': ['sqrt', 'log2'],
        #         },
        #         "Random_Forest": {
        #             #'criterion': ['squared_error', 'absolute_error', 'poisson'],
        #             # 'max_features': ['sqrt', 'log2', 'None'],
                    
        #             'n_estimators': [8,16,32,64,128,256]
                    
        #         },
        #         "Gradient Boosting":{
        #             # 'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'].
        #             'learning_rate': [0.01, 0.1, 0.05, 0.001],
        #             'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
        #             #'criterion': ['squared_error', 'absolute_error', 'poisson'],
        #             # 'max_features': ['sqrt', 'log2', 'None'],
        #             'n_estimators': [8,16,32,64,128,256]
        #         },
        #         "linear Regression": {},
        #         "XGBRegressor": {
        #             'learning_rate': [0.01, 0.1, 0.05, 0.001],
        #             'n_estimators': [8,16,32,64,128,256]
        #         },
        #         "CatBoostRegressor": {
        #             'depth': [6, 8, 10, 12],
        #             'learning_rate': [0.01, 0.1, 0.05, 0.001],
        #             'iternations': [30,50,100]
        #         },
        #         "AdaBoostRegressor": {
        #             'learning_rate': [0.01, 0.1, 0.05, 0.001],
        #             # 'loss': ['linear', 'square', 'exponential'],
        #             'n_estimators': [8,16,32,64,128,256]
        #         }
                
        #     }
            
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models)
            
            ## To get the best model score from the dictionary 
            best_model_score = max(sorted(model_report.values()))
            
            ## To get the best model name from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)
            
            r2 = r2_score(y_test, predicted)
            return r2
        
        
        
        except Exception as e:
            raise CustomException(e, sys)
