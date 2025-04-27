import os
import sys
import dill
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from source.exception import CustomException
from source.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models,param):
    try:
        report = {}    
        
        for i in range(len(list(models))):
            model=list(models.values())[i]
            para =param[list(models.keys())[i]]
            
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train, y_train) # Fit the model with the best parameters
            
            
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train) # Train the model with the best parameters
            
            # model.fit(X_train, y_train)     # Train the model
            
            y_train_pred = model.predict(X_train) # Predict on training data
            
            y_test_pred = model.predict(X_test) # Predict on test data
            
            train_model_score = r2_score(y_train, y_train_pred) # Calculate R2 score on training data
            
            test_model_score = r2_score(y_test, y_test_pred) # Calculate R2 score on training data
            
            report[list(models.keys())[i]] = test_model_score # store the test score in the report dictionary
            
        return report    
        
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)