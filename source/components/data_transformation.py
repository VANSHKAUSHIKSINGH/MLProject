import sys 
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from source.utils import save_object
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


from source.exception import CustomException
from source.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        """
            This function is responsible for data transformation.
            It will perform the following steps:
            1. Identify the numerical and categorical columns
            2. Apply the imputer on the numerical columns
            3. Apply the imputer on the categorical columns
            4. Apply the scaler on the numerical columns
            5. Apply the one hot encoder on the categorical columns
            6. Return the preprocessor object
        """
            
            
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
                
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                    
                ]
            )
                
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehotencoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
                
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")
                 
                
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
                               
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
                
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
             
            logging.info("Read the train and test data")
            
            print("Train DataFrame columns:", train_df.columns.tolist())
            print("Test DataFrame columns:", test_df.columns.tolist())
             
            logging.info("Obtaining preprocessor object")
             
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = "math_score"
            numerical_columns = ['writing_score', 'reading_score']
             
            input_features_train_df= train_df.drop(columns=[target_column_name], axis=1)
            target_features_train_df = train_df[target_column_name]
             
            input_features_test_df= test_df.drop(columns=[target_column_name], axis=1)
            target_features_test_df = test_df[target_column_name]
            
            if target_column_name in train_df.columns:
                input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
                target_feature_train_df = train_df[target_column_name]
            else:
                raise CustomException(f"Column '{target_column_name}' not found in training DataFrame.", sys)
         
        # Check if the target column exists in the testing DataFrame
            if target_column_name in test_df.columns:
                input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
                target_feature_test_df = test_df[target_column_name]
            else:
                raise CustomException(f"Column '{target_column_name}' not found in testing DataFrame.", sys)
             
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
             
            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)
             
            train_arr = np.c_ [
                   input_feature_train_arr, np.array(target_features_train_df)
            ]
            test_arr = np.c_ [
                   input_feature_test_arr, np.array(target_features_test_df)
            ]
            
            logging.info(f"Saved preprocessing object.")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
             
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )            
             
        except Exception as e:
            raise CustomException(e, sys)
        
        
# if __name__=="__main__":
#     obj = DataTransformation()
#     obj.initiate_data_transformation()
