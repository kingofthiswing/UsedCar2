import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import logging
import sys
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass
import os


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self, train_df):
        try:
            # Identify categorical and numerical columns
            categorical_columns = ['name', 'company', 'fuel_type']
            numerical_columns = ['year', 'kms_driven']

            # Create transformers for the columns
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            # Create the column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_columns),
                    ('cat', categorical_transformer, categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Obtain the preprocessing object
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object(train_df)

            # Target column name
            target_column_name = "price"

            # Separate features and target variables for both train and test data
            input_feature_train_df = train_df[["year", "kms_driven", "name", "company", "fuel_type"]]
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df[["year", "kms_driven", "name", "company", "fuel_type"]]
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training and testing dataframes.")

            # Apply transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Ensure arrays are 2D
            if isinstance(input_feature_train_arr, np.ndarray):
                input_feature_train_arr = np.array(input_feature_train_arr)
                input_feature_test_arr = np.array(input_feature_test_arr)
            else:
                input_feature_train_arr = input_feature_train_arr.toarray()
                input_feature_test_arr = input_feature_test_arr.toarray()

            # Reshaping the arrays to ensure the correct shape
            input_feature_train_arr = input_feature_train_arr.reshape(input_feature_train_arr.shape[0], -1)
            input_feature_test_arr = input_feature_test_arr.reshape(input_feature_test_arr.shape[0], -1)

            # Reshaping target features to match the input array dimensions
            target_feature_train_arr = target_feature_train_df.values.reshape(-1, 1)
            target_feature_test_arr = target_feature_test_df.values.reshape(-1, 1)

            # Stack the features and targets horizontally to form the final dataset
            train_arr = np.hstack((input_feature_train_arr, target_feature_train_arr))
            test_arr = np.hstack((input_feature_test_arr, target_feature_test_arr))

            # Save preprocessing object
            logging.info(f"Saved preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return the correct number of values
            return train_arr, test_arr

        except Exception as e:
            raise CustomException(e, sys)
