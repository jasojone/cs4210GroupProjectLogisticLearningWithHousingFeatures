# -*- coding: utf-8 -*-
"""cs4210.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VZN0nrWAxFJ86FeJEAeVg6UMbG5x0eET
Data:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=train.csv
OverLeaf:
https://www.overleaf.com/project/653bf6f606c54ddd7ea1f19d

"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Mounting Google Drive is optional and only required if you're running this on Google Colab
# from google.colab import drive
# drive.mount('/content/drive')
#

class HousingPricePrediction:
    """
    HousingPricePrediction class for the Kaggle House Prices: Advanced Regression Techniques competition.
    This class contains methods to explore the data, preprocess the data, train a model, make predictions, and prepare a submission file.
    """
    def __init__(self, train_filepath, test_filepath):
        """
        Constructor for HousingPricePrediction class to initialize the train and test data.

        Args:
            train_filepath (str): Filepath for the training data
            test_filepath (str): Filepath for the test data
        """
        self.train_data = pd.read_csv(train_filepath)
        self.test_data = pd.read_csv(test_filepath)
        self.sample_submission = None  # Will be loaded later
        self.features = None
        self.target = 'SalePrice'  # Update if the target variable is different
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.test_ids = None  # Will be used to store the Id column from the test data

    def explore_data(self):
        """
        Method to explore the data. This method prints the first 5 rows of the train and test data, and the info of the train and test data.
        """
        print(self.train_data.head())
        print(self.train_data.info())
        print(self.test_data.head())
        print(self.test_data.info())
    def preprocess_data(self):
        """ 
        Method to preprocess the data. This method separates the features and target variable, and applies transformations to the train and test data.
        """
        # Separate features and target variable
        self.test_ids = self.test_data['Id']  # Store the 'Id' column from the test data
        self.train_ids = self.train_data['Id']  # Store the 'Id' column from the train data
        self.X_train = self.train_data.drop(columns=[self.target, 'Id'])
        self.y_train = self.train_data[self.target]
        self.X_test = self.test_data.drop(columns=['Id'])

        # Define the transformers for numeric and categorical data
        numeric_transformer = SimpleImputer(strategy='median')
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Identify the numeric and categorical features
        numeric_features = self.X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.X_train.select_dtypes(include=['object']).columns

        # Bundling preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Applying transformations to train and test data
        self.X_train = preprocessor.fit_transform(self.X_train)
        self.X_test = preprocessor.transform(self.X_test)



    def train_model(self):
        """
        Method to train a model. This method trains a linear regression model on the train data.
        """
        # Training a linear regression model
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """
        Method to evaluate the model. This method evaluates the trained model using cross-validation or a validation set.
        """
        # Evaluate the trained model using cross-validation or a validation set
        # Placeholder: add evaluation steps if necessary
        pass

    def make_predictions(self):
        """
        Method to make predictions. This method makes predictions on the test data using the trained model.

        Returns:
            predictions (numpy.ndarray): Predictions made by the model on the test data
        """
        # Making predictions on the test data
        predictions = self.model.predict(self.X_test)
        return predictions

    def prepare_submission(self, predictions, submission_filepath):
        """
        Method to prepare a submission file. This method prepares a submission file using the predictions made by the model.

        Args:
            predictions (numpy.ndarray): Predictions made by the model on the test data
            submission_filepath (str): Filepath to save the submission file
        """
        # Preparing the submission file
        self.sample_submission = pd.DataFrame({'Id': self.test_ids, 'SalePrice': predictions})
        self.sample_submission.to_csv(submission_filepath, index=False)
        print(f"Submission file saved to: {submission_filepath}")

if __name__ == "__main__":
    # Replace placeholders with the actual file paths
    train_filepath = "train.csv"
    test_filepath = "test.csv"
    submission_filepath = "submission.csv"

    project = HousingPricePrediction(train_filepath=train_filepath, test_filepath=test_filepath)
    print("Exploring data...--------------------------------------------------")
    project.explore_data()
    print("Training data info:--------------------------------------------------")
    print(project.train_data.info())
    print("Training data shape:--------------------------------------------------")
    print(project.train_data.shape)
    print("Training data columns:--------------------------------------------------")
    print(project.train_data.columns)

    print("Preprocessing data...--------------------------------------------------")
    project.preprocess_data()
    print("Training model...--------------------------------------------------")
    project.train_model()
    print("Evaluating model...--------------------------------------------------")
    project.evaluate_model()
    print("Making predictions...--------------------------------------------------")
    predictions = project.make_predictions()
    print("Preparing submission...--------------------------------------------------")
    project.prepare_submission(predictions, submission_filepath)
