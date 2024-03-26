README
This Python script predicts house prices using a neural network model. The script uses the pandas library for data manipulation, sklearn for data splitting, and keras for defining, compiling, training, and evaluating the neural network model.

CODE DESCRIPTION
Data Preprocessing: The script begins by removing certain columns from the train and test datasets. It then merges the one-hot encoded data with the original train and test datasets.
Feature Selection: The script selects all columns from the train and test datasets as features, excluding ‘SalePrice’ and ‘Id’. The target variable ‘SalePrice’ is stored in y.
Train-Test Split: The script splits the data into training and testing sets using a 70-30 split.
Model Definition: The script defines a neural network model with four hidden layers, each with 64 neurons. The output layer has one neuron as this is a regression problem.
Model Compilation: The script compiles the model using the Adam optimizer and Mean Squared Error (MSE) as the loss function.
Model Training: The script trains the model for 100 epochs.
Model Evaluation: The script evaluates the model on the training and testing sets and prints the MSE.
Predictions: The script makes predictions on the training and testing sets.

DATASET DESCRIPTION
The dataset used in this script is related to house prices and includes the following files:
train.csv: This is the training set that contains the details of the houses along with their sale prices.
test.csv: This is the test set that contains the details of the houses. The sale prices of these houses are what we aim to predict.
data_description.txt: This file contains a full description of each column. It was originally prepared by Dean De Cock but has been lightly edited to match the column names used here.
sample_submission.csv: This is a benchmark submission file derived from a linear regression on year and month of sale, lot square footage, and number of bedrooms.

REQUIREMENTS
To run this script, you need Python 3 and the following libraries installed:
pandas
sklearn
keras
