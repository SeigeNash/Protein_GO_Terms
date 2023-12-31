#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load your dataset (replace 'your_dataset.csv' with your dataset file)
df = pd.read_csv("C:/Users/ancha/OneDrive/Desktop/Datasets/archive/submission_best_public_merge_output_col.csv")

# Encode categorical columns
encoder_sequence = LabelEncoder()
df['Header'] = encoder_sequence.fit_transform(df['Header'])

encoder_ontology = LabelEncoder()
df['Ontology'] = encoder_ontology.fit_transform(df['Ontology'])

# Assuming 'X' contains your features and 'y' contains your target variable
X = df['Ontology'] # Features
y = df['Target'] # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x = X_train.to_numpy()
x_ = X_test.to_numpy()

# Create and train the Bayesian Ridge Regression model
bayesian_regressor = BayesianRidge()
bayesian_regressor.fit(x.reshape(-1,1), y_train)

# Make predictions
bayesian_predictions = bayesian_regressor.predict(x_.reshape(-1,1))

# Evaluate the model
mse = mean_squared_error(y_test, bayesian_predictions)
r2 = r2_score(y_test, bayesian_predictions)

# Print the results
print("Bayesian Ridge Regression:")
print(f"Mean Squared Error: {mse:.4f}")

