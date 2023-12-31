#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

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

# Create and fit the Huber Regressor
huber = HuberRegressor(epsilon=1.35)  # You can adjust the epsilon parameter
huber.fit(x.reshape(-1,1), y_train)

# Make predictions on the test set
y_pred = huber.predict(x_.reshape(-1,1))

# Evaluate the model (you can use your own evaluation metrics)
# For example, you can calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse:.4f}")

