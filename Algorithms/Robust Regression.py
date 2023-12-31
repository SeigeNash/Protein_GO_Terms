#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import statsmodels.api as sm
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

# Create a robust regression model
model = sm.RLM(y, X, M=sm.robust.norms.HuberT())  # Huber's T norm is robust to outliers

# Fit the robust regression model
results = model.fit()

# Print the summary of the robust regression results
print(results.summary())

# Make predictions
predictions = results.predict(X)

# Evaluate the model (you can use your own evaluation metrics)
# For example, you can calculate Mean Absolute Error (MAE)
mae = abs(predictions - y).mean()
print(f"Mean Absolute Error: {mae:.4f}")

