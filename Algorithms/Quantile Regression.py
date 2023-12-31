#!/usr/bin/env python
# coding: utf-8

# In[4]:


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

# Choose the quantile(s) you want to estimate (e.g., 0.25 for the 25th percentile)
quantiles = [0.25, 0.50, 0.75]  # You can choose multiple quantiles

# Perform quantile regression for each specified quantile
for quantile in quantiles:
    model = sm.QuantReg(y, X)
    results = model.fit(q=quantile)

    # Print the summary of the quantile regression results
    print(f"Quantile: {quantile}")
    print(results.summary())

    # Make predictions
    predictions = results.predict(X)

    # Evaluate the model (you can use your own evaluation metrics)
    # For example, you can calculate Mean Absolute Error (MAE)
    mae = abs(predictions - y).mean()
    print(f"Mean Absolute Error: {mae:.4f}\n")

