#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
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

# Standardize the features (recommended for SVR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x.reshape(-1,1))
X_test_scaled = scaler.transform(x_.reshape(-1,1))

# Create and train the SVR model
svr_model = SVR(kernel='linear', C=1.0, epsilon=0.2)  # You can adjust the kernel, C, and epsilon parameters
svr_model.fit(X_train_scaled, y_train)

# Make predictions
svr_predictions = svr_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, svr_predictions)
r2 = r2_score(y_test, svr_predictions)

# Print the results
print("Support Vector Regression:")
print(f"Mean Squared Error: {mse:.4f}")

