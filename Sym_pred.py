import pickle
import numpy as np
from flask import Flask, render_template, request


# Load the trained Logistic Regression model from the pickle file
with open('/models/LogisticRegression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Feature names in the same order as used during training
feature_names = ['Fatigue', 'Slowing', 'Pain', 'Hygiene', 'Movement']

# Sample input data as a numpy array
# Adjust the values based on your input feature format
# For this example, we'll use placeholder values
sample_input = np.array([0.8614045259198462, 0.044451409591983196, 0.44575688224860566, 0.5041639647807894, 0.3982567101330152], dtype=float)

# Reshape the input to match the expected shape for prediction
sample_input = sample_input.reshape(1, -1)

# Predict proneness
predicted_proneness = model.predict(sample_input)

# Print the predicted proneness
print("Predicted Proneness:", predicted_proneness)