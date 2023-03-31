# production_model.py
import pandas as pd
import joblib

# Load the saved model
voting_clf_loaded = joblib.load('voting_clf_model.pkl')

# Load the test samples
test_samples = pd.read_csv('test_samples.csv')

# Split the test samples into features (X) and target (y)
X_test_samples = test_samples.drop(columns=['cardio'])

# Get predictions and prediction probabilities
y_pred_samples = voting_clf_loaded.predict(X_test_samples)
y_pred_proba_samples = voting_clf_loaded.predict_proba(X_test_samples)

# Create a new DataFrame with prediction results
results_df = pd.DataFrame({
    'probability class 0': y_pred_proba_samples[:, 0],
    'probability class 1': y_pred_proba_samples[:, 1],
    'prediction': y_pred_samples
})

# Export the results to a new CSV file
results_df.to_csv('prediction_results.csv', index=False)