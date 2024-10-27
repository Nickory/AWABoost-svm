import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
import pandas as pd
import joblib


# Define AWABoostSVM class
class AWABoostSVM:
    def __init__(self, n_estimators=50, theta=1, C=5):
        self.n_estimators = n_estimators
        self.theta = theta
        self.C = C
        self.alphas = []
        self.models = []

    # Prediction function
    def predict(self, X):
        """
        Predict labels for given input data X using the trained AWABoost-SVM model.

        Parameters:
        X (np.array): Input features for prediction.

        Returns:
        np.array: Predicted labels.
        """
        pred = np.zeros(X.shape[0])
        for model, alpha in zip(self.models, self.alphas):
            pred += alpha * model.predict(X)
        return np.sign(pred)


# Load the trained model
model_file = 'awabost_svm_model.pkl'
try:
    awaboost_svm = joblib.load(model_file)
    print(f"Loaded model from {model_file}.")
except FileNotFoundError:
    raise FileNotFoundError(f"Model file {model_file} not found. Make sure the model file is in the current directory.")

# Load test data (Cleveland dataset)
data_file_path = 'heart_cleveland_upload.csv'
try:
    data_df = pd.read_csv(data_file_path, header=None, names=[f'feature_{i}' for i in range(13)] + ['label'])
    data_df['label'] = data_df['label'].map({1: 1, 0: -1})  # Map labels to -1 and 1
    print(f"Loaded dataset from {data_file_path}.")
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset file {data_file_path} not found. Please add the file to the current directory.")

# Prepare features (X) and labels (y)
X = data_df.drop('label', axis=1).values
y = data_df['label'].values

# Predict using the trained model
predictions = awaboost_svm.predict(X)

# Evaluate the model performance using various metrics
accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions)
recall = recall_score(y, predictions)
f1 = f1_score(y, predictions)
auc = roc_auc_score(y, predictions)

# Output the results
print("Model Evaluation Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
