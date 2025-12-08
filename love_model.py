import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import warnings

# Use specific features for Love as optimized in previous steps
LOVE_FEATURES = [
    'upper_lip_ratio',
    'lower_lip_ratio',
    'eye_distance_ratio',
    'eye_symmetry',
    'eye_curvature_ratio'
]

class LoveModel:
    def __init__(self):
        self.model = None
        self.features = LOVE_FEATURES
        self.scaler = StandardScaler()
      
    def predict(self, input_features_df):
        """
        Predicts 'Love' label for a new person.
        Expects a DataFrame with columns matching the full feature set or just the specific subset.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Ensure we have the right columns
        try:
            X_input = input_features_df[self.features].values.astype(np.float32)
        except KeyError as e:
            raise ValueError(f"Input data missing required features for Love Model: {e}")

        prediction = self.model.predict(X_input)[0]
        return int(prediction)
