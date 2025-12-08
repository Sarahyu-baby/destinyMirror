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
        
    def train(self, X_df, y_series):
        """
        Trains the specialized LightGBM model for Love.
        X_df: Full dataframe of features
        y_series: Series of 0/1 labels for Love
        """
        print(f"\n[LoveModel] Initializing LightGBM Training...")
        print(f"[LoveModel] Using specialized features: {self.features}")

        # Filter only relevant features
        X_sub = X_df[self.features].values.astype(np.float32)
        y_sub = y_series.values.astype(int)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)

        # Define Pipeline
        pipe = Pipeline([
            ('scaler', self.scaler),
            ('classifier', LGBMClassifier(random_state=42, verbose=-1))
        ])

        # Optimized Parameter Grid for LightGBM
        param_distributions = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__num_leaves': [15, 31, 50],
            'classifier__max_depth': [3, 5, 10, -1],
            'classifier__min_child_samples': [5, 10, 20]
        }

        # Search
        search = RandomizedSearchCV(
            pipe, param_distributions, n_iter=20, cv=3, n_jobs=-1, verbose=0, random_state=42
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_train, y_train)

        self.model = search.best_estimator_

        # Evaluate
        acc = accuracy_score(y_test, self.model.predict(X_test))
        print(f"[LoveModel] Training Complete. Accuracy: {acc:.2%}")

        return acc
    
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
