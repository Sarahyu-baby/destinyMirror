import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import warnings
import sys

# Import the separated Love Model
# Note: Ensure love_model.py is in the same directory
try:
    from love_model import LoveModel
except ImportError:
    print("Error: 'love_model.py' not found. Please ensure both files are in the same folder.")
    sys.exit(1)

# FORCE IGNORE WARNINGS
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. CONFIGURATION ---
DATA_FILE = 'merged_celebrity_data.csv'
DESC_FILE = 'destiny_labels.csv'

# Full feature list used by General/Wealth/Health models
ALL_FEATURES = [
    'face_lw_ratio', 'forehead_ratio', 'midface_ratio', 'lowerface_ratio',
    'eye_distance_ratio', 'nose_ratio', 'mouth_chin_ratio', 'jaw_angle',
    'upper_lip_ratio', 'lower_lip_ratio', 'eye_aspect_ratio',
    'eye_curvature_ratio', 'eye_symmetry'
]

# Targets managed by THIS main script (using XGBoost)
XGB_TARGETS_SPECIAL = ['Wealth', 'Health', 'Later-life']


# 'Love' is managed by external LoveModel

# --- 2. HELPERS ---

def load_label_descriptions(desc_filepath):
    try:
        desc_df = pd.read_csv(desc_filepath)
        meaning_map = {}
        for _, row in desc_df.iterrows():
            raw_label = str(row['label']).strip()
            norm_label = raw_label.replace('-', '').lower()
            val = int(row['value'])
            desc = row['description']
            if norm_label not in meaning_map:
                meaning_map[norm_label] = {}
            meaning_map[norm_label][val] = desc
        return meaning_map
    except Exception as e:
        print(f"Warning: Could not load label descriptions: {e}")
        return {}


def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # Validation
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Fill NA
    X = df[ALL_FEATURES].fillna(df[ALL_FEATURES].mean())

    # Get all potential targets present in CSV
    target_cols = [
        'Career', 'Love', 'Love2', 'Wealth', 'Health',
        'Children', 'Social', 'Authority',
        'Authority2', 'Later-life', 'Social2'
    ]
    available_targets = [c for c in target_cols if c in df.columns]
    y = df[available_targets].fillna(0)

    print(f"Successfully loaded {len(df)} records.")
    return X, y


# --- 3. TRAINING XGBOOST MODELS ---

def train_xgboost_specialized(X, y, label):
    """Trains a specific XGBoost model for Wealth, Health, etc. Returns model and accuracy."""
    print(f"\n[XGBoost] Training Specialized Model for '{label}'...")

    y_target = y[label].values.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X.values, y_target, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
    ])

    params = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7],
        'classifier__subsample': [0.8, 1.0],
        'classifier__colsample_bytree': [0.8, 1.0]
    }

    search = RandomizedSearchCV(pipe, params, n_iter=20, cv=3, n_jobs=-1, verbose=0, random_state=42)
    search.fit(X_train, y_train)

    acc = accuracy_score(y_test, search.best_estimator_.predict(X_test))
    print(f"   > Accuracy: {acc:.2%}")
    return search.best_estimator_, acc


def train_general_model(X, y, exclude_labels):
    """Trains a Multi-Output XGBoost model for all remaining labels. Returns model, targets, and accuracies."""
    print(f"\n[XGBoost] Training General Model for remaining labels...")

    targets = [col for col in y.columns if col not in exclude_labels]

    if not targets:
        return None, [], []

    y_gen = y[targets].values.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X.values, y_gen, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', MultiOutputClassifier(XGBClassifier(random_state=42, eval_metric='logloss')))
    ])

    params = {
        'classifier__estimator__n_estimators': [100, 200],
        'classifier__estimator__learning_rate': [0.01, 0.05],
        'classifier__estimator__max_depth': [3, 5]
    }

    search = RandomizedSearchCV(pipe, params, n_iter=10, cv=3, n_jobs=-1, verbose=0, random_state=42)
    search.fit(X_train, y_train)

    # Eval
    accuracies = []
    y_pred = search.best_estimator_.predict(X_test)
    for i, col in enumerate(targets):
        acc = accuracy_score(y_test[:, i], y_pred[:, i])
        accuracies.append(acc)
        print(f"   > {col:12}: {acc:.2%}")

    return search.best_estimator_, targets, accuracies


# --- 4. MAIN ORCHESTRATION ---

def main():
    try:
        # 1. Load Resources
        meaning_map = load_label_descriptions(DESC_FILE)
        X, y = load_data(DATA_FILE)

        models = {}
        all_accuracies = []  # List to store accuracy of every single label

        # 2. Train External Love Model (LightGBM)
        if 'Love' in y.columns:
            love_model = LoveModel()  # From separate file
            love_acc = love_model.train(X, y['Love'])
            all_accuracies.append(love_acc)
            models['Love'] = love_model

        # 3. Train Specialized XGBoost Models (Wealth, Health, Later-life)
        for label in XGB_TARGETS_SPECIAL:
            if label in y.columns:
                model, acc = train_xgboost_specialized(X, y, label)
                all_accuracies.append(acc)
                models[label] = model

        # 4. Train General XGBoost Model
        exclude = XGB_TARGETS_SPECIAL + ['Love']
        gen_model, gen_targets, gen_accs = train_general_model(X, y, exclude)

        if gen_model:
            all_accuracies.extend(gen_accs)
            models['GENERAL'] = {'model': gen_model, 'targets': gen_targets}

        # 5. Show Total System Accuracy
        if all_accuracies:
            total_acc = np.mean(all_accuracies)
            print("\n" + "=" * 40)
            print(f"TOTAL SYSTEM ACCURACY: {total_acc:.2%}")
            print("=" * 40)

        # 6. Simulate Prediction
        print("\n" + "-" * 40)
        print("SIMULATING PREDICTION FOR NEW PERSON")
        print("-" * 40)

        # Example Data
        new_person = [
            0.89, 0.31, 0.31, 0.41, 0.24, 1.29, 1.10,
            138.8, 0.12, 0.17, 0.25, 0.05, 0.91
        ]
        input_df = pd.DataFrame([new_person], columns=ALL_FEATURES)

        results = {}

        # Predict: Love (LightGBM)
        if 'Love' in models:
            pred = models['Love'].predict(input_df)
            results['Love'] = pred

        # Predict: Special XGBoost
        for label in XGB_TARGETS_SPECIAL:
            if label in models:
                pred = models[label].predict(input_df.values)[0]
                results[label] = int(pred)

        # Predict: General XGBoost
        if 'GENERAL' in models:
            gen_preds = models['GENERAL']['model'].predict(input_df.values)[0]
            for i, target in enumerate(models['GENERAL']['targets']):
                results[target] = int(gen_preds[i])

        # Display Results
        priority = ['Love', 'Wealth', 'Health', 'Later-life']

        for p in priority:
            if p in results:
                desc = get_description(p, results[p], meaning_map)
                print(f"[{p}]: {desc}")

        for k, v in results.items():
            if k not in priority:
                desc = get_description(k, v, meaning_map)
                print(f"[{k}]: {desc}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


def get_description(label, val, meaning_map):
    norm_label = label.replace('-', '').lower()
    if norm_label in meaning_map and val in meaning_map[norm_label]:
        return meaning_map[norm_label][val]
    return f"Result: {val}"


if __name__ == "__main__":
    main()