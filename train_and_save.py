import joblib
import os
import pandas as pd
from love_model import LoveModel
import othermodels


def save_all_models():
    print("Starting training process...")

    # 1. Load Data
    try:
        meaning_map = othermodels.load_label_descriptions(othermodels.DESC_FILE)
        X, y = othermodels.load_data(othermodels.DATA_FILE)
    except Exception as e:
        print(f"Error: Could not find CSV files. Make sure they are in this folder.\n{e}")
        return

    saved_data = {}

    # 2. Train Love Model
    if 'Love' in y.columns:
        print("Training Love Model...")
        love = LoveModel()
        love.train(X, y['Love'])
        saved_data['Love'] = love

    # 3. Train Specialized XGBoost Models (Wealth, Health, etc.)
    for label in othermodels.XGB_TARGETS_SPECIAL:
        if label in y.columns:
            print(f"Training {label} Model...")
            model, _ = othermodels.train_xgboost_specialized(X, y, label)
            saved_data[label] = model




if __name__ == "__main__":
    save_all_models()