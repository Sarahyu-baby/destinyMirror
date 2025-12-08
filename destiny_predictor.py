import pandas as pd
import joblib
import os
import othermodels


class DestinyPredictor:
    """
    Loads a pre-trained model file (.pkl) so raw CSVs are not required at runtime.
    """

    def __init__(self):
        self.models = {}
        self.meaning_map = {}
        self.is_ready = False
        self.model_file = 'destiny_brain.pkl'

        print(f"[DestinyPredictor] Loading AI Brain: {self.model_file}...")

        if not os.path.exists(self.model_file):
            print(f"[Error] Model file {self.model_file} not found! Please run train_and_save.py first.")
            return

        try:
            # Load the brain from disk
            saved_data = joblib.load(self.model_file)

            # Restore the meaning map (dictionary)
            self.meaning_map = saved_data.get('meaning_map', {})

            # Restore the models
            for key, value in saved_data.items():
                if key != 'meaning_map':
                    self.models[key] = value

            self.is_ready = True
            print("[DestinyPredictor] AI Loaded and Ready!")

        except Exception as e:
            print(f"[DestinyPredictor] Failed to load brain: {e}")
            self.is_ready = False

    def predict_fortune(self, feature_dict):
        """
        Input: Dictionary of facial ratios
        Output: Dictionary of fortune results, where each value is a dict:
                {'label': 'Short Label', 'sentence': 'Full fortune sentence'}
        """
        if not self.is_ready:
            return {"Error": {'label': "Error", 'sentence': "AI Models not loaded."}}

        # Convert input dictionary to DataFrame
        try:
            input_data = [feature_dict.get(feat, 0) for feat in othermodels.ALL_FEATURES]
            input_df = pd.DataFrame([input_data], columns=othermodels.ALL_FEATURES)
        except Exception as e:
            return {"Error": {'label': "Error", 'sentence': f"Data processing failed: {e}"}}

        results = {}
        fortune_results = {}

        # 1. Predict Love
        if 'Love' in self.models:
            try:
                # Use the loaded LoveModel
                pred = self.models['Love'].predict(input_df)
                results['Love'] = pred
            except Exception as e:
                print(f"Love prediction error: {e}")
                results['Love'] = 0

        # 2. Predict Specialized XGBoost (Wealth, Health, etc.)
        special_keys = ['Wealth', 'Health', 'Later-life']
        for label in special_keys:
            if label in self.models:
                try:
                    pred = self.models[label].predict(input_df.values)[0]
                    results[label] = int(pred)
                except:
                    results[label] = 0

        # 3. Predict General Model
        if 'GENERAL' in self.models:
            try:
                gen_data = self.models['GENERAL']
                gen_model = gen_data['model']
                gen_targets = gen_data['targets']

                gen_preds = gen_model.predict(input_df.values)[0]
                for i, target in enumerate(gen_targets):
                    results[target] = int(gen_preds[i])
            except:
                pass

        # 4. Convert Numbers to Text and format the output
        display_order = ['Love', 'Wealth', 'Health', 'Career', 'Later-life', 'Authority']

        # Add prioritized items first
        for key in display_order:
            if key in results:
                self._add_result(fortune_results, key, results[key])

        # Add remaining items
        for key, val in results.items():
            self._add_result(fortune_results, key, val)

        return fortune_results

    def _add_result(self, fortune_results, key, val):
        """Helper to format and add a result to the dictionary."""
        if key not in fortune_results:
            text = self._get_text(key, val)
            label = key.replace("_", " ").upper()
            fortune_results[key] = {'label': label, 'sentence': text}

    def _get_text(self, label, val):
        """Helper to look up the meaning in the dictionary"""
        norm_label = label.replace('-', '').lower()
        if norm_label in self.meaning_map and val in self.meaning_map[norm_label]:
            return self.meaning_map[norm_label][val]
        return f"Result: {val}"