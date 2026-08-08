import joblib
import pandas as pd

class NaiveBayesModel:
    def __init__(self, model_path='model/Naive_Bayes.pkl', scaler_path='model/scaler.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, data):
        scaled_data = self.scaler.transform(data)
        return self.model.predict(scaled_data)