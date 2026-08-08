import joblib
import pandas as pd

class KNNModel:
    def __init__(self, model_path='model/kNN.pkl', scaler_path='model/scaler.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, data):
        scaled_data = self.scaler.transform(data)
        return self.model.predict(scaled_data)