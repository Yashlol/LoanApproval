from django.apps import AppConfig
import joblib
import os

class PredictorConfig(AppConfig):
    name = 'predictor'
    model_path = os.path.join(os.path.dirname(__file__), '..', 'loan_approval_model.pkl')
    encoder_path = os.path.join(os.path.dirname(__file__), '..', 'loan_label_encoders.pkl')

    # Load model and encoders at startup
    model = joblib.load(model_path)
    encoders = joblib.load(encoder_path)
