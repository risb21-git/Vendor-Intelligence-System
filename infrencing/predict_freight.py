import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "predict_freight_model.pkl"

def load_model(model_path: Path = MODEL_PATH):
    """
    Load trained model
    """
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    return model

def predict_freight_cost(input_data):
    """
    predict freight cost for new vendor invoices.

    parameters
    input_df : dict
    returns
    pd.DataFrame with predicted freight cost
    """

    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['predicted_freight'] = model.predict(input_df).round()
    return input_df

if __name__ == '__main__':

    sample_data = {
        'Dollars': [10500,9000,3000]
    }

    prediction = predict_freight_cost(sample_data)
    print(prediction)
