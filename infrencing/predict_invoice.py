import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "predict_flag_invoice.pkl"

def load_model(model_path: Path = MODEL_PATH):
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
        return model

def predict_invoice_flag(input_data):
    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['predicted_flag'] = model.predict(input_df).round()

    return input_df

if __name__ == '__main__':
    sample_data = {
        'invoice_quantity': [10, 20, 30],
        'invoice_dollars': [1000, 2000, 3000],
        'Freight': [50, 100, 150],
        'total_item_quantity': [15, 25, 35],
        'total_item_dollars': [1200, 2200, 3200]
    }

    prediction = predict_invoice_flag(sample_data)
    print(prediction)
