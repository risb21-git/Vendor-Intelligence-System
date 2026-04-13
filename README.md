# Invoice Intelligent System

Vendor invoice intelligence portal that combines:
- Freight cost prediction
- Invoice risk flagging

The project includes model training scripts, inference helpers, notebooks, and a Streamlit UI.

**Key features**
- Predict freight cost from historical vendor invoice data
- Flag invoices that look risky or anomalous
- Streamlit dashboard for quick experimentation

## Project structure
```
Project 10-Invoice Intelligent System/
  app.py
  data/
  freight_cost_prediction/
    data_preprocessing.py
    model_evaluation.py
    train.py
  invoice flagging/
    data_preprocessing.py
    modelling_evaluaton.py
    train.py
  infrencing/
    predict_freight.py
    predict_invoice.py
  models/
    predict_freight_model.pkl
    predict_flag_invoice.pkl
    scaler.pkl
  notebooks/
    Predicting Freight Cost.ipynb
    invoisce_flagging.ipynb
```

## Setup
This project uses global Python (no venv). Install the required libraries:

```powershell
py -m pip install --upgrade pip
py -m pip install pandas numpy scikit-learn joblib streamlit
py -m pip install seaborn matplotlib
```

Optional (for notebooks):
```powershell
py -m pip install ipykernel
```

## Run the app
From the project root:

```powershell
py -m streamlit run app.py
```

The app provides two modes:
- Freight Cost Prediction
- Invoice Flagging

## Train the freight model
Important: `freight_cost_prediction/train.py` uses a relative path that assumes you run the script from the parent folder:
`C:\Users\RISHAV\OneDrive\Desktop\Python`

Run it like this:
```powershell
cd "C:\Users\RISHAV\OneDrive\Desktop\Python"
py "Project 10-Invoice Intelligent System\freight_cost_prediction\train.py"
```

This will read:
`Project 10-Invoice Intelligent System/data/inventory.db`

and write the model here:
`Project 10-Invoice Intelligent System/freight_cost_prediction/models/predict_freight_model.pkl`

## Train the invoice flagging model
The invoice training script uses absolute paths to the model/scaler under:
`C:\Users\RISHAV\OneDrive\Desktop\Python\Project 10-Invoice Intelligent System\models`

Run:
```powershell
cd "C:\Users\RISHAV\OneDrive\Desktop\Python\Project 10-Invoice Intelligent System\invoice flagging"
py train.py
```

## Inference helpers
Freight prediction:
- `infrencing/predict_freight.py` loads:
  `C:\Users\RISHAV\OneDrive\Desktop\Python\Project 10-Invoice Intelligent System\models\predict_freight_model.pkl`

Invoice flagging:
- `infrencing/predict_invoice.py` has an empty `MODEL_PATH` and must be set before use.

## Notes and known issues
- `app.py` currently passes `quantity` and `dollars_per_unit` to the freight predictor, while the model in `freight_cost_prediction/train.py` is trained on `Dollars` only. If predictions look wrong, align the feature names between training and inference.
- Several scripts use absolute paths. If you move the project, update those paths or refactor to build paths relative to the script file.

## Notebooks
- `notebooks/Predicting Freight Cost.ipynb`
- `notebooks/invoisce_flagging.ipynb`

## Data
The SQLite database is expected at:
`data/inventory.db`

If the database schema changes, update the SQL in:
- `freight_cost_prediction/data_preprocessing.py`
- `invoice flagging/data_preprocessing.py`
