import joblib
from pathlib import Path


from data_preprocessing import load_vendor_invoice_data, select_features_target, split_data
from model_evaluation import (
    train_decision_tree,
    train_linear_model,
    train_random_forest,
    model_evaluation
)


def main():
    db_path = 'Project 10-Invoice Intelligent System/data/inventory.db'
    model_dir = Path('Project 10-Invoice Intelligent System/freight_cost_prediction/models')
    model_dir.mkdir(exist_ok=True)

    # load data
    df = load_vendor_invoice_data(db_path)

    # select features
    X, y = select_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # model training
    lr_model = train_linear_model(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # model evaluation
    results = []
    results.append(model_evaluation(lr_model, X_test, y_test, 'Linear Regression'))
    results.append(model_evaluation(dt_model, X_test, y_test, 'Decision Tree'))
    results.append(model_evaluation(rf_model, X_test, y_test, 'Random Forest'))

    # Select best model
    best_model_info = min(results, key=lambda x: x["mae"])
    best_model_name = best_model_info["model_name"]

    best_model = {
        "Linear Regression": lr_model,
        "Decision Tree": dt_model,
        "Random Forest": rf_model
    }[best_model_name]

    # Save model
    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"\nBest model saved: {best_model_name}")
    print(f"Model path: {model_path}")

if __name__ == "__main__":
    main()