from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import root_mean_squared_error


def train_linear_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    

    return model


def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    

    return model


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    

    return model


def model_evaluation(model, X_test, y_test, model_name:str) -> dict:

    y_pred = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{model.__class__.__name__}: RMSE = {rmse}, MSE = {mse}, MAE = {mae}, R2 = {r2}")

    return {
        'model_name': model_name,
        'mae': mae,
        'rmse': rmse,
        'mse': mse,
        'r2': r2
    }