import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split


def load_vendor_invoice_data(db_path: str):

    conn = sqlite3.connect(db_path)
    query = 'select * from  vendor_invoice'
    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def select_features_target(df: pd.DataFrame):
    # select features and target variable
    X = df[['Dollars']]
    y = df[['Freight']]

    return X, y

def split_data(X, y, test_size=0.2, random_state=42):

    return train_test_split(X, y, test_size=test_size, random_state=random_state)