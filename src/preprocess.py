import pandas as pd

def load_data(path="data/churn_modelling.csv"):
    df = pd.read_csv(path)

    # Drop unnecessary columns
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    # One-hot encoding
    df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

    # Feature Engineering
    df["Balance_per_Product"] = df["Balance"] / (df["NumOfProducts"] + 1)
    df["Age_Group"] = df["Age"] // 10

    return df