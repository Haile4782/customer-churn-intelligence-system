import pandas as pd

def load_data(path="data/churn_modelling.csv"):
    df = pd.read_csv(path)

    # Drop unnecessary columns
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    # Convert categorical variables
    df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

    return df