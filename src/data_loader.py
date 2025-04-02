import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_mock_data():
    """Creates a small mock dataset with drug and user features."""
    data = {
        'molecular_weight': [300.5, 450.2, 180.0],
        'logP': [2.1, 4.5, 1.3],
        'pKa': [4.8, 7.2, 5.5],
        'age': [25, 60, 45],
        'weight': [70, 85, 55],
        'sex': ['male', 'female', 'female'],
        'route_admin': ['oral', 'oral', 'iv'],
        'bioavailability': [0.75, 0.60, 0.95]  # Target variable
    }
    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    """Preprocesses the dataset: scales numbers and encodes categories."""
    features = df.drop(columns='bioavailability')
    target = df['bioavailability']

    numeric_features = ['molecular_weight', 'logP', 'pKa', 'age', 'weight']
    categorical_features = ['sex', 'route_admin']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

    X_processed = preprocessor.fit_transform(features)
    return X_processed, target
