import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Global encoder to be used in train and predict
formulation_encoder = LabelEncoder()

def load_mock_data():
    data = {
        'molecular_weight': [300.5, 450.2, 180.0],
        'logP': [2.1, 4.5, 1.3],
        'pKa': [4.8, 7.2, 5.5],
        'age': [25, 60, 45],
        'weight': [70, 85, 55],
        'sex': ['male', 'female', 'female'],
        'route_admin': ['oral', 'oral', 'iv'],
        'bioavailability': [0.75, 0.60, 0.95],
        'tmax': [2.5, 3.0, 1.8],
        'cmax': [20, 15, 30],
        'dose': [250, 300, 200],
        'formulation_type': ['tablet', 'liquid', 'delayed release']
    }
    return pd.DataFrame(data)

def preprocess_data(df):
    regression_targets = df[['bioavailability', 'tmax', 'cmax', 'dose']]
    class_target = formulation_encoder.fit_transform(df['formulation_type'])

    features = df.drop(columns=['bioavailability', 'tmax', 'cmax', 'dose', 'formulation_type'])

    numeric_features = ['molecular_weight', 'logP', 'pKa', 'age', 'weight']
    categorical_features = ['sex', 'route_admin']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

    X_processed = preprocessor.fit_transform(features)
    return X_processed, regression_targets, class_target, preprocessor
