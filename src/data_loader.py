import pandas as pd
from pathlib import Path

def load_data():
    base = Path(__file__).resolve().parent.parent
    df = pd.read_csv(base / 'data' / 'raw' / 'chembl_drug_database.csv')

    # Add synthetic user inputs
    df['age'] = 35
    df['weight'] = 70
    df['sex'] = 'male'
    df['route_admin'] = 'oral'

    # Add synthetic training targets
    df['tmax'] = df['molecular_weight'] * 0.02
    df['cmax'] = df['logP'] * 20
    df['dose'] = df['molecular_weight'] * 0.1
    df['formulation_type'] = 'tablet'

    # Drop rows with missing values in critical columns
    features = ['molecular_weight', 'logP', 'pKa', 'age', 'weight', 'sex', 'route_admin',
                'strength_mg_per_unit', 'formulation_concentration']
    targets = ['bioavailability', 'tmax', 'cmax', 'dose']
    target_class = 'formulation_type'

    df = df.dropna(subset=features + targets + [target_class])

    X = df[features]
    y_reg = df[targets]
    y_class = df[target_class]

    return X, y_reg, y_class
