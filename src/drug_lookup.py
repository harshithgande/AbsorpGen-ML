import pandas as pd
import os

def lookup_drug_features(drug_name, db_path='data/raw/chembl_drug_database.csv'):
    """
    Look up drug features including PK properties and strength information.
    """
    base_dir = os.path.dirname(__file__)
    full_path = os.path.join(base_dir, '..', db_path)
    df = pd.read_csv(full_path)

    match = df[df['drug_name'].str.lower() == drug_name.lower()]
    if match.empty:
        raise ValueError(f"Drug '{drug_name}' not found in the database.")

    row = match.iloc[0]
    return {
        'molecular_weight': float(row['molecular_weight']),
        'logP': float(row['logP']),
        'pKa': float(row['pKa']),
        'bioavailability': float(row['bioavailability']),
        'strength_mg_per_unit': float(row['strength_mg_per_unit']),
        'formulation_concentration': float(row['formulation_concentration'])
    }

def suggest_alternative_drug(min_bioavailability=0.7, db_path='data/raw/chembl_drug_database.csv'):
    """
    Suggest a drug with higher bioavailability if the original is too low.
    """
    base_dir = os.path.dirname(__file__)
    full_path = os.path.join(base_dir, '..', db_path)
    df = pd.read_csv(full_path)

    df_filtered = df[df['bioavailability'] > min_bioavailability]

    if df_filtered.empty:
        raise ValueError("No alternative drug found with bioavailability above threshold.")

    best = df_filtered.sort_values(by='bioavailability', ascending=False).iloc[0]

    return best['drug_name'], {
        'molecular_weight': float(best['molecular_weight']),
        'logP': float(best['logP']),
        'pKa': float(best['pKa']),
        'bioavailability': float(best['bioavailability']),
        'strength_mg_per_unit': float(best['strength_mg_per_unit']),
        'formulation_concentration': float(best['formulation_concentration'])
    }
