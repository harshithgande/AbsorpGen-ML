import pandas as pd
import os
from .rxnorm_lookup import get_most_common_brand

def load_drug_indications():
    """
    Load drug indications from the CSV file.
    """
    base_dir = os.path.dirname(__file__)
    full_path = os.path.join(base_dir, '..', 'data', 'processed', 'drug_indications.csv')
    df = pd.read_csv(full_path)
    return df

# Common OTC drugs with their properties
OTC_DRUGS = {
    'ACETAMINOPHEN': {
        'molecular_weight': 151.16,
        'logP': 0.5,
        'pKa': 9.5,
        'bioavailability': 0.88,
        'strength_mg_per_unit': 500,
        'formulation_concentration': 160,
        'formulation': 'tablet'
    },
    'IBUPROFEN': {
        'molecular_weight': 206.28,
        'logP': 3.5,
        'pKa': 4.4,
        'bioavailability': 0.95,
        'strength_mg_per_unit': 200,
        'formulation_concentration': 100,
        'formulation': 'tablet'
    },
    'DEXTROMETHORPHAN': {
        'molecular_weight': 271.4,
        'logP': 3.2,
        'pKa': 9.2,
        'bioavailability': 0.75,
        'strength_mg_per_unit': 15,
        'formulation_concentration': 7.5,
        'formulation': 'liquid'
    },
    'NYQUIL': {
        'molecular_weight': 271.4,  # Using dextromethorphan's properties
        'logP': 3.2,
        'pKa': 9.2,
        'bioavailability': 0.75,
        'strength_mg_per_unit': 15,
        'formulation_concentration': 7.5,
        'formulation': 'liquid',
        'is_liquid': True  # Explicit flag for liquid formulation
    },
    'NYQUILL': {  # Common misspelling
        'molecular_weight': 271.4,
        'logP': 3.2,
        'pKa': 9.2,
        'bioavailability': 0.75,
        'strength_mg_per_unit': 15,
        'formulation_concentration': 7.5,
        'formulation': 'liquid',
        'is_liquid': True  # Explicit flag for liquid formulation
    }
}

def lookup_drug_features(drug_name, db_path='data/raw/chembl_drug_database.csv'):
    """
    Look up drug features including PK properties and strength information.
    """
    # First check if input is a brand name and convert to drug name
    drug_name = get_most_common_brand(drug_name)
    
    # First check OTC drugs
    if drug_name.upper() in OTC_DRUGS:
        features = OTC_DRUGS[drug_name.upper()].copy()
        # Add indications from drug_indications.csv
        try:
            indications_df = load_drug_indications()
            drug_indications = indications_df[indications_df['drug_name'].str.upper() == drug_name.upper()]
            if not drug_indications.empty:
                features['indications'] = drug_indications['indications'].iloc[0].split(',')
        except Exception:
            pass
        return features
    
    # Then check database
    try:
        base_dir = os.path.dirname(__file__)
        full_path = os.path.join(base_dir, '..', db_path)
        df = pd.read_csv(full_path)

        match = df[df['drug_name'].str.lower() == drug_name.lower()]
        if match.empty:
            raise ValueError(f"Drug '{drug_name}' not found in the database.")
        
        row = match.iloc[0]
        features = {
            'molecular_weight': float(row['molecular_weight']),
            'logP': float(row['logP']),
            'pKa': float(row['pKa']),
            'bioavailability': float(row['bioavailability']),
            'strength_mg_per_unit': float(row['strength_mg_per_unit']),
            'formulation_concentration': float(row['formulation_concentration'])
        }
        
        # Add indications from drug_indications.csv
        try:
            indications_df = load_drug_indications()
            drug_indications = indications_df[indications_df['drug_name'].str.upper() == drug_name.upper()]
            if not drug_indications.empty:
                features['indications'] = drug_indications['indications'].iloc[0].split(',')
        except Exception:
            pass
            
        return features
    except Exception as e:
        # If database lookup fails, try OTC drugs again
        if drug_name.upper() in OTC_DRUGS:
            return OTC_DRUGS[drug_name.upper()]
        raise ValueError(f"Drug '{drug_name}' not found in the database and is not a known OTC drug.")

def suggest_alternative_drug(min_bioavailability=0.7, db_path='data/raw/chembl_drug_database.csv'):
    """
    Suggest a drug with higher bioavailability if the original is too low.
    """
    # First check OTC drugs
    for drug, features in OTC_DRUGS.items():
        if features['bioavailability'] > min_bioavailability:
            return drug, features
    
    # Then check database
    try:
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
    except Exception:
        # If database lookup fails, try OTC drugs again
        for drug, features in OTC_DRUGS.items():
            if features['bioavailability'] > min_bioavailability:
                return drug, features
        raise ValueError("No alternative drug found with bioavailability above threshold.")
