import sys
import os
from pathlib import Path
import json
import pandas as pd

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from current directory
from .predict import predict_new
from .safety_checker import SafetyChecker
from .drug_lookup import lookup_drug_features, suggest_alternative_drug
from .rxnorm_lookup import get_most_common_brand

def load_drug_database():
    """
    Load drug database from ChEMBL.
    """
    try:
        # Load the ChEMBL drug database
        drug_db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   "data", "chembl_drug_database.csv")
        drug_db = pd.read_csv(drug_db_path)
        
        # Convert to dictionary format for easier access
        drug_dict = {}
        for _, row in drug_db.iterrows():
            drug_name = row['drug_name'].upper()
            if drug_name not in drug_dict:
                drug_dict[drug_name] = {
                    "info": {
                        "name": drug_name,
                        "brand": row['brand_name'],
                        "strengths": eval(row['available_strengths']),
                        "formulation": row['formulation'],
                        "symptoms": eval(row['indications']),
                        "bioavailability_threshold": float(row['bioavailability_threshold']),
                        "max_daily_dose": float(row['max_daily_dose']),
                        "pain_level_range": eval(row['pain_level_range']) if pd.notna(row['pain_level_range']) else None
                    },
                    "alternatives": []
                }
            
            # Add alternative formulations if available
            if pd.notna(row['alternative_formulations']):
                alternatives = eval(row['alternative_formulations'])
                for alt in alternatives:
                    drug_dict[drug_name]["alternatives"].append({
                        "name": alt['name'],
                        "brand": alt['brand'],
                        "strengths": alt['strengths'],
                        "formulation": alt['formulation']
                    })
        
        return drug_dict
    except Exception as e:
        print(f"Error loading drug database: {str(e)}")
        return {}

def get_user_input():
    """
    Get user input for simulation parameters.
    """
    print("\n=== Patient Information ===")
    age = int(input("Age: "))
    weight = float(input("Weight (kg): "))
    sex = input("Sex (male/female): ").lower()
    height = float(input("Height (cm): "))
    
    print("\n=== Medical Information ===")
    current_symptoms = input("Current symptoms (comma-separated): ").strip()
    current_medications = input("Current medications (comma-separated, press Enter if none): ").strip()
    allergies = input("Allergies (comma-separated, press Enter if none): ").strip()
    medical_conditions = input("Medical conditions (comma-separated, press Enter if none): ").strip()
    
    print("\n=== Pain Information ===")
    pain_type = input("Type of pain (e.g., headache, back pain): ").strip()
    pain_level = int(input("Pain level (1-10): "))
    
    print("\n=== Drug Preference (Optional) ===")
    preferred_drug = input("Preferred drug (press Enter if none): ").strip()
    
    # Convert comma-separated strings to lists
    current_symptoms = [s.strip() for s in current_symptoms.split(',')] if current_symptoms else []
    current_medications = [m.strip() for m in current_medications.split(',')] if current_medications else []
    allergies = [a.strip() for a in allergies.split(',')] if allergies else []
    medical_conditions = [c.strip() for c in medical_conditions.split(',')] if medical_conditions else []
    
    return {
        "age": age,
        "weight": weight,
        "sex": sex,
        "height": height,
        "current_symptoms": current_symptoms,
        "current_medications": current_medications,
        "allergies": allergies,
        "medical_conditions": medical_conditions,
        "pain_type": pain_type,
        "pain_level": pain_level,
        "preferred_drug": preferred_drug,
        "route_admin": "oral"  # Default to oral administration
    }

def find_alternative_drug(current_drug: str, required_dose: float, formulation: str, drug_db: dict) -> tuple[str, float, str]:
    """
    Find an alternative drug that better matches the required dose.
    """
    if current_drug not in drug_db:
        return "", 0, ""
    
    # Find the best matching alternative
    best_match = None
    smallest_diff = float('inf')
    
    for alt in drug_db[current_drug]:
        if alt["formulation"] != formulation:
            continue
            
        for strength in alt["strengths"]:
            # Calculate how many units would be needed
            units_needed = required_dose / strength
            # Round to nearest whole unit
            rounded_units = round(units_needed)
            # Calculate the actual dose this would give
            actual_dose = rounded_units * strength
            # Calculate the difference
            diff = abs(actual_dose - required_dose) / required_dose
            
            if diff < smallest_diff:
                smallest_diff = diff
                best_match = (alt["brand"], strength, alt["formulation"])
    
    return best_match if best_match else ("", 0, "")

def format_dose(dose_mg: float, formulation: str, strength_mg_per_unit: float, concentration_mg_per_ml: float = None) -> str:
    """
    Format the dose into user-friendly units (tablets or mL).
    """
    if formulation.lower() == 'liquid' or (concentration_mg_per_ml is not None and concentration_mg_per_ml > 0):
        if concentration_mg_per_ml is None:
            concentration_mg_per_ml = strength_mg_per_unit
        ml_needed = dose_mg / concentration_mg_per_ml
        # Ensure minimum dose of 0.5 mL and round to nearest 0.5 mL
        ml_needed = max(0.5, round(ml_needed * 2) / 2)
        return f"{ml_needed} mL"
    else:
        # Calculate number of tablets needed
        tablets_needed = dose_mg / strength_mg_per_unit
        
        # If less than 1 tablet, use the smallest available tablet
        if tablets_needed < 1:
            return f"1 tablet of {int(strength_mg_per_unit)} mg"
        
        # For multiple tablets, round to nearest whole tablet
        tablets_needed = round(tablets_needed)
        
        # If we need more than one tablet, try to split into convenient combinations
        if tablets_needed > 1:
            # Try to find a combination of tablet strengths
            for combo in [(1, tablets_needed-1), (2, tablets_needed-2)]:
                if combo[0] > 0 and combo[1] > 0:
                    return f"{combo[0]} tablet(s) of {int(strength_mg_per_unit)} mg and {combo[1]} tablet(s) of {int(strength_mg_per_unit)} mg"
        
        return f"{tablets_needed} tablet(s) of {int(strength_mg_per_unit)} mg"

def select_initial_drug(symptoms: list, pain_level: int, pain_type: str) -> str:
    """
    Select the most appropriate initial drug based on symptoms and pain.
    """
    # Get drug features for common OTC medications
    common_drugs = ["ACETAMINOPHEN", "IBUPROFEN", "DEXTROMETHORPHAN"]
    drug_features = {}
    
    for drug in common_drugs:
        try:
            features = lookup_drug_features(drug)
            drug_features[drug] = features
        except Exception:
            continue
    
    # If no symptoms provided, use pain level to select drug
    if not symptoms:
        if pain_level >= 7:
            return "IBUPROFEN"  # Stronger pain relief
        else:
            return "ACETAMINOPHEN"  # Milder pain relief
    
    # Check symptoms against drug indications
    for drug, features in drug_features.items():
        if "indications" in features:
            for symptom in symptoms:
                if symptom.lower() in [ind.lower() for ind in features["indications"]]:
                    return drug
    
    # Default to pain-based selection if no symptom match
    if pain_level >= 7:
        return "IBUPROFEN"
    else:
        return "ACETAMINOPHEN"

def run_simulation():
    # Initialize safety checker
    safety_checker = SafetyChecker()
    
    # Get user input
    user_input = get_user_input()
    
    try:
        # Select initial drug based on symptoms and pain
        initial_drug = select_initial_drug(
            user_input["current_symptoms"],
            user_input["pain_level"],
            user_input["pain_type"]
        )
        
        # Override with user preference if provided
        if user_input.get("preferred_drug"):
            initial_drug = user_input["preferred_drug"]
        
        # Get prediction
        prediction = predict_new(user_input, initial_drug)
        
        # Check for safety concerns
        warnings = safety_checker.check_safety(
            drug_name=prediction['final_drug_used'],
            current_medications=user_input["current_medications"],
            allergies=user_input["allergies"],
            conditions=user_input["medical_conditions"]
        )
        
        # Format the dose
        formatted_dose = format_dose(
            dose_mg=prediction['dose'],
            formulation=prediction['recommended_formulation'],
            strength_mg_per_unit=prediction['strength_mg_per_unit'],
            concentration_mg_per_ml=prediction.get('formulation_concentration')
        )
        
        # Get brand name
        brand_name = get_most_common_brand(prediction['final_drug_used'])
        if "❌" in brand_name:
            brand_name = prediction['final_drug_used']
        
        # Print results
        print("\n=== Recommendation ===")
        print(f"Brand: {brand_name}")
        print(f"Formulation: {prediction['recommended_formulation']}")
        print(f"Recommended Dose: {formatted_dose}")
        print(f"Bioavailability: {prediction['bioavailability']:.2f}")
        print(f"Tmax: {prediction['tmax']:.2f} hours")
        print(f"Cmax: {prediction['cmax']:.2f} ng/mL")
        
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"- {warning}")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please try again with valid inputs.")

if __name__ == "__main__":
    run_simulation() 