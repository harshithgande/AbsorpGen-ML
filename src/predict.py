import torch
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .model import AbsorpGenMultiTaskModel
from .drug_lookup import lookup_drug_features, suggest_alternative_drug
from .rxnorm_lookup import get_most_common_brand

def predict_new(user_input, drug_name):
    # Load drug features
    drug_features = lookup_drug_features(drug_name)

    # Prepare patient features
    patient_features = {
        'age': user_input['age'] / 100,  # Scale to 0-1 range
        'weight': user_input['weight'] / 200,  # Scale to 0-1 range
        'sex': 1 if user_input['sex'].lower() == 'male' else 0,  # Binary encoding
        'height': user_input['height'] / 200  # Scale to 0-1 range
    }

    # Prepare drug features
    drug_features_scaled = {
        'molecular_weight': drug_features['molecular_weight'] / 1000,  # Scale to kg/mol
        'logP': drug_features['logP'],
        'pKa': drug_features['pKa'],
        'route_admin': 1 if user_input['route_admin'].lower() == 'oral' else 0,  # Binary encoding
        'strength_mg_per_unit': drug_features['strength_mg_per_unit'] / 1000,  # Scale to g/unit
        'formulation_concentration': drug_features['formulation_concentration'] / 1000 if drug_features['formulation_concentration'] else 0  # Scale to g/mL
    }

    # Combine all features
    all_features = {**patient_features, **drug_features_scaled}

    df = pd.DataFrame([all_features])
    for col in ['bioavailability', 'tmax', 'cmax', 'dose', 'formulation_type']:
        df[col] = 0

    # Load preprocessor and encoder
    base = Path(__file__).resolve().parent.parent
    preprocessor = joblib.load(base / 'models' / 'preprocessor_pipeline.pkl')
    encoder = joblib.load(base / 'models' / 'formulation_encoder.pkl')

    # Align with training features
    X_input = df.drop(columns=['bioavailability', 'tmax', 'cmax', 'dose', 'formulation_type'])
    X_input = X_input[preprocessor.feature_names_in_]
    X = preprocessor.transform(X_input)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Load model
    model = AbsorpGenMultiTaskModel(X_tensor.shape[1])
    model.load_state_dict(torch.load(base / 'models' / 'absorpgen_multitask.pt'))
    model.eval()

    # Predict
    with torch.no_grad():
        reg_output, class_logits = model(X_tensor)
        class_probs = torch.softmax(class_logits, dim=1)
        class_pred = class_probs.argmax(dim=1).item()

    predicted_bioavailability = reg_output.numpy().flatten()[0]

    # 🔁 Fallback if bioavailability too low
    if predicted_bioavailability < 0.7:
        alt_name, alt_features = suggest_alternative_drug(min_bioavailability=0.7)
        print(f"\n⚠️  Bioavailability too low for '{drug_name}'. Switching to alternative: '{alt_name}'")
        all_features.update(alt_features)
        drug_name = alt_name
        df = pd.DataFrame([all_features])
        for col in ['bioavailability', 'tmax', 'cmax', 'dose', 'formulation_type']:
            df[col] = 0
        X_input = df.drop(columns=['bioavailability', 'tmax', 'cmax', 'dose', 'formulation_type'])
        X_input = X_input[preprocessor.feature_names_in_]
        X = preprocessor.transform(X_input)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            reg_output, class_logits = model(X_tensor)
            class_probs = torch.softmax(class_logits, dim=1)
            class_pred = class_probs.argmax(dim=1).item()

    # Format output
    regression_outputs = reg_output.numpy().flatten()
    # Use formulation from drug lookup instead of model prediction
    formulation = drug_features['formulation']
    labels = ['bioavailability', 'tmax', 'cmax', 'dose']
    results = dict(zip(labels, regression_outputs))
    results.update({
        'recommended_formulation': formulation,
        'final_drug_used': drug_name,
        'strength_mg_per_unit': drug_features['strength_mg_per_unit'],
        'formulation_concentration': drug_features['formulation_concentration']
    })

    return results

# 🧪 Manual test
if __name__ == "__main__":
    user_input = {
        'age': 30,
        'weight': 68,
        'sex': 'male',
        'route_admin': 'oral'
    }

    drug_name = "CIPROFLOXACIN"
    advanced_mode = False

    predicted = predict_new(user_input, drug_name)

    # Dose formatting
    dose = predicted['dose']
    form = predicted['recommended_formulation']
    strength = predicted['strength_mg_per_unit']
    concentration = predicted['formulation_concentration']

    if form == 'liquid':
        mL = max(1.0, dose / concentration)
        formatted_dose = f"{round(mL, 1)} mL"
    else:
        tablets = max(1, round(dose / strength))
        formatted_dose = f"{tablets} tablet(s) of {int(strength)} mg"

    # ✅ Get most common brand name
    brand_name = get_most_common_brand(predicted['final_drug_used'])
    if "❌" in brand_name:
        final_display = predicted['final_drug_used']
    else:
        final_display = f"{brand_name} ({predicted['final_drug_used']})"

    # ✅ Output
    print("\n✅ Drug Recommendation Complete")
    print(f"Final Drug: {final_display}")
    print(f"Formulation: {form}")
    print(f"Recommended Dose: {formatted_dose}")
    print("Note: This dose was personalized based on your age, weight, and drug properties.")

    if advanced_mode:
        print("\n--- Advanced Prediction Details ---")
        print(f"Predicted Bioavailability: {round(predicted['bioavailability'], 3)}")
        print(f"Predicted Tmax: {round(predicted['tmax'], 2)} hrs")
        print(f"Predicted Cmax: {round(predicted['cmax'], 2)} ng/mL")
