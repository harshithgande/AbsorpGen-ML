import pandas as pd
import torch
from termcolor import cprint
from src.predict import predict_new
from src.drug_lookup import lookup_drug_features
from scripts.rxnorm_lookup import get_most_common_brand

# Prioritize practical/OTC drugs if multiple match a symptom
OTC_PRIORITY = ['ACETAMINOPHEN', 'IBUPROFEN', 'NAPROXEN', 'ASPIRIN', 'DIPHENHYDRAMINE', 'LORATADINE']

def search_drug_for_symptom(symptom: str, path: str = "data/processed/drug_indications.csv"):
    df = pd.read_csv(path)
    df.columns = [col.lower() for col in df.columns]  # standardize column names

    symptom = symptom.lower()
    if "condition" not in df.columns or "drug_name" not in df.columns:
        raise KeyError("CSV must contain 'condition' and 'drug_name' columns.")

    matches = df[df["condition"].str.lower().str.contains(symptom, na=False)]

    if matches.empty:
        return None

    # Priority: use OTC-friendly drugs first
    otc_matches = matches[matches["drug_name"].str.upper().isin(OTC_PRIORITY)]
    if not otc_matches.empty:
        return otc_matches["drug_name"].value_counts().idxmax().upper()

    return matches["drug_name"].value_counts().idxmax().upper()


def simulate_absorpgen():
    print("\n🎓 Welcome to AbsorpGen AI – Personalized Drug Assistant\n")

    # 🧑‍⚕️ User Input
    name = input("Enter your name: ").strip()
    age = int(input("Enter your age: ").strip())
    sex = input("Enter your sex (male/female): ").strip().lower()
    weight = float(input("Enter your weight (kg): ").strip())
    height = float(input("Enter your height (cm): ").strip())
    pain = input("What are you experiencing pain from? (e.g., headache, back pain): ").strip()
    pain_level = int(input("On a scale from 1 to 10, how severe is it?: ").strip())
    current_drug = input("💊 Are you currently taking any drug for this? (Leave blank if none): ").strip().upper()

    user_input = {
        "age": age,
        "weight": weight,
        "sex": sex,
        "route_admin": "oral"
    }

    print("\n🧠 Processing your case...\n")

    # 💊 No drug provided — suggest one
    if not current_drug:
        print(f"💡 Suggesting best-fit drug based on {pain} pain level {pain_level}/10...")
        current_drug = search_drug_for_symptom(pain)
        if not current_drug:
            cprint("❌ Sorry, we couldn't find a matching drug for that symptom.", "red")
            return

    try:
        print(f"🔎 Analyzing bioavailability of your current drug: {current_drug}")
        result = predict_new(user_input, current_drug)

        # Format dose
        dose = result["dose"]
        form = result["recommended_formulation"]
        strength = result["strength_mg_per_unit"]
        concentration = result["formulation_concentration"]

        if form == "liquid":
            mL = max(1.0, dose / concentration)
            formatted_dose = f"{round(mL, 1)} mL"
        else:
            tablets = max(1, round(dose / strength))
            formatted_dose = f"{tablets} tablet(s) of {int(strength)} mg"

        # Get display name with brand
        brand_name = get_most_common_brand(result["final_drug_used"])
        display_name = f"{brand_name} ({result['final_drug_used']})" if brand_name else result["final_drug_used"]

        # Output
        cprint("\n✅ Drug Recommendation Complete", "green")
        print(f"Final Drug: {display_name}")
        print(f"Formulation: {form}")
        print(f"Recommended Dose: {formatted_dose}")
        print("Note: This dose was personalized based on your age, weight, and drug properties.\n")

    except Exception as e:
        cprint(f"\n❌ Error: {e}", "red")


if __name__ == "__main__":
    simulate_absorpgen()
