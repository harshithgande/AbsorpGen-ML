import requests
import json
from pathlib import Path

# 📁 Define cache path
CACHE_PATH = Path(__file__).resolve().parent / "brand_cache.json"

def load_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

def get_most_common_brand(generic_name):
    generic_name = generic_name.lower()
    cache = load_cache()

    # ✅ Return from cache if exists
    if generic_name in cache:
        return cache[generic_name]

    # 🌐 Query RxNorm API
    url = f"https://rxnav.nlm.nih.gov/REST/drugs.json?name={generic_name}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        concepts = data.get("drugGroup", {}).get("conceptGroup", [])

        brand_names = []
        for group in concepts:
            if group.get("tty") == "BN":  # Brand Name
                for concept in group.get("conceptProperties", []):
                    brand_names.append(concept.get("name"))

        # 🎯 Priority picks
        priority_brands = ["Tylenol", "Advil", "Motrin", "Aleve", "Benadryl", "Claritin"]
        for name in priority_brands:
            if name in brand_names:
                cache[generic_name] = name
                save_cache(cache)
                return name

        # Filter out pediatric names
        filtered = [b for b in brand_names if not any(kid in b.lower() for kid in ["infant", "child", "kids"])]
        if filtered:
            cache[generic_name] = filtered[0]
        elif brand_names:
            cache[generic_name] = brand_names[0]
        else:
            cache[generic_name] = generic_name.capitalize()  # fallback

        save_cache(cache)
        return cache[generic_name]

    except Exception:
        return generic_name.capitalize()


def get_generic_name(brand_name):
    """
    Converts a brand name to its generic name using RxNorm.
    Falls back to input if mapping not found.
    """
    url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={brand_name}"
    try:
        rxcui_res = requests.get(url).json()
        rxcui = rxcui_res.get("idGroup", {}).get("rxnormId", [None])[0]
        if not rxcui:
            return brand_name.upper()

        props_url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/property.json?propName=RxNorm%20Name"
        props_res = requests.get(props_url).json()
        return props_res.get("propConceptGroup", {}).get("propConcept", [{}])[0].get("propValue", brand_name.upper())

    except Exception:
        return brand_name.upper()
