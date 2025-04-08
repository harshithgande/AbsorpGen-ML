def get_most_common_brand(drug_name: str) -> str:
    """
    Get the most common brand name for a given drug.
    Returns the drug name itself if no brand name is found.
    """
    # Common brand names mapping
    brand_mapping = {
        "ACETAMINOPHEN": "Tylenol",
        "IBUPROFEN": "Advil",
        "DEXTROMETHORPHAN": "Robitussin",
        "PRAZOSIN": "Minipress",
        "DOXAZOSIN": "Cardura",
        "TERAZOSIN": "Hytrin",
        "CIPROFLOXACIN": "Cipro",
        "WARFARIN": "Coumadin",
        "NYQUIL": "NyQuil",
        "NYQUILL": "NyQuil"  # Common misspelling
    }
    
    # Reverse mapping for brand to drug name
    drug_mapping = {
        "Tylenol": "ACETAMINOPHEN",
        "Advil": "IBUPROFEN",
        "Robitussin": "DEXTROMETHORPHAN",
        "Minipress": "PRAZOSIN",
        "Cardura": "DOXAZOSIN",
        "Hytrin": "TERAZOSIN",
        "Cipro": "CIPROFLOXACIN",
        "Coumadin": "WARFARIN",
        "NyQuil": "DEXTROMETHORPHAN"  # NyQuil contains dextromethorphan
    }
    
    # If input is a brand name, return the corresponding drug name
    if drug_name in drug_mapping:
        return drug_mapping[drug_name]
    
    # If input is a drug name, return the brand name
    return brand_mapping.get(drug_name.upper(), drug_name) 