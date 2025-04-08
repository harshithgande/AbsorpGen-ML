from typing import List, Dict, Set
import requests
from pathlib import Path
import json

class SafetyChecker:
    def __init__(self):
        self.interactions_db = self._load_interactions_db()
        self.contraindications_db = self._load_contraindications_db()
        self.allergy_db = self._load_allergy_db()

    def _load_interactions_db(self) -> Dict[str, List[str]]:
        """Load drug interactions database from file."""
        base = Path(__file__).resolve().parent.parent
        db_path = base / 'data' / 'drug_interactions.json'
        if db_path.exists():
            with open(db_path, 'r') as f:
                return json.load(f)
        return {}

    def _load_contraindications_db(self) -> Dict[str, List[str]]:
        """Load contraindications database from file."""
        base = Path(__file__).resolve().parent.parent
        db_path = base / 'data' / 'contraindications.json'
        if db_path.exists():
            with open(db_path, 'r') as f:
                return json.load(f)
        return {}

    def _load_allergy_db(self) -> Dict[str, List[str]]:
        """Load allergy cross-reactivity database."""
        base = Path(__file__).resolve().parent.parent
        db_path = base / 'data' / 'allergy_cross_reactivity.json'
        if db_path.exists():
            with open(db_path, 'r') as f:
                return json.load(f)
        return {}

    def check_safety(self, 
                    drug_name: str, 
                    current_medications: List[str] = None,
                    allergies: List[str] = None,
                    conditions: List[str] = None) -> List[str]:
        """
        Check for potential safety issues with the drug.
        Returns a list of warnings if any safety issues are found.
        """
        warnings = []
        
        # Check drug interactions
        if current_medications:
            interactions = self.check_interactions(drug_name, current_medications)
            warnings.extend(interactions)
        
        # Check contraindications
        if conditions:
            contraindications = self.check_contraindications(drug_name, conditions)
            warnings.extend(contraindications)
        
        # Check allergies
        if allergies:
            allergy_warnings = self.check_allergies(drug_name, allergies)
            warnings.extend(allergy_warnings)
        
        return warnings

    def check_interactions(self, drug_name: str, current_medications: List[str]) -> List[str]:
        """Check for drug-drug interactions."""
        warnings = []
        for med in current_medications:
            if med in self.interactions_db.get(drug_name, []):
                warnings.append(f"Warning: {drug_name} may interact with {med}")
        return warnings

    def check_contraindications(self, drug_name: str, conditions: List[str]) -> List[str]:
        """Check for contraindications based on medical conditions."""
        warnings = []
        for condition in conditions:
            if condition in self.contraindications_db.get(drug_name, []):
                warnings.append(f"Warning: {drug_name} is contraindicated in {condition}")
        return warnings

    def check_allergies(self, drug_name: str, allergies: List[str]) -> List[str]:
        """Check for potential allergic reactions."""
        warnings = []
        for allergy in allergies:
            if allergy in self.allergy_db.get(drug_name, []):
                warnings.append(f"Warning: {drug_name} may cause cross-reactivity with {allergy} allergy")
        return warnings

    def update_databases(self):
        """Update safety databases from external sources."""
        # TODO: Implement OpenFDA API integration for real-time updates
        pass 