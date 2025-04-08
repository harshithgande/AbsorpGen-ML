from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import predict_new
from safety_checker import SafetyChecker

app = FastAPI(
    title="AbsorpGen AI API",
    description="API for personalized drug recommendations and pharmacokinetic predictions",
    version="1.0.0"
)

# Initialize safety checker
safety_checker = SafetyChecker()

class UserInput(BaseModel):
    age: int
    weight: float
    sex: str
    height: Optional[float] = None
    route_admin: str
    current_symptoms: Optional[List[str]] = None
    current_medications: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    medical_conditions: Optional[List[str]] = None

class DrugRecommendation(BaseModel):
    drug_name: str
    recommended_formulation: str
    recommended_dose: str
    bioavailability: float
    tmax: float
    cmax: float
    warnings: List[str]
    brand_name: Optional[str] = None

@app.post("/predict", response_model=DrugRecommendation)
async def predict_drug_recommendation(user_input: UserInput, drug_name: str):
    """
    Get personalized drug recommendations based on user characteristics and drug properties.
    """
    try:
        # Convert Pydantic model to dict for our existing predict function
        user_input_dict = user_input.dict()
        
        # Call our existing prediction function
        prediction = predict_new(user_input_dict, drug_name)
        
        # Check for safety issues
        warnings = safety_checker.check_safety(
            drug_name=drug_name,
            current_medications=user_input.current_medications,
            allergies=user_input.allergies,
            conditions=user_input.medical_conditions
        )
        
        # Format the response
        return DrugRecommendation(
            drug_name=prediction['final_drug_used'],
            recommended_formulation=prediction['recommended_formulation'],
            recommended_dose=f"{prediction['dose']} mg",
            bioavailability=prediction['bioavailability'],
            tmax=prediction['tmax'],
            cmax=prediction['cmax'],
            warnings=warnings,
            brand_name=prediction.get('brand_name')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "healthy"} 