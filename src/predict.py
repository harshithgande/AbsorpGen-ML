import torch
import pandas as pd
import os
import joblib
from model import AbsorpGenMultiTaskModel

def predict_new(data_dict):
    df = pd.DataFrame([data_dict])
    for col in ['bioavailability', 'tmax', 'cmax', 'dose', 'formulation_type']:
        df[col] = 0  # dummy targets for structure

    # Load preprocessor and label encoder
    base = os.path.dirname(__file__)
    preprocessor = joblib.load(os.path.join(base, '..', 'models', 'preprocessor_pipeline.pkl'))
    encoder = joblib.load(os.path.join(base, '..', 'models', 'formulation_encoder.pkl'))

    # Preprocess
    X = preprocessor.transform(df.drop(columns=['bioavailability', 'tmax', 'cmax', 'dose', 'formulation_type']))
    X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)

    # Load model
    model = AbsorpGenMultiTaskModel(X_tensor.shape[1])
    model.load_state_dict(torch.load(os.path.join(base, '..', 'models', 'absorpgen_multitask.pt')))
    model.eval()

    with torch.no_grad():
        reg_output, class_logits = model(X_tensor)
        class_probs = torch.softmax(class_logits, dim=1)
        class_pred = class_probs.argmax(dim=1).item()

    regression_outputs = reg_output.numpy().flatten()
    formulation = encoder.inverse_transform([class_pred])[0]

    labels = ['bioavailability', 'tmax', 'cmax', 'dose']
    results = dict(zip(labels, regression_outputs))
    results['recommended_formulation'] = formulation
    return results

# Example usage
if __name__ == "__main__":
    new_input = {
        'molecular_weight': 310.0,
        'logP': 2.8,
        'pKa': 5.0,
        'age': 30,
        'weight': 68,
        'sex': 'male',
        'route_admin': 'oral'
    }

    predicted = predict_new(new_input)
    print("Predicted Values:")
    for key, value in predicted.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")
