import torch
import pandas as pd
import joblib
from pathlib import Path
from src.data_loader import load_data
from src.model import AbsorpGenMultiTaskModel
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

def train():
    # Load data
    X, y_reg, y_class = load_data()

    # Load preprocessor and encoder
    base = Path(__file__).resolve().parent.parent
    preprocessor = joblib.load(base / 'models' / 'preprocessor_pipeline.pkl')
    encoder = joblib.load(base / 'models' / 'formulation_encoder.pkl')

    # Align feature columns to preprocessor
    X = X[preprocessor.feature_names_in_]
    X_processed = preprocessor.transform(X)
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)
    y_reg_tensor = torch.tensor(y_reg.values, dtype=torch.float32)
    y_class_tensor = torch.tensor(encoder.transform(y_class), dtype=torch.long)

    # Split into train/test
    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
        X_tensor, y_reg_tensor, y_class_tensor, test_size=0.2, random_state=42
    )

    # Initialize model
    model = AbsorpGenMultiTaskModel(X_tensor.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    reg_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        reg_output, class_logits = model(X_train)
        loss_reg = reg_criterion(reg_output, y_reg_train)
        loss_class = class_criterion(class_logits, y_class_train)
        loss = loss_reg + loss_class
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), base / 'models' / 'absorpgen_multitask.pt')
    print("✅ Model trained and saved.")

if __name__ == "__main__":
    train()
