from data_loader import load_mock_data, preprocess_data, formulation_encoder
from model import AbsorpGenMultiTaskModel
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
import joblib
import os

# Load data
df = load_mock_data()
X, y_reg, y_class, preprocessor = preprocess_data(df)

# Save preprocessor and label encoder
joblib.dump(preprocessor, 'models/preprocessor_pipeline.pkl')
joblib.dump(formulation_encoder, 'models/formulation_encoder.pkl')

# Convert to tensors
X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)
y_reg_tensor = torch.tensor(y_reg.values, dtype=torch.float32)
y_class_tensor = torch.tensor(y_class, dtype=torch.long)

# Model
input_dim = X_tensor.shape[1]
model = AbsorpGenMultiTaskModel(input_dim)
reg_criterion = nn.MSELoss()
class_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train
epochs = 200
for epoch in range(epochs):
    model.train()
    reg_output, class_logits = model(X_tensor)

    reg_loss = reg_criterion(reg_output, y_reg_tensor)
    class_loss = class_criterion(class_logits, y_class_tensor)
    total_loss = reg_loss + class_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}, Total Loss: {total_loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}, Class Loss: {class_loss.item():.4f}")

# Save model
torch.save(model.state_dict(), 'models/absorpgen_multitask.pt')
print("Model, preprocessor, and label encoder saved.")
