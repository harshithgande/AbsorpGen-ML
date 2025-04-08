import torch
import torch.nn as nn

class BioavailabilityMLP(nn.Module):
    def __init__(self, input_dim):
        super(BioavailabilityMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Output: bioavailability
        )

    def forward(self, x):
        return self.model(x)

class AbsorpGenMultiTaskModel(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()

        # Patient characteristics branch
        self.patient_branch = nn.Sequential(
            nn.Linear(4, 32),  # age, weight, sex, height
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # Drug properties branch
        self.drug_branch = nn.Sequential(
            nn.Linear(input_dim - 4, 32),  # molecular_weight, logP, pKa, etc.
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # Combined features
        self.combined = nn.Sequential(
            nn.Linear(32, 64),  # 16 + 16 from both branches
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.regression_head = nn.Linear(32, 4)        # 4 pharmacokinetic outputs
        self.classification_head = nn.Linear(32, num_classes)  # formulation type

    def forward(self, x):
        # Split input into patient and drug features
        patient_features = x[:, :4]  # age, weight, sex, height
        drug_features = x[:, 4:]     # molecular_weight, logP, pKa, etc.

        # Process through respective branches
        patient_out = self.patient_branch(patient_features)
        drug_out = self.drug_branch(drug_features)

        # Combine features
        combined = torch.cat([patient_out, drug_out], dim=1)
        shared_output = self.combined(combined)

        # Get predictions
        reg_output = self.regression_head(shared_output)
        class_logits = self.classification_head(shared_output)
        return reg_output, class_logits
