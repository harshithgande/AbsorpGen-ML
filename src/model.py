import torch
import torch.nn as nn

class BioavailabilityMLP(nn.Module):
    def __init__(self, input_dim):
        super(BioavailabilityMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Output: bioavailability
        )

    def forward(self, x):
        return self.model(x)
class AbsorpGenMultiTaskModel(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.regression_head = nn.Linear(32, 4)        # 4 pharmacokinetic outputs
        self.classification_head = nn.Linear(32, num_classes)  # formulation type

    def forward(self, x):
        shared_output = self.shared(x)
        reg_output = self.regression_head(shared_output)
        class_logits = self.classification_head(shared_output)
        return reg_output, class_logits
