import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define the ChessCNN model class


class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=6, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        # Use sigmoid to ensure output is between 0 and 1
        x = torch.sigmoid(self.fc3(x))
        return x


# Instantiate the model
model = ChessCNN()

# Load the model weights from the file
model_path = "/Users/adamforward/Desktop/chess/chess_rust/src/model4.pt"
model.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
model.eval()

# Convert the model to TorchScript
scripted_model = torch.jit.script(model)

# Re-save the scripted model
torchscript_model_path = "/Users/adamforward/Desktop/chess/chess_rust/src/model4_scripted.pt"
scripted_model.save(torchscript_model_path)

print(
    f"Model has been re-saved in TorchScript format at {torchscript_model_path}")

# Random evaluation
# Generate a random input tensor with values constrained to -1.0, 0.0, and 1.0
random_input = torch.tensor(np.random.choice(
    [0.0, 1.0, -1.0], size=(1, 6, 8, 8)), dtype=torch.float32)

# Run the model with the random input
output = scripted_model(random_input)
print(f"Random input evaluation output: {output.item()}")
