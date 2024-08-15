import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import numpy as np

# Define the model class as before


class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=6, out_channels=32, kernel_size=4, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.AvgPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.AvgPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 1 * 1, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = torch.sigmoid(self.fc3(x))
        return x


# Instantiate the model
model = ChessCNN()

# Load the model state dict, mapping to CPU
model.load_state_dict(torch.load(
    'ai/chess_cnn_model_final_f16.pt', map_location=torch.device('cpu')))
print("Model loaded successfully")

# Export the model to ONNX format
dummy_input = torch.randn(1, 6, 8, 8)  # Adjust the input size accordingly
torch.onnx.export(model, dummy_input, "model.onnx", verbose=True)
print("Model exported to ONNX format successfully")
# for now testing locally, we're gonna use cpu. Will be deployed to the cloud.
