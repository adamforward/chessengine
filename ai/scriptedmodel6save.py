import torch
import torch.nn as nn
import torch.nn.functional as F
# Define your model class (ChessCNN)


class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=6, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool1(x)
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = torch.dropout(x, 0.5, self.training)
        x = torch.relu(self.fc2(x))
        x = torch.dropout(x, 0.5, self.training)
        x = torch.sigmoid(self.fc3(x))
        return x


# Instantiate the model
model = ChessCNN()

# Set the model to evaluation mode
model.eval()

# Script the model
scripted_model = torch.jit.script(model)

# Save the scripted model using TorchScript
scripted_model.save(
    "/Users/adamforward/Desktop/chess/chess_rust/src/model6_scripted.pt")

print("Model saved successfully using TorchScript.")
