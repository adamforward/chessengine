import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the ChessCNN model based on your architecture


class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=6, out_channels=32, kernel_size=4, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.AvgPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Adjust the input size for the fully connected layer
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
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        # Use sigmoid to ensure output is between 0 and 1
        x = torch.sigmoid(self.fc3(x))
        return x


# Load the model's state_dict from the existing saved file
model = ChessCNN()
model.load_state_dict(torch.load(
    "/Users/adamforward/Desktop/chess/chess_rust/src/model2.pt"))

# Convert the model to TorchScript
scripted_model = torch.jit.script(model)

# Save the scripted model to a new file (or overwrite the existing one)
scripted_model.save(
    "/Users/adamforward/Desktop/chess/chess_rust/src/model2_scripted.pt")

print("Model2 has been converted to TorchScript and saved.")
