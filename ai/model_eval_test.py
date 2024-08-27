import torch
import os
import torch.nn as nn
import torch.nn.functional as F
# Define the paths to your models and batch
model_paths = [
    # "/Users/adamforward/Desktop/chess/chess_rust/src/chess_cnn_checkpoint_epoch_15.pt",
    # "/Users/adamforward/Desktop/chess/chess_rust/src/chess_cnn_model_final_3x3x3x3_f16.pt",
    # "/Users/adamforward/Desktop/chess/chess_rust/src/chess_cnn_model_final_3xpool2xconv_f16.pt",
    # "/Users/adamforward/Desktop/chess/chess_rust/src/chess_cnn_model_final_f16_3_convs.pt",
    "/Users/adamforward/Desktop/chess/chess_rust/src/chess_cnn_model_final_f16.pt"
]
batch_path = "/Users/adamforward/Desktop/chess/ai/batch_0.pt"


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
# class ChessCNN(nn.Module):
#     def __init__(self):
#         super(ChessCNN, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_channels=6, out_channels=32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
#         self.pool1 = nn.AvgPool2d(2, 2)

#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool2 = nn.MaxPool2d(2, 2)

#         self.fc1 = nn.Linear(64 * 2 * 2, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 1)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.pool1(x)
#         x = F.relu(self.conv3(x))
#         x = self.pool2(x)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, 0.5)
#         x = F.relu(self.fc2(x))
#         x = F.dropout(x, 0.5)
#         # Use sigmoid to ensure output is between 0 and 1
#         x = torch.sigmoid(self.fc3(x))
#         return x


# Load the batch
batch = torch.load(batch_path, map_location=torch.device('cpu'))
positions, labels = zip(*batch)

# Convert to tensors and ensure they are in float16
positions_tensor = torch.stack(positions).float().to(torch.float16)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to load and evaluate a model


def evaluate_model(model_path, positions_tensor):
    # Initialize the ChessCNN model architecture
    model = ChessCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.half()  # Convert model to float16
    model.eval()

    # Ensure the model is on the correct device
    model.to(device)

    # Get the neural network output
    with torch.no_grad():
        outputs = model(positions_tensor.to(device))

    print(f"Model: {os.path.basename(model_path)} - Output: {outputs}")


# Evaluate all models
for model_path in model_paths:
    evaluate_model(model_path, positions_tensor)
