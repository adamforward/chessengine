import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import os

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Enable CUDA launch blocking for better debugging

# Check GPU availability
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Print GPU details if available
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

# Define the model with stacked convolutional layers


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


# Instantiate the model
model = ChessCNN()

# Debug print statement before moving to device
print("Model parameters dtype before moving to device:",
      [param.dtype for param in model.parameters()])

# Move the model to the GPU
model.to(device)
print("Model moved to device successfully")

# Debug print statement after moving to device
print("Model parameters dtype after moving to device:",
      [param.dtype for param in model.parameters()])

# Load and combine batches
output_dir = '/content/drive/My Drive/chess_batches'
all_batches = []
for file_name in os.listdir(output_dir):
    if file_name.endswith('.pt'):
        batch = torch.load(os.path.join(output_dir, file_name))
        all_batches.append(batch)

# Combine all batches into a single dataset
positions = []
labels = []
for batch in all_batches:
    for position, label in batch:
        positions.append(position)
        labels.append(label)

# Ensure there are no NaNs or Infs
positions_np = np.array(positions)
labels_np = np.array(labels)
print(f"Any NaNs in positions: {np.isnan(positions_np).any()}")
print(f"Any Infs in positions: {np.isinf(positions_np).any()}")
print(f"Any NaNs in labels: {np.isnan(labels_np).any()}")
print(f"Any Infs in labels: {np.isinf(labels_np).any()}")

# Add more detailed checking and debugging before moving to tensors


def check_shapes_and_types(positions, labels):
    for i, (pos, lbl) in enumerate(zip(positions, labels)):
        if pos.shape != (6, 8, 8):
            print(f"Shape mismatch at index {i}: {pos.shape}")
        if not isinstance(pos, torch.Tensor):
            print(f"Type mismatch at index {i}: {type(pos)}")
        if lbl not in [0.0, 1.0]:
            print(f"Label value error at index {i}: {lbl}")


check_shapes_and_types(positions, labels)

# Convert data to tensors
positions_tensor = torch.stack(
    [pos.clone().detach() for pos in positions]).float()  # Convert data to float32
labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(
    1)  # Convert labels to float32

# Move tensors to device incrementally
positions_tensor = positions_tensor.to(device)
labels_tensor = labels_tensor.to(device)

# Debug print statements
print("Positions tensor dtype:", positions_tensor.dtype)
print("Labels tensor dtype:", labels_tensor.dtype)

# Create a TensorDataset and DataLoader
dataset = TensorDataset(positions_tensor, labels_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=128,
                          shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=128,
                        shuffle=False, drop_last=False)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if outputs.shape != labels.shape:
                print(
                    f"Shape mismatch: outputs {outputs.shape}, labels {labels.shape}")
                labels = labels[:outputs.shape[0]]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")

        # Save a checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_path = f'/content/drive/My Drive/chess_cnn_checkpoint_epoch_{epoch}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}")

    # Save the final model in float32
    final_model_path = '/content/drive/My Drive/chess_cnn_model_final_f32.pt'
    torch.save(model.state_dict(), final_model_path)
    print("Final model saved successfully")

    # Convert the final model to float16
    model.half()
    final_model_path_f16 = '/content/drive/My Drive/chess_cnn_model_final_f16.pt'
    torch.save(model.state_dict(), final_model_path_f16)
    print("Final model saved as float16 successfully")


train_model(model, train_loader, val_loader,
            criterion, optimizer, num_epochs=20)
