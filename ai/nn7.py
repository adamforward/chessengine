import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import boto3
from torch.utils.data import DataLoader, Dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AWS S3 configuration
s3_bucket_name = 'chessengineaf'
s3 = boto3.client('s3')

# Directory to save the model locally before uploading to S3
local_model_dir = '/tmp/chess_models'
os.makedirs(local_model_dir, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom dataset class


class ChessDataset(Dataset):
    def __init__(self, s3_bucket_name, batch_prefix):
        self.s3_bucket_name = s3_bucket_name
        self.batch_keys = [obj['Key'] for obj in s3.list_objects_v2(
            Bucket=s3_bucket_name, Prefix=batch_prefix)['Contents']]

    def __len__(self):
        return len(self.batch_keys)

    def __getitem__(self, idx):
        batch_key = self.batch_keys[idx]
        local_batch_path = os.path.join('/tmp', os.path.basename(batch_key))

        try:
            # Download the batch file from S3
            s3.download_file(self.s3_bucket_name, batch_key, local_batch_path)
            batch_data = torch.load(local_batch_path)
        except Exception as e:
            logger.error(f"Failed to download or load batch {batch_key}: {e}")
            raise
        finally:
            if os.path.exists(local_batch_path):
                os.remove(local_batch_path)  # Ensure cleanup after download

        # Since no augmentation is applied, return the batch data directly
        return batch_data


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query(x).view(
            batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=6, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        self.resblock1 = ResidualBlock(512, 512)
        self.attention = SelfAttention(512)

        self.fc1 = nn.Linear(512 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.resblock1(x)
        x = self.attention(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=25, save_dir='/tmp/chess_models'):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_loader:
            # Assuming the batch contains tensors in the form (input_tensor, label_tensor, dtm_value_tensor)
            inputs, labels, _ = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        scheduler.step()
        logger.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Save the model after each epoch
        epoch_model_path = os.path.join(
            save_dir, f'chess_cnn_model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), epoch_model_path)
        logger.info(
            f'Model saved after epoch {epoch + 1} to {epoch_model_path}')

    logger.info('Training complete.')


# Example Usage
model = ChessCNN()
criterion = nn.MSELoss()  # Assuming a regression problem
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

# Initialize dataset and dataloader
train_dataset = ChessDataset(
    s3_bucket_name=s3_bucket_name, batch_prefix='chess_batches_f32/')
train_loader = DataLoader(train_dataset, batch_size=64,
                          shuffle=True, num_workers=4)

# Train the model
train_model(model, train_loader, criterion,
            optimizer, scheduler, num_epochs=25)

# Save model to a local path
local_model_path = os.path.join(local_model_dir, 'chess_cnn_model.pth')
torch.save(model.state_dict(), local_model_path)

# Upload the model to S3
s3.upload_file(local_model_path, s3_bucket_name, 'models/chess_cnn_model.pth')
logger.info('Model saved to S3.')
