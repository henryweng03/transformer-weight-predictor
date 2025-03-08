import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
from pathlib import Path


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def evaluate_model(model, test_loader, criterion, device):
    """Evaluate the model on the test set and return accuracy and loss."""
    model.eval()
    correct = 0
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy


def train_base_model(num_epochs=20, snapshot_dir="./data/snapshots/training_runs", saves_per_epoch=1):
    """Train a LeNet-5 model on MNIST and save weight snapshots during training.
    
    Args:
        num_epochs: Total number of epochs to train the model
        snapshot_dir: Directory to save weight snapshots
        saves_per_epoch: Number of times to save snapshots per epoch 
                         (1 = only at end of epoch, >1 = save at regular intervals within epoch)
    """
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    # Use a smaller batch size (32 instead of 64) and different learning rate to 
    # ensure more interesting weight dynamics
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Initialize the model
    model = LeNet5().to(device)
    
    # Set up loss and optimizer with slightly modified hyperparameters
    # Use a higher learning rate (0.005) to make the learning more dynamic
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Create snapshot directory if it doesn't exist
    os.makedirs(snapshot_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        epoch_loss = 0.0
        total_batches = len(train_loader)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Print training progress
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}/{num_epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
            
            # Save intermediate snapshots within epoch for more granular tracking
            if saves_per_epoch > 1:
                # Calculate how many batches to skip between saves
                save_interval = total_batches // saves_per_epoch
                
                # Save at regular intervals throughout the epoch
                # Skip the last batch as it will be saved at the end of epoch
                if (
                    save_interval > 0 and 
                    batch_idx > 0 and 
                    batch_idx % save_interval == 0 and 
                    batch_idx < total_batches - 1
                ):
                    # Evaluate model to get test accuracy
                    test_loss, accuracy = evaluate_model(model, test_loader, criterion, device)
                    
                    # Include both epoch and batch percentage in filename
                    progress_percent = (batch_idx / total_batches) * 100
                    snapshot_path = Path(snapshot_dir) / f"epoch_{epoch}_batch_{progress_percent:.0f}pct_acc_{accuracy:.2f}.pt"
                    torch.save(model.state_dict(), snapshot_path)
                    print(f"Saved intermediate snapshot to {snapshot_path} ({batch_idx}/{total_batches} batches)")
                    print(f"Intermediate Test Loss: {test_loss:.6f} Test Accuracy: {accuracy:.2f}%")
                    
                    # Set model back to training mode
                    model.train()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch: {epoch}/{num_epochs} Average Training Loss: {avg_loss:.6f}')
        
        # Evaluation at the end of each epoch
        test_loss, accuracy = evaluate_model(model, test_loader, criterion, device)
        print(f'Epoch: {epoch}/{num_epochs} Test Loss: {test_loss:.6f} Test Accuracy: {accuracy:.2f}%')
        
        # Always save model weights snapshot at the end of each epoch
        snapshot_path = Path(snapshot_dir) / f"epoch_{epoch}_done_acc_{accuracy:.2f}.pt"
        torch.save(model.state_dict(), snapshot_path)
        print(f"Saved model snapshot to {snapshot_path}")
    
    print("Training completed.")
    return model


if __name__ == "__main__":
    train_base_model()