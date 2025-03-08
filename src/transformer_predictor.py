import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import math
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from base_model import LeNet5
from dataset import prepare_dataloaders


class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer model."""
    
    def __init__(self, d_model, max_len=20):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Handle both even and odd dimensions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Make sure div_term is the right length for the odd/even case
        if d_model % 2 == 0:
            # Even case
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # Odd case
            div_term = div_term[:d_model//2 + 1]  # Add one extra element for odd case
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2][:, :d_model//2] = torch.cos(position * div_term[:-1])
        
        # Register buffer (persistent state)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Output tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]


class TransformerPredictor(nn.Module):
    """Transformer-based weight predictor model."""
    
    def __init__(self, input_dim, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        
        # Model hyperparameters
        self.input_dim = input_dim
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(input_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(input_dim, input_dim)
    
    def forward(self, src):
        """
        Args:
            src: Input sequence tensor [batch_size, seq_len, input_dim]
        
        Returns:
            Predicted next weight vector [batch_size, input_dim]
        """
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Pass through transformer encoder
        output = self.transformer_encoder(src)
        
        # Take the last sequence element and project
        last_hidden = output[:, -1]
        prediction = self.output_projection(last_hidden)
        
        return prediction


def train_transformer_predictor(snapshot_dir, sequence_length=3, train_split=0.5, batch_size=32,
                                apply_pca=True, n_components=500, learning_rate=1e-4,
                                num_epochs=100, device=None, model_save_path="./models/transformer_predictor.pt"):
    """Train the transformer-based weight predictor."""
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, test_loader = prepare_dataloaders(
        snapshot_dir=snapshot_dir,
        sequence_length=sequence_length,
        train_split=train_split,
        batch_size=batch_size,
        apply_pca=apply_pca,
        n_components=n_components
    )
    
    # Get input dimension from the first batch
    inputs, _ = next(iter(train_loader))
    input_dim = inputs.shape[2]  # [batch_size, seq_len, features]
    
    # Create model
    # Adjust number of heads to be compatible with input dimension
    if input_dim % 8 == 0:
        nhead = 8
    elif input_dim % 5 == 0:
        nhead = 5
    else:
        # Find the largest divisor <= 8
        for h in range(8, 0, -1):
            if input_dim % h == 0:
                nhead = h
                break
        else:
            # If no clean divisor found, adjust input_dim with padding
            original_input_dim = input_dim
            for adj in range(1, 9):
                if (input_dim + adj) % 8 == 0:
                    input_dim = input_dim + adj
                    break
            print(f"Adjusted input dimension from {original_input_dim} to {input_dim} to be compatible with 8 attention heads")
            nhead = 8
    
    print(f"Using {nhead} attention heads for input dimension {input_dim}")
    
    model = TransformerPredictor(
        input_dim=input_dim,
        nhead=nhead,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1
    ).to(device)
    
    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            
            # Create directory if it doesn't exist
            Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'sequence_length': sequence_length,
                'n_components': n_components,
                'input_dim': input_dim
            }, model_save_path)
            print(f"Model saved to {model_save_path}")
        
        # Print progress
        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{num_epochs} | Time: {elapsed:.2f}s | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.savefig('./models/loss_curve.png')
    plt.close()
    
    return model, train_losses, test_losses


def evaluate_predictor(model, snapshot_dir, sequence_length=3, train_split=0.5, 
                       apply_pca=True, n_components=500, device=None):
    """Evaluate the trained predictor on held-out epochs."""
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataloaders
    _, test_loader = prepare_dataloaders(
        snapshot_dir=snapshot_dir,
        sequence_length=sequence_length,
        train_split=train_split,
        batch_size=1,  # Process one sequence at a time for visualization
        apply_pca=apply_pca,
        n_components=n_components
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Collect actual and predicted weights
    actual_weights = []
    predicted_weights = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            
            actual_weights.append(targets.cpu().numpy())
            predicted_weights.append(predictions.cpu().numpy())
    
    actual_weights = np.vstack(actual_weights)
    predicted_weights = np.vstack(predicted_weights)
    
    # Calculate MSE
    mse = np.mean((actual_weights - predicted_weights) ** 2)
    print(f"Test MSE: {mse:.6f}")
    
    # PCA for visualization (2D projection)
    combined = np.vstack([actual_weights, predicted_weights])
    pca_vis = PCA(n_components=2)
    combined_2d = pca_vis.fit_transform(combined)
    
    actual_2d = combined_2d[:len(actual_weights)]
    predicted_2d = combined_2d[len(actual_weights):]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(actual_2d[:, 0], actual_2d[:, 1], c='blue', label='Actual', alpha=0.7)
    plt.scatter(predicted_2d[:, 0], predicted_2d[:, 1], c='red', label='Predicted', alpha=0.7)
    
    # Draw lines connecting corresponding points
    for i in range(len(actual_2d)):
        plt.plot([actual_2d[i, 0], predicted_2d[i, 0]], [actual_2d[i, 1], predicted_2d[i, 1]], 'k-', alpha=0.2)
    
    plt.title('PCA Visualization of Weight Space')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.savefig('./models/weight_space_visualization.png')
    plt.close()
    
    return mse, actual_weights, predicted_weights


def extrapolation_test(model, snapshot_dir, sequence_length=3, train_split=0.5, num_steps=5, 
                       apply_pca=True, n_components=500, device=None):
    """Test multi-step extrapolation using the predictor."""
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get dataset for the starting sequence
    _, test_loader = prepare_dataloaders(
        snapshot_dir=snapshot_dir,
        sequence_length=sequence_length,
        train_split=train_split,
        batch_size=1,
        apply_pca=apply_pca,
        n_components=n_components
    )
    
    # Get the first sequence from the test set
    for inputs, _ in test_loader:
        initial_sequence = inputs.to(device)
        break
    
    # Get PCA object from the dataset
    test_dataset = test_loader.dataset
    pca = test_dataset.pca if hasattr(test_dataset, 'pca') else None
    
    # Perform multi-step prediction
    current_sequence = initial_sequence.clone()
    predicted_vectors = []
    
    model.eval()
    with torch.no_grad():
        for _ in range(num_steps):
            # Predict next weight vector
            next_vector = model(current_sequence)
            predicted_vectors.append(next_vector.cpu().numpy()[0])
            
            # Update the sequence (remove first, add predicted)
            current_sequence = torch.cat([current_sequence[:, 1:], next_vector.unsqueeze(1)], dim=1)
    
    # Load actual weight vectors for comparison
    # This assumes we're predicting beyond the train_split point
    snapshot_files = sorted(Path(snapshot_dir).glob("epoch_*.pt"), 
                           key=lambda x: int(x.stem.split('_')[1]))
    
    split_idx = int(len(snapshot_files) * train_split)
    test_files = snapshot_files[split_idx:]
    
    # Load the true weight vectors for comparison
    true_vectors = []
    
    # Skip the ones that were in the initial sequence
    start_idx = sequence_length
    end_idx = start_idx + num_steps
    
    for i in range(start_idx, min(end_idx, len(test_files))):
        state_dict = torch.load(test_files[i], map_location=torch.device('cpu'))
        
        # Extract parameters
        weight_vector = []
        for param_name, param in state_dict.items():
            weight_vector.append(param.flatten())
        
        # Concatenate into a single vector
        weight_vector = torch.cat(weight_vector).numpy()
        
        # Apply PCA if needed
        if pca is not None:
            weight_vector = pca.transform(weight_vector.reshape(1, -1))[0]
        
        true_vectors.append(weight_vector)
    
    # Calculate MSE for each step
    mse_per_step = []
    
    for i in range(min(num_steps, len(true_vectors))):
        mse = np.mean((true_vectors[i] - predicted_vectors[i]) ** 2)
        mse_per_step.append(mse)
        print(f"Step {i+1} MSE: {mse:.6f}")
    
    # Plot MSE per step
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(mse_per_step) + 1), mse_per_step, marker='o')
    plt.xlabel('Extrapolation Step')
    plt.ylabel('MSE')
    plt.title('Error Accumulation in Multi-Step Prediction')
    plt.grid(True)
    plt.savefig('./models/extrapolation_error.png')
    plt.close()
    
    return mse_per_step, predicted_vectors, true_vectors


def plug_in_test(predicted_weights, snapshot_dir, train_split=0.5, device=None):
    """Test the predicted weights by plugging them into the base model.
    
    This test takes the predicted weight vectors, inverse-transforms them (if PCA was used),
    and plugs them into a copy of the base model to evaluate performance on MNIST.
    """
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get dataset to load the PCA transformer
    _, test_loader = prepare_dataloaders(
        snapshot_dir=snapshot_dir,
        sequence_length=3,  # This doesn't matter here
        train_split=train_split,
        batch_size=1,
        apply_pca=True  # We assume PCA was used
    )
    
    # Get PCA object from the dataset
    test_dataset = test_loader.dataset
    pca = test_dataset.pca if hasattr(test_dataset, 'pca') else None
    
    # Check if we're working with multiple runs
    is_multiple_runs = (Path(snapshot_dir) / "run_1").exists()
    
    if is_multiple_runs:
        print("Using multiple runs directory structure")
        # Use the first run's first snapshot
        run_dir = Path(snapshot_dir) / "run_1"
        snapshot_files = sorted([f for f in run_dir.glob("epoch_*.pt") if "batch" not in f.name], 
                              key=lambda x: int(x.stem.split('_')[1]))
    else:
        # Standard single run handling
        snapshot_files = sorted([f for f in Path(snapshot_dir).glob("epoch_*.pt") if "batch" not in f.name], 
                              key=lambda x: int(x.stem.split('_')[1]))
    
    if not snapshot_files:
        raise ValueError(f"No snapshot files found in {snapshot_dir}")
        
    # Get a template state dict to know parameter names and shapes
    template_state_dict = torch.load(snapshot_files[0], map_location=torch.device('cpu'))
    
    # Initialize the base model
    model = LeNet5().to(device)
    
    # For each predicted weight vector
    accuracy_results = []
    
    for i, weight_vector in enumerate(predicted_weights):
        # Inverse transform if PCA was used
        if pca is not None:
            weight_vector = pca.inverse_transform(weight_vector.reshape(1, -1))[0]
        
        # Reconstruct the state dict
        new_state_dict = {}
        start_idx = 0
        
        for param_name, param in template_state_dict.items():
            param_size = param.numel()
            param_shape = param.shape
            
            # Extract the relevant portion of the weight vector
            param_flat = weight_vector[start_idx:start_idx + param_size]
            
            # Reshape to original parameter shape
            param_reshaped = torch.tensor(param_flat.reshape(param_shape), dtype=param.dtype)
            
            # Add to state dict
            new_state_dict[param_name] = param_reshaped
            
            # Update index
            start_idx += param_size
        
        # Load state dict into model
        model.load_state_dict(new_state_dict)
        
        # Evaluate on MNIST
        accuracy = evaluate_mnist(model, device)
        accuracy_results.append(accuracy)
        
        print(f"Step {i+1} Predicted Weights Accuracy: {accuracy:.2f}%")
    
    return accuracy_results


def evaluate_mnist(model, device=None):
    """Evaluate a model on the MNIST test set."""
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Import required modules
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    # Set up test data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Evaluate
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    # Train the transformer predictor
    model, train_losses, test_losses = train_transformer_predictor(
        snapshot_dir="./data/snapshots",
        sequence_length=3,
        train_split=0.5,
        batch_size=32,
        apply_pca=True,
        n_components=500,
        learning_rate=1e-4,
        num_epochs=100
    )
    
    # Evaluate on held-out epochs
    mse, actual_weights, predicted_weights = evaluate_predictor(
        model=model,
        snapshot_dir="./data/snapshots",
        sequence_length=3,
        train_split=0.5
    )
    
    # Test multi-step extrapolation
    mse_per_step, predicted_vectors, true_vectors = extrapolation_test(
        model=model,
        snapshot_dir="./data/snapshots",
        sequence_length=3,
        train_split=0.5,
        num_steps=5
    )
    
    # Plugin test
    accuracy_results = plug_in_test(
        predicted_weights=predicted_vectors,
        snapshot_dir="./data/snapshots",
        train_split=0.5
    )