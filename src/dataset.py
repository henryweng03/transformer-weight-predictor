import torch
import numpy as np
from pathlib import Path
import os
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader


class WeightSequenceDataset(Dataset):
    """Dataset for training a model to predict weight evolutions.
    
    Creates sequences of k consecutive weight snapshots as inputs
    and the (k+1)th snapshot as the target.
    """
    
    def __init__(self, snapshot_dir, sequence_length=3, train_split=0.5, is_train=True, 
                 apply_pca=True, n_components=500):
        """
        Args:
            snapshot_dir: Directory containing the weight snapshots
            sequence_length: Number of consecutive snapshots to use as input
            train_split: Fraction of snapshots to use for training
            is_train: Whether this dataset is for training or testing
            apply_pca: Whether to apply PCA dimensionality reduction
            n_components: Number of PCA components to keep
        """
        self.snapshot_dir = Path(snapshot_dir)
        self.sequence_length = sequence_length
        self.apply_pca = apply_pca
        self.n_components = n_components
        
        # Find all weight snapshot files
        self.snapshot_files = sorted([f for f in self.snapshot_dir.glob("epoch_*.pt")], 
                                     key=lambda x: int(x.stem.split('_')[1]))
        
        # Determine total number of epochs
        self.total_epochs = len(self.snapshot_files)
        self.split_idx = int(self.total_epochs * train_split)
        
        # Select snapshots based on train/test split
        if is_train:
            self.snapshot_files = self.snapshot_files[:self.split_idx]
        else:
            self.snapshot_files = self.snapshot_files[self.split_idx:]
        
        # Load all weight vectors
        self.weight_vectors = self._load_weight_vectors()
        
        # Apply PCA if requested
        if self.apply_pca:
            self._apply_pca()
        
        # Create sequences
        self.sequences = []
        for i in range(len(self.weight_vectors) - sequence_length):
            input_seq = self.weight_vectors[i:i+sequence_length]
            target = self.weight_vectors[i+sequence_length]
            self.sequences.append((input_seq, target))
    
    def _load_weight_vectors(self):
        """Load and flatten weight snapshots from files."""
        weight_vectors = []
        
        for file in self.snapshot_files:
            # Load state dict
            state_dict = torch.load(file, map_location=torch.device('cpu'))
            
            # Extract target parameters (for simplicity, just take everything)
            weight_vector = []
            for param_name, param in state_dict.items():
                weight_vector.append(param.flatten())
            
            # Concatenate into a single vector
            weight_vector = torch.cat(weight_vector).numpy()
            weight_vectors.append(weight_vector)
        
        return weight_vectors
    
    def _apply_pca(self):
        """Apply PCA dimensionality reduction to weight vectors."""
        # Stack weight vectors into a matrix
        X = np.vstack(self.weight_vectors)
        
        # Fit PCA
        self.pca = PCA(n_components=min(self.n_components, X.shape[1]))
        X_reduced = self.pca.fit_transform(X)
        
        # Replace weight vectors with reduced versions
        self.weight_vectors = [X_reduced[i] for i in range(X_reduced.shape[0])]
        
        print(f"Applied PCA: {X.shape[1]} -> {X_reduced.shape[1]} dimensions")
        print(f"Explained variance ratio: {sum(self.pca.explained_variance_ratio_):.4f}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target = self.sequences[idx]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def prepare_dataloaders(snapshot_dir, sequence_length=3, train_split=0.5, batch_size=32, 
                        apply_pca=True, n_components=500):
    """Create train and test dataloaders for the weight prediction task."""
    
    # Create datasets
    train_dataset = WeightSequenceDataset(
        snapshot_dir=snapshot_dir,
        sequence_length=sequence_length,
        train_split=train_split,
        is_train=True,
        apply_pca=apply_pca,
        n_components=n_components
    )
    
    test_dataset = WeightSequenceDataset(
        snapshot_dir=snapshot_dir,
        sequence_length=sequence_length,
        train_split=train_split,
        is_train=False,
        apply_pca=apply_pca,
        n_components=n_components
    )
    
    # Copy PCA from train to test to ensure consistent transformation
    if apply_pca:
        test_dataset.pca = train_dataset.pca
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Example usage
    train_loader, test_loader = prepare_dataloaders(
        snapshot_dir="./data/snapshots",
        sequence_length=3,
        train_split=0.5,
        batch_size=32
    )
    
    # Print dataset info
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get a batch
    inputs, targets = next(iter(train_loader))
    print(f"Input shape: {inputs.shape}")  # [batch_size, sequence_length, n_features]
    print(f"Target shape: {targets.shape}")  # [batch_size, n_features]