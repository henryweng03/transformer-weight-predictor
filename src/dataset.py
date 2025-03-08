import torch
import numpy as np
from pathlib import Path
import os
import re
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
            train_split: Fraction of snapshots in each run to use for training 
                         (e.g., 0.7 means first 70% of snapshots in each run for training)
            is_train: Whether this dataset is for training or testing
            apply_pca: Whether to apply PCA dimensionality reduction
            n_components: Number of PCA components to keep
        """
        self.snapshot_dir = Path(snapshot_dir)
        self.sequence_length = sequence_length
        self.apply_pca = apply_pca
        self.n_components = n_components
        
        # Check if we are working with the training runs directory structure
        is_training_runs = False
        if (self.snapshot_dir / "run_1").exists():
            is_training_runs = True
            print(f"Found training runs in {snapshot_dir}")
        
        if is_training_runs:
            # Collect all run directories
            run_dirs = sorted([d for d in self.snapshot_dir.glob("run_*") if d.is_dir()], 
                             key=lambda x: int(x.name.split('_')[1]))
            
            # Collect snapshots from all run directories and split within each run
            self.snapshot_files = []
            
            for run_dir in run_dirs:
                # New file pattern: epoch_1_batch_10pct_acc_86.88.pt or epoch_1_done_acc_97.43.pt
                # Find all .pt files in the directory
                pt_files = list(run_dir.glob("*.pt"))
                
                # Process files to extract epoch and batch information
                file_info = []
                for f in pt_files:
                    # Extract information using regex
                    if "done" in f.name:
                        # End of epoch file
                        match = re.search(r'epoch_(\d+)_done_acc_([\d\.]+)\.pt', f.name)
                        if match:
                            epoch = int(match.group(1))
                            accuracy = float(match.group(2))
                            # We give it a high percentage to ensure it sorts last in the epoch
                            file_info.append((epoch + 1.0, accuracy, f))
                    else:
                        # Batch file
                        match = re.search(r'epoch_(\d+)_batch_(\d+)pct_acc_([\d\.]+)\.pt', f.name)
                        if match:
                            epoch = int(match.group(1))
                            pct = int(match.group(2))
                            accuracy = float(match.group(3))
                            # Use decimal position to order correctly (e.g., epoch 1.05, 1.10, 1.15)
                            file_info.append((epoch + pct/100, accuracy, f))
                
                # Sort files by epoch/batch position
                file_info.sort()
                
                # Calculate the split index for this run
                split_idx = int(len(file_info) * train_split)
                
                # Select files based on train/test split
                if is_train:
                    selected_files = file_info[:split_idx]
                else:
                    selected_files = file_info[split_idx:]
                
                run_files = [f for _, _, f in selected_files]
                
                # Add run information to each file
                run_id = int(run_dir.name.split('_')[1])
                self.snapshot_files.extend([(run_id, f) for f in run_files])
                
                mode = "training" if is_train else "testing"
                print(f"Run {run_id}: Added {len(run_files)} snapshots to {mode} set")
        else:
            # Error, cancel script
            raise ValueError("Training runs directory not found. Directory should be in format: ./data/snapshots/training_runs/run_1/epoch_1_batch_10pct_acc_86.88.pt")
        
        # Determine total number of snapshots
        self.total_snapshots = len(self.snapshot_files)
        print(f"Found {self.total_snapshots} snapshots in {'training' if is_train else 'testing'} set")
        
        # Load all weight vectors
        self.weight_vectors, self.run_info = self._load_weight_vectors()
        
        # Apply PCA if requested
        if self.apply_pca:
            self._apply_pca(force_n_components=self.n_components)
        
        # Create sequences - only create sequences within the same run
        self.sequences = []
        current_run = None
        seq_buffer = []
        
        for i, (run_id, vector) in enumerate(zip(self.run_info, self.weight_vectors)):
            # If we're starting a new run or this is the first element
            if current_run != run_id:
                # Clear the buffer when switching runs
                seq_buffer = [vector]
                current_run = run_id
            else:
                # Add to the buffer
                seq_buffer.append(vector)
                
                # If we have enough elements in the buffer, create a sequence
                if len(seq_buffer) > sequence_length:
                    input_seq = seq_buffer[-sequence_length-1:-1]
                    target = seq_buffer[-1]
                    self.sequences.append((input_seq, target))
                    
        print(f"Created {len(self.sequences)} sequences from {self.total_snapshots} snapshots")
    
    def _load_weight_vectors(self):
        """Load and flatten weight snapshots from files."""
        weight_vectors = []
        run_info = []
        
        for run_id, file in self.snapshot_files:
            # Load state dict
            state_dict = torch.load(file, map_location=torch.device('cpu'))
            
            # Extract target parameters (for simplicity, just take everything)
            weight_vector = []
            for param_name, param in state_dict.items():
                weight_vector.append(param.flatten())
            
            # Concatenate into a single vector
            weight_vector = torch.cat(weight_vector).numpy()
            weight_vectors.append(weight_vector)
            run_info.append(run_id)
        
        return weight_vectors, run_info
    
    def _apply_pca(self, force_n_components=None):
        """Apply PCA dimensionality reduction to weight vectors.
        
        Args:
            force_n_components: If provided, use this exact number of components regardless of data size
        """
        # Stack weight vectors into a matrix
        X = np.vstack(self.weight_vectors)
        
        # Determine number of components
        if force_n_components is not None:
            n_components = min(force_n_components, X.shape[0], X.shape[1])
        else:
            # Use min(self.n_components, X.shape[0], X.shape[1]) to avoid errors
            n_components = min(self.n_components, X.shape[0], X.shape[1])
            
        self.pca = PCA(n_components=n_components)
        X_reduced = self.pca.fit_transform(X)
        
        # Replace weight vectors with reduced versions
        self.weight_vectors = [X_reduced[i] for i in range(X_reduced.shape[0])]
        
        print(f"Applied PCA: {X.shape[1]} -> {X_reduced.shape[1]} dimensions")
        print(f"Explained variance ratio: {sum(self.pca.explained_variance_ratio_):.4f}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target = self.sequences[idx]
        # Convert lists to numpy arrays first for better performance
        input_np = np.array(input_seq)
        target_np = np.array(target)
        return torch.tensor(input_np, dtype=torch.float32), torch.tensor(target_np, dtype=torch.float32)


def prepare_dataloaders(snapshot_dir, sequence_length=3, train_split=0.2, batch_size=32, 
                        apply_pca=True, n_components=500):
    """Create train and test dataloaders for the weight prediction task.
    
    Args:
        snapshot_dir: Directory containing the weight snapshots
        sequence_length: Number of consecutive snapshots to use as input
        train_split: Fraction of snapshots in each run to use for training
                    (e.g., 0.7 means first 70% of snapshots in each run for training)
        batch_size: Batch size for the dataloaders
        apply_pca: Whether to apply PCA dimensionality reduction
        n_components: Number of PCA components to keep
    """
    
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
    
    # For PCA consistency, we need to load the raw vectors again for test set
    # and apply the exact same transformation
    if apply_pca:
        # Copy PCA from train to test for consistent transform
        test_dataset.pca = train_dataset.pca
        
        # Reload weight vectors without PCA for test set to get raw vectors
        test_weight_vectors = []
        
        for run_id, file in test_dataset.snapshot_files:
            # Load state dict
            state_dict = torch.load(file, map_location=torch.device('cpu'))
            
            # Extract parameters
            weight_vector = []
            for param_name, param in state_dict.items():
                weight_vector.append(param.flatten())
            
            # Concatenate into a single vector
            weight_vector = torch.cat(weight_vector).numpy()
            test_weight_vectors.append(weight_vector)
        
        # Apply the same PCA transform using the train PCA
        X_test = np.vstack(test_weight_vectors)
        X_test_reduced = test_dataset.pca.transform(X_test)
        
        # Replace weight vectors with consistently transformed versions
        test_dataset.weight_vectors = [X_test_reduced[i] for i in range(X_test_reduced.shape[0])]
        
        # Rebuild sequences with the new weight vectors
        test_dataset.sequences = []
        current_run = None
        seq_buffer = []
        
        for i, (run_id, vector) in enumerate(zip(test_dataset.run_info, test_dataset.weight_vectors)):
            # If we're starting a new run or this is the first element
            if current_run != run_id:
                # Clear the buffer when switching runs
                seq_buffer = [vector]
                current_run = run_id
            else:
                # Add to the buffer
                seq_buffer.append(vector)
                
                # If we have enough elements in the buffer, create a sequence
                if len(seq_buffer) > sequence_length:
                    input_seq = seq_buffer[-sequence_length-1:-1]
                    target = seq_buffer[-1]
                    test_dataset.sequences.append((input_seq, target))
        
        print(f"Rebuilt {len(test_dataset.sequences)} test sequences with consistent PCA dimensions")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Example usage
    train_loader, test_loader = prepare_dataloaders(
        snapshot_dir="./data/snapshots/training_runs",
        sequence_length=3,
        train_split=0.7,  # Use first 70% of snapshots from each run for training, last 30% for testing
        batch_size=32
    )
    
    # Print dataset info
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get a batch
    inputs, targets = next(iter(train_loader))
    print(f"Input shape: {inputs.shape}")  # [batch_size, sequence_length, n_features]
    print(f"Target shape: {targets.shape}")  # [batch_size, n_features]