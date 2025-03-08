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
        
        # Check if we are working with multiple runs
        is_multiple_runs = False
        if (self.snapshot_dir / "run_1").exists():
            is_multiple_runs = True
            print(f"Found multiple training runs in {snapshot_dir}")
        
        if is_multiple_runs:
            # Collect snapshots from all run directories
            self.snapshot_files = []
            run_dirs = sorted([d for d in self.snapshot_dir.glob("run_*") if d.is_dir()], 
                             key=lambda x: int(x.name.split('_')[1]))
            
            for run_dir in run_dirs:
                # Match both regular epoch snapshots and intermediate (batch-level) snapshots
                run_files = []
                
                # Add epoch snapshots
                epoch_files = sorted([f for f in run_dir.glob("epoch_*.pt") if "batch" not in f.name], 
                                   key=lambda x: int(x.stem.split('_')[1]))
                run_files.extend(epoch_files)
                
                # Add intermediate batch snapshots if present
                batch_files = []
                for f in run_dir.glob("epoch_*_batch_*.pt"):
                    # Extract epoch and batch percentage
                    parts = f.stem.split('_')
                    epoch = int(parts[1])
                    pct = int(parts[3].replace('pct', ''))
                    # Use decimal position to order correctly (e.g., epoch 1.2, 1.4, 1.6)
                    batch_files.append((epoch + pct/100, f))
                
                # Sort batch files and add to the list
                batch_files.sort()
                run_files.extend([f for _, f in batch_files])
                
                # Add run information to each file
                run_id = int(run_dir.name.split('_')[1])
                self.snapshot_files.extend([(run_id, f) for f in run_files])
        else:
            # Regular single run handling - include both epoch and batch snapshots
            # First, get regular epoch snapshots
            epoch_files = sorted([f for f in self.snapshot_dir.glob("epoch_*.pt") if "batch" not in f.name], 
                               key=lambda x: int(x.stem.split('_')[1]))
            
            # Then, get batch snapshots with decimal ordering
            batch_files = []
            for f in self.snapshot_dir.glob("epoch_*_batch_*.pt"):
                parts = f.stem.split('_')
                epoch = int(parts[1])
                pct = int(parts[3].replace('pct', ''))
                # Store with a decimal position for correct ordering
                batch_files.append((epoch + pct/100, f))
            
            # Combine all files, properly sorted
            all_files = epoch_files.copy()
            batch_files.sort()
            all_files.extend([f for _, f in batch_files])
            
            # Use run_id 0 for single runs
            self.snapshot_files = [(0, f) for f in all_files]
        
        # Determine total number of snapshots
        self.total_snapshots = len(self.snapshot_files)
        self.split_idx = int(self.total_snapshots * train_split)
        
        # Select snapshots based on train/test split
        if is_train:
            self.snapshot_files = self.snapshot_files[:self.split_idx]
        else:
            self.snapshot_files = self.snapshot_files[self.split_idx:]
        
        # Load all weight vectors
        self.weight_vectors, self.run_info = self._load_weight_vectors()
        
        # Apply PCA if requested
        if self.apply_pca:
            self._apply_pca()
        
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
    
    def _apply_pca(self):
        """Apply PCA dimensionality reduction to weight vectors."""
        # Stack weight vectors into a matrix
        X = np.vstack(self.weight_vectors)
        
        # Fit PCA with appropriate number of components
        # Use min(self.n_components, X.shape[0], X.shape[1]) to avoid the error
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