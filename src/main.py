import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

from base_model import train_base_model
from dataset import prepare_dataloaders
from transformer_predictor import (
    train_transformer_predictor,
    evaluate_predictor,
    extrapolation_test,
    plug_in_test
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a weight predictor model")
    
    # Experiment mode
    parser.add_argument("--mode", type=str, default="full", 
                        choices=["full", "train_base", "train_predictor", "evaluate", "extrapolate", "plugin_test"],
                        help="Which part of the experiment to run")
    
    # Base model training params
    parser.add_argument("--train_base", action="store_true", help="Train the base LeNet model (legacy, use --mode instead)")
    parser.add_argument("--base_epochs", type=int, default=20, help="Number of epochs to train the base model")
    parser.add_argument("--saves_per_epoch", type=int, default=1, 
                        help="Number of times to save snapshots per epoch (1 = end of epoch only, >1 = save at regular intervals within epoch)")
    parser.add_argument("--random_seed", type=int, default=None, 
                        help="Random seed for reproducibility")
    parser.add_argument("--num_runs", type=int, default=1, 
                        help="Number of base model training runs with different random seeds")
    
    # Meta predictor params
    parser.add_argument("--sequence_length", type=int, default=3, help="Number of consecutive snapshots to use as input")
    parser.add_argument("--train_split", type=float, default=0.5, help="Fraction of snapshots to use for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training the meta predictor")
    parser.add_argument("--apply_pca", action="store_true", help="Apply PCA to weight vectors")
    parser.add_argument("--n_components", type=int, default=500, help="Number of PCA components to keep")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training the meta predictor")
    parser.add_argument("--predictor_epochs", type=int, default=100, help="Number of epochs to train the meta predictor")
    
    # Evaluation params
    parser.add_argument("--extrapolation_steps", type=int, default=5, help="Number of steps for multi-step extrapolation")
    
    # Model loading
    parser.add_argument("--model_path", type=str, default=None, 
                        help="Path to a saved meta predictor model (for evaluation/extrapolation modes)")
    
    # Other params
    parser.add_argument("--snapshot_dir", type=str, default="./data/snapshots", help="Directory to save weight snapshots")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    
    return parser.parse_args()


def setup_experiment_dirs(args):
    """Set up experiment directories and return paths."""
    
    # Create timestamp for experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment directory
    experiment_dir = Path(args.results_dir) / f"experiment_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    model_dir = experiment_dir / "models"
    plot_dir = experiment_dir / "plots"
    
    model_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    
    # Path for saving the meta predictor model
    model_path = model_dir / "transformer_predictor.pt"
    
    return experiment_dir, model_dir, plot_dir, model_path


def save_experiment_config(experiment_dir, args):
    """Save experiment configuration to a file."""
    
    config_path = experiment_dir / "config.txt"
    
    with open(config_path, "w") as f:
        f.write("Experiment Configuration:\n")
        f.write("=" * 50 + "\n")
        
        # Write all arguments
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        
        f.write("\n")
        
        # Add device info
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        f.write(f"Device: {device}\n")
        
        if device.type == "cuda":
            f.write(f"CUDA Device: {torch.cuda.get_device_name(0)}\n")
            f.write(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")


def save_results_summary(experiment_dir, base_model_accuracy, mse, mse_per_step, accuracy_results):
    """Save a summary of experiment results."""
    
    summary_path = experiment_dir / "results_summary.txt"
    
    with open(summary_path, "w") as f:
        f.write("Experiment Results Summary:\n")
        f.write("=" * 50 + "\n\n")
        
        # Base model results
        if base_model_accuracy is not None:
            f.write(f"Base Model Final Accuracy: {base_model_accuracy:.2f}%\n\n")
        
        # Direct prediction error
        f.write(f"Direct Prediction MSE: {mse:.6f}\n\n")
        
        # Multi-step prediction errors
        f.write("Multi-step Prediction MSEs:\n")
        for i, step_mse in enumerate(mse_per_step):
            f.write(f"  Step {i+1}: {step_mse:.6f}\n")
        f.write("\n")
        
        # Downstream task evaluation
        f.write("Plug-in Test Accuracies:\n")
        for i, accuracy in enumerate(accuracy_results):
            f.write(f"  Step {i+1}: {accuracy:.2f}%\n")


def plot_weight_trajectory(plot_dir, actual_weights, predicted_weights):
    """Create a PCA visualization of the weight trajectory."""
    
    # Apply PCA for visualization
    from sklearn.decomposition import PCA
    
    # Combine actual and predicted weights
    combined = np.vstack([actual_weights, predicted_weights])
    
    # Apply PCA
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)
    
    # Split back into actual and predicted
    actual_2d = combined_2d[:len(actual_weights)]
    predicted_2d = combined_2d[len(actual_weights):]
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot trajectory lines
    plt.plot(actual_2d[:, 0], actual_2d[:, 1], 'b-', label='Actual Trajectory')
    plt.plot(predicted_2d[:, 0], predicted_2d[:, 1], 'r-', label='Predicted Trajectory')
    
    # Plot points
    plt.scatter(actual_2d[:, 0], actual_2d[:, 1], c='blue', alpha=0.7)
    plt.scatter(predicted_2d[:, 0], predicted_2d[:, 1], c='red', alpha=0.7)
    
    # Add arrows to show direction
    for i in range(len(actual_2d) - 1):
        plt.arrow(actual_2d[i, 0], actual_2d[i, 1], 
                 actual_2d[i+1, 0] - actual_2d[i, 0], actual_2d[i+1, 1] - actual_2d[i, 1],
                 head_width=0.05, head_length=0.1, fc='blue', ec='blue', alpha=0.5)
        
        plt.arrow(predicted_2d[i, 0], predicted_2d[i, 1], 
                 predicted_2d[i+1, 0] - predicted_2d[i, 0], predicted_2d[i+1, 1] - predicted_2d[i, 1],
                 head_width=0.05, head_length=0.1, fc='red', ec='red', alpha=0.5)
    
    # Add labels and title
    plt.title('PCA Visualization of Weight Space Trajectory')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.savefig(plot_dir / 'weight_trajectory.png', dpi=300)
    plt.close()


def load_model(model_path, device):
    """Load a saved transformer predictor model."""
    from transformer_predictor import TransformerPredictor
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model instance
    model = TransformerPredictor(
        input_dim=checkpoint['input_dim'],
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {model_path}")
    return model, checkpoint


def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Set up experiment directories
    experiment_dir, model_dir, plot_dir, model_path = setup_experiment_dirs(args)
    
    # Save experiment configuration
    save_experiment_config(experiment_dir, args)
    
    # Initialize variables
    base_model_accuracy = None
    model = None
    mse = None
    mse_per_step = []
    accuracy_results = []
    
    # For backwards compatibility
    if args.train_base and args.mode == "full":
        args.mode = "train_base"
    
    # Process based on mode
    if args.mode == "train_base" or args.mode == "full":
        print("\n=== Training Base Model ===")
        
        # Set up base models directory structure for multiple runs
        base_snapshot_dir = Path(args.snapshot_dir)
        if args.num_runs > 1:
            # Create a parent directory for all runs
            base_snapshot_dir = Path(args.snapshot_dir) / "multiple_runs"
            base_snapshot_dir.mkdir(exist_ok=True)
        
        for run in range(1, args.num_runs + 1):
            # Set a different random seed for each run
            if args.random_seed is not None:
                run_seed = args.random_seed + run - 1
                torch.manual_seed(run_seed)
                np.random.seed(run_seed)
                print(f"Run {run}/{args.num_runs} with random seed: {run_seed}")
            else:
                print(f"Run {run}/{args.num_runs} with random initialization")
            
            # Set up snapshot directory for this run
            if args.num_runs > 1:
                run_snapshot_dir = base_snapshot_dir / f"run_{run}"
                run_snapshot_dir.mkdir(exist_ok=True)
            else:
                run_snapshot_dir = base_snapshot_dir
            
            # Train the base model with the specified number of saves per epoch
            base_model = train_base_model(
                num_epochs=args.base_epochs, 
                snapshot_dir=str(run_snapshot_dir),
                saves_per_epoch=args.saves_per_epoch
            )
            
            # Evaluate base model
            base_model.eval()
            from transformer_predictor import evaluate_mnist
            run_accuracy = evaluate_mnist(base_model, device)
            print(f"Base model final accuracy (Run {run}): {run_accuracy:.2f}%")
            
            # Store the accuracy of the last run
            if run == args.num_runs:
                base_model_accuracy = run_accuracy
    
    if args.mode == "train_predictor" or args.mode == "full":
        print("\n=== Training Meta Predictor ===")
        
        # Update snapshot directory if using multiple runs
        if args.num_runs > 1 and (args.mode == "full" or Path(args.snapshot_dir).name == "multiple_runs"):
            print("Using snapshots from multiple runs for training the predictor")
            
            # If in full mode, use the directory we just created
            if args.mode == "full":
                predictor_snapshot_dir = str(base_snapshot_dir)
            else:
                # If only in train_predictor mode, use the specified directory
                predictor_snapshot_dir = args.snapshot_dir
        else:
            predictor_snapshot_dir = args.snapshot_dir
            
        model, train_losses, test_losses = train_transformer_predictor(
            snapshot_dir=predictor_snapshot_dir,
            sequence_length=args.sequence_length,
            train_split=args.train_split,
            batch_size=args.batch_size,
            apply_pca=args.apply_pca,
            n_components=args.n_components,
            learning_rate=args.learning_rate,
            num_epochs=args.predictor_epochs,
            device=device,
            model_save_path=str(model_path)
        )
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training and Testing Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_dir / 'training_loss.png')
        plt.close()
    
    # For evaluation modes, load model if not already created
    if args.mode in ["evaluate", "extrapolate", "plugin_test"] and model is None:
        if args.model_path is None:
            print("Error: --model_path must be specified for evaluation modes")
            return
        model, checkpoint = load_model(args.model_path, device)
    
    if args.mode == "evaluate" or args.mode == "full":
        print("\n=== Evaluating Direct Prediction ===")
        mse, actual_weights, predicted_weights = evaluate_predictor(
            model=model,
            snapshot_dir=args.snapshot_dir,
            sequence_length=args.sequence_length,
            train_split=args.train_split,
            apply_pca=args.apply_pca,
            n_components=args.n_components,
            device=device
        )
        
        # Create weight trajectory visualization
        plot_weight_trajectory(plot_dir, actual_weights, predicted_weights)
    
    if args.mode == "extrapolate" or args.mode == "full":
        print("\n=== Testing Multi-step Extrapolation ===")
        mse_per_step, predicted_vectors, true_vectors = extrapolation_test(
            model=model,
            snapshot_dir=args.snapshot_dir,
            sequence_length=args.sequence_length,
            train_split=args.train_split,
            num_steps=args.extrapolation_steps,
            apply_pca=args.apply_pca,
            n_components=args.n_components,
            device=device
        )
        
        # Plot MSE per step
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(mse_per_step) + 1), mse_per_step, marker='o')
        plt.xlabel('Extrapolation Step')
        plt.ylabel('MSE')
        plt.title('Error Accumulation in Multi-Step Prediction')
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_dir / 'extrapolation_error.png')
        plt.close()
        
        # Save predicted vectors for plugin test
        if args.mode != "full":  # In full mode we'll do plugin test right away
            torch.save({
                'predicted_vectors': predicted_vectors,
                'true_vectors': true_vectors
            }, model_dir / 'extrapolation_results.pt')
    
    if args.mode == "plugin_test" or args.mode == "full":
        # For standalone plugin test, load predicted vectors if not already available
        if args.mode == "plugin_test" and 'predicted_vectors' not in locals():
            extrapolation_results_path = model_dir / 'extrapolation_results.pt' if args.model_path is None else Path(args.model_path).parent / 'extrapolation_results.pt'
            
            if not extrapolation_results_path.exists() and args.model_path is not None:
                print("\n=== Testing Multi-step Extrapolation First ===")
                mse_per_step, predicted_vectors, true_vectors = extrapolation_test(
                    model=model,
                    snapshot_dir=args.snapshot_dir,
                    sequence_length=args.sequence_length,
                    train_split=args.train_split,
                    num_steps=args.extrapolation_steps,
                    apply_pca=args.apply_pca,
                    n_components=args.n_components,
                    device=device
                )
            else:
                print(f"Loading extrapolation results from {extrapolation_results_path}")
                results = torch.load(extrapolation_results_path)
                predicted_vectors = results['predicted_vectors']
        
        print("\n=== Performing Plug-in Test ===")
        accuracy_results = plug_in_test(
            predicted_weights=predicted_vectors,
            snapshot_dir=args.snapshot_dir,
            train_split=args.train_split,
            device=device
        )
        
        # Plot accuracy results
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(accuracy_results) + 1), accuracy_results, marker='o')
        plt.axhline(y=base_model_accuracy if base_model_accuracy is not None else 0, 
                    linestyle='--', color='r', label='Base Model Final Accuracy')
        plt.xlabel('Extrapolation Step')
        plt.ylabel('Accuracy (%)')
        plt.title('Downstream Task Performance with Predicted Weights')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(plot_dir / 'plug_in_accuracy.png')
        plt.close()
    
    # Save results summary if we have results to save
    if mse is not None or mse_per_step or accuracy_results:
        save_results_summary(experiment_dir, base_model_accuracy, 
                            mse if mse is not None else float('nan'), 
                            mse_per_step, 
                            accuracy_results)
    
    print(f"\nExperiment completed! Results saved to {experiment_dir}")


if __name__ == "__main__":
    main()