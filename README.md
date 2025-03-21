# Transformer Weight Predictor

This project implements an experiment to investigate whether a transformer-based meta predictor can forecast future neural network weight updates from past training snapshots—even extrapolating beyond the range it was trained on.

## Project Overview

The experiment involves:

1. Training a base model (LeNet-5) on MNIST and collecting weight snapshots at regular intervals
2. Creating sequences of consecutive weight snapshots to train a transformer-based predictor
3. Evaluating the predictor's ability to forecast future weight values, both:
   - Through direct prediction error (MSE)
   - Through downstream task performance (MNIST classification accuracy)

## Setup

### Requirements

This project requires:

- Python 3.7+
- PyTorch 1.8+
- torchvision
- numpy
- matplotlib
- scikit-learn

Install dependencies:

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

### Project Structure

```
transformer-weight-predictor/
├── data/               # Directory for data and weight snapshots
│   └── snapshots/      # Weight snapshots saved during base model training
├── models/             # Directory for saved models
├── results/            # Directory for experiment results
├── src/                # Source code
│   ├── base_model.py   # LeNet-5 implementation and training
│   ├── dataset.py      # Dataset utilities for weight sequences
│   ├── main.py         # Main script to run the experiment
│   └── transformer_predictor.py # Transformer-based weight predictor
└── README.md           # This file
```

## Running the Experiment

The experiment can be run in several modes, either executing the entire pipeline or individual components.

### Modes of Operation

You can use the `--mode` flag to specify which part of the experiment to run:

- `full`: Run the complete experiment pipeline (default)
- `train_base`: Train only the base LeNet model
- `train_predictor`: Train only the transformer-based meta predictor
- `evaluate`: Evaluate the predictor on held-out epochs (direct prediction)
- `extrapolate`: Test multi-step extrapolation
- `plugin_test`: Perform plug-in test on predicted weights

### Examples

#### 1. Train the Base Model Only

Train the LeNet-5 model on MNIST and save weight snapshots:

```bash
# Train multiple models with different random seeds
python3 src/main.py --mode train_base --training_epochs 2 --num_runs 10 --random_seed 42 --snapshots_per_epoch 20 --snapshot_dir ./data/snapshots
```

#### 2. Train the Weight Predictor Only

After collecting snapshots, train the transformer-based weight predictor:

```bash
python3 src/main.py --mode train_predictor --sequence_length 3 --train_split 0.5 --batch_size 32 --apply_pca --n_components 500 --predictor_epochs 100 --snapshot_dir ./data/snapshots
```

#### 3. Evaluate a Trained Model

Evaluate a trained predictor on held-out epochs:

```bash
python3 src/main.py --mode evaluate --model_path ./results/experiment_TIMESTAMP/models/transformer_predictor.pt --snapshot_dir ./data/snapshots --apply_pca --n_components 500
```

#### 4. Test Multi-step Extrapolation

Test a trained predictor on multi-step extrapolation:

```bash
python3 src/main.py --mode extrapolate --model_path ./results/experiment_TIMESTAMP/models/transformer_predictor.pt --extrapolation_steps 5 --snapshot_dir ./data/snapshots --apply_pca --n_components 500
```

#### 5. Run Plug-in Test

Test how predicted weights perform when plugged into the base model:

```bash
python3 src/main.py --mode plugin_test --model_path ./results/experiment_TIMESTAMP/models/transformer_predictor.pt --snapshot_dir ./data/snapshots --apply_pca --n_components 500
```

#### 6. Full Experiment Pipeline

To run the complete experiment pipeline, including base model training, meta predictor training, and all evaluations:

```bash
# Full pipeline with multiple training runs
python3 src/main.py --mode full --training_epochs 2 --num_runs 5 --random_seed 42 --snapshots_per_epoch 10 --sequence_length 3 --train_split 0.5 --batch_size 32 --apply_pca --n_components 64 --predictor_epochs 100 --extrapolation_steps 5
```

#### Quick Testing

For quick testing of the pipeline with minimal computational resources:

```bash
python3 src/main.py --mode full --base_epochs 3 --sequence_length 3 --train_split 0.5 --batch_size 32 --apply_pca --n_components 100 --predictor_epochs 5 --extrapolation_steps 2
```

### Command-Line Arguments

The main script accepts the following arguments:

#### Experiment Mode:

- `--mode`: Which part of the experiment to run (choices: "full", "train_base", "train_predictor", "evaluate", "extrapolate", "plugin_test") (default: "full")

#### Base Model Training:

- `--train_base`: Legacy flag to train the base LeNet model (use --mode instead)
- `--training_epochs`: Number of epochs to train the base model (default: 20)
- `--snapshots_per_epoch`: Number of weight snapshots to save per epoch (default: 1)
- `--random_seed`: Set a random seed for reproducibility
- `--num_runs`: Number of training runs with different random seeds (default: 3)

#### Meta Predictor Parameters:

- `--sequence_length`: Number of consecutive snapshots to use as input (default: 3)
- `--train_split`: Fraction of snapshots to use for training (default: 0.5)
- `--batch_size`: Batch size for training the meta predictor (default: 32)
- `--apply_pca`: Flag to apply PCA to weight vectors
- `--n_components`: Number of PCA components to keep (default: 500)
- `--learning_rate`: Learning rate for training the meta predictor (default: 1e-4)
- `--predictor_epochs`: Number of epochs to train the meta predictor (default: 100)

#### Evaluation Parameters:

- `--extrapolation_steps`: Number of steps for multi-step extrapolation (default: 5)
- `--model_path`: Path to a saved meta predictor model (required for evaluate/extrapolate/plugin_test modes)

#### Other Parameters:

- `--snapshot_dir`: Directory to save weight snapshots (default: ./data/snapshots)
- `--results_dir`: Directory to save results (default: ./results)
- `--no_cuda`: Flag to disable CUDA

## Understanding Results

Experiment results are saved in the `results` directory, with a timestamped folder for each run. Each experiment folder contains:

1. Configuration information in `config.txt`
2. Results summary in `results_summary.txt` with three key metrics:
   - **Direct Prediction MSE**: Mean squared error between predicted weights and actual weights
   - **Multi-step Prediction MSEs**: Error accumulation over multiple prediction steps
   - **Plug-in Test Accuracies**: MNIST classification accuracy when using predicted weights
3. Saved model weights in the `models` subdirectory
4. Visualizations in the `plots` subdirectory:
   - Training and testing loss curves
   - PCA visualization of weight space trajectories
   - Error accumulation in multi-step prediction
   - Downstream task performance with predicted weights

### Interpreting Plug-in Accuracy

The "plug-in accuracy" is a crucial metric that evaluates how well the predicted weights perform on the actual downstream task (MNIST classification). This measures the practical usefulness of our weight predictions:

1. For each prediction step, we take the weights predicted by our transformer model
2. "Plug" these weights into a fresh LeNet5 model (replacing its normal weights)
3. Evaluate the model on the MNIST test dataset
4. Report the classification accuracy percentage

A high plug-in accuracy indicates that our predicted weights are not just mathematically close to the actual weights (MSE), but also functionally equivalent in terms of model performance.

Typically, we see a progression in accuracy across prediction steps, which shows how the transformer's predictions improve as it generates weights corresponding to later training stages. For example, a pattern like:

- Step 1: 25% (poor performance)
- Step 3: 60% (moderate performance)
- Step 5: 90% (excellent performance)

This progression demonstrates that the transformer is learning meaningful patterns in how model weights evolve during training.

## Extending the Project

There are several ways to extend this project:

1. Implement the diffusion-based predictor option as an alternative
2. Try different base model architectures or datasets
3. Experiment with different sequence lengths and train/test splits
4. Explore richer feature representations for the weight vectors
5. Implement a learned optimizer using the predictive model

## License

This project is open source and available under the MIT License.
