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

### 1. Train the Base Model and Collect Snapshots

First, train the LeNet-5 model on MNIST and save weight snapshots:

```bash
python src/main.py --train_base --base_epochs 20 --snapshot_dir ./data/snapshots
```

### 2. Train and Evaluate the Weight Predictor

Train the transformer-based weight predictor using the collected snapshots:

```bash
python src/main.py --sequence_length 3 --train_split 0.5 --batch_size 32 --apply_pca --n_components 500 --predictor_epochs 100 --snapshot_dir ./data/snapshots
```

### Full Experiment Pipeline

To run the complete experiment pipeline, including base model training, meta predictor training, and evaluation:

```bash
python src/main.py --train_base --base_epochs 20 --sequence_length 3 --train_split 0.5 --batch_size 32 --apply_pca --n_components 500 --predictor_epochs 100 --extrapolation_steps 5
```

### Command-Line Arguments

The main script accepts the following arguments:

#### Base Model Training:
- `--train_base`: Flag to train the base LeNet model
- `--base_epochs`: Number of epochs to train the base model (default: 20)

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

#### Other Parameters:
- `--snapshot_dir`: Directory to save weight snapshots (default: ./data/snapshots)
- `--results_dir`: Directory to save results (default: ./results)
- `--no_cuda`: Flag to disable CUDA

## Experiment Results

Experiment results are saved in the `results` directory, with a timestamped folder for each run. Each experiment folder contains:

1. Configuration information in `config.txt`
2. Results summary in `results_summary.txt`
3. Saved model weights in the `models` subdirectory
4. Visualizations in the `plots` subdirectory:
   - Training and testing loss curves
   - PCA visualization of weight space trajectories
   - Error accumulation in multi-step prediction
   - Downstream task performance with predicted weights

## Extending the Project

There are several ways to extend this project:

1. Implement the diffusion-based predictor option as an alternative
2. Try different base model architectures or datasets
3. Experiment with different sequence lengths and train/test splits
4. Explore richer feature representations for the weight vectors
5. Implement a learned optimizer using the predictive model

## License

This project is open source and available under the MIT License.# transformer-weight-predictor
