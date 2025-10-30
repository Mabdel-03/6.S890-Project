# Project Structure Guide

This document explains the organization and purpose of each directory and key file in the Echo(I) project repository.

## Overview

The repository is organized into distinct phases of the ML research pipeline:
1. **Planning**: Project management and timeline
2. **Data**: Data collection, preprocessing, and management
3. **Representation Learning**: Feature learning and encoding
4. **End-to-End Modeling**: Complete model training and evaluation

## Directory Structure

### 📋 Root Level

- **`README.md`**: Main project documentation and quick start guide
- **`requirements.txt`**: Python package dependencies
- **`setup.py`**: Package installation configuration
- **`.gitignore`**: Git version control ignore rules
- **`Makefile`**: Common development commands
- **PDFs**: Project proposal and roadmap documents

### 📊 Planning (`planning/`)

Contains project planning documents:
- **`project_plan.md`**: Detailed timeline, milestones, and task breakdown
- Use this directory for:
  - Meeting notes
  - Literature reviews
  - Experiment plans
  - Design documents

### 💾 Data (`data/`)

Organized data management:

- **`raw/`**: Original, immutable data
  - Never modify files here
  - Document data sources
  - Include download scripts if applicable

- **`processed/`**: Cleaned and preprocessed data ready for modeling
  - Normalized/standardized data
  - Train/val/test splits
  - Feature-engineered datasets

- **`interim/`**: Intermediate data transformations
  - Temporary processing steps
  - Cached computations

### 📓 Notebooks (`notebooks/`)

Jupyter notebooks for exploration and analysis:
- **`01_data_exploration.ipynb`**: Initial data analysis
- Naming convention: `XX_descriptive_name.ipynb`
- Use for:
  - Exploratory data analysis
  - Visualization
  - Prototyping
  - Results analysis
  
**Note**: For production code, migrate logic from notebooks to `src/`

### 🔧 Source Code (`src/`)

Main codebase organized by functionality:

#### `src/data/`
Data loading and preprocessing modules:
- **`dataset.py`**: PyTorch Dataset implementations
- **`dataloader.py`**: DataLoader utilities and configurations
- **`transforms.py`**: (Add) Data augmentation and transformations
- **`preprocessing.py`**: (Add) Data cleaning functions

#### `src/models/`
Model architecture definitions:
- **`base_model.py`**: Base model class and common interfaces
- Add specific model implementations here
- Keep models modular and reusable

#### `src/representation/`
Representation learning modules:
- **`encoder.py`**: Encoder architectures for representation learning
- **`losses.py`**: (Add) Contrastive losses, reconstruction losses
- **`embeddings.py`**: (Add) Embedding layers and projections
- Implement self-supervised learning approaches here

#### `src/training/`
Training pipeline and optimization:
- **`trainer.py`**: Main training loop implementation
- **`optimizer.py`**: (Add) Custom optimizers and schedulers
- **`callbacks.py`**: (Add) Training callbacks (early stopping, etc.)

#### `src/evaluation/`
Model evaluation and metrics:
- **`metrics.py`**: Metric computation functions
- **`visualize.py`**: (Add) Result visualization functions
- **`analysis.py`**: (Add) Error analysis tools

#### `src/utils/`
Utility functions and helpers:
- **`config.py`**: Configuration file management
- **`logging.py`**: Logging setup and utilities
- **`checkpointing.py`**: (Add) Model checkpoint management
- **`reproducibility.py`**: (Add) Random seed and reproducibility utilities

### ⚙️ Configurations (`configs/`)

YAML/JSON configuration files:
- **`default_config.yaml`**: Default hyperparameters and settings
- Create experiment-specific configs: `experiment_name.yaml`
- Organization:
  ```yaml
  data:      # Data parameters
  model:     # Model architecture
  training:  # Training hyperparameters
  evaluation: # Evaluation settings
  ```

### 📜 Scripts (`scripts/`)

Executable Python scripts:
- **`train.py`**: Main training script
- **`evaluate.py`**: Model evaluation script
- **`preprocess.py`**: (Add) Data preprocessing pipeline
- **`export_model.py`**: (Add) Model export for deployment

Usage:
```bash
python scripts/train.py --config configs/my_experiment.yaml
```

### 🧪 Experiments (`experiments/`)

Track individual experiments:
```
experiments/
└── experiment_2024_10_30/
    ├── config.yaml
    ├── logs/
    ├── checkpoints/
    └── results.json
```

- Each experiment gets its own directory
- Include configuration for reproducibility
- Store experiment-specific outputs

### 📈 Results (`results/`)

Centralized output storage:

- **`checkpoints/`**: Model checkpoints (`.pth`, `.ckpt` files)
  - Save best model, latest model, epoch-specific checkpoints
  
- **`figures/`**: Generated plots and visualizations
  - Training curves
  - Evaluation plots
  - Analysis visualizations
  
- **`metrics/`**: Quantitative evaluation results
  - JSON/CSV files with metrics
  - Comparison tables

### 🧪 Tests (`tests/`)

Unit tests for code validation:
- **`test_sample.py`**: Example test file
- Naming convention: `test_*.py`
- Test coverage:
  - Data loading and preprocessing
  - Model forward passes
  - Training step correctness
  - Metric computation

Run tests:
```bash
pytest tests/
```

### 📚 Documentation (`docs/`)

Additional documentation:
- **`API.md`**: API reference for modules
- **`CONTRIBUTING.md`**: Contribution guidelines
- **`PROJECT_STRUCTURE.md`**: This file
- Add:
  - Model architecture descriptions
  - Training procedures
  - Troubleshooting guides

## Workflow

### 1. Starting a New Experiment

1. Create experiment plan in `planning/`
2. Prepare data in `data/raw/`
3. Explore data in `notebooks/`
4. Create configuration in `configs/`
5. Implement/modify code in `src/`
6. Run training with `scripts/train.py`
7. Evaluate with `scripts/evaluate.py`
8. Analyze results in `notebooks/`

### 2. Adding New Features

1. Implement in appropriate `src/` module
2. Add tests in `tests/`
3. Update documentation
4. Add example in `notebooks/` if applicable

### 3. Reproducing Results

1. Use configuration file from `experiments/`
2. Ensure same data splits
3. Set random seed
4. Run with same hyperparameters

## Best Practices

### Code Organization
- Keep modules small and focused
- Use clear naming conventions
- Document functions and classes
- Follow PEP 8 style guidelines

### Data Management
- Never modify raw data
- Document preprocessing steps
- Version control preprocessing scripts
- Keep data and code separate

### Experiment Tracking
- Use meaningful experiment names
- Save all configurations
- Log hyperparameters
- Track metrics systematically

### Version Control
- Commit frequently with clear messages
- Don't commit large files (use Git LFS or `.gitignore`)
- Don't commit data or model checkpoints
- Keep notebooks clean before committing

## File Naming Conventions

- Python modules: `lowercase_with_underscores.py`
- Classes: `CamelCase`
- Functions: `lowercase_with_underscores`
- Constants: `UPPERCASE_WITH_UNDERSCORES`
- Notebooks: `XX_descriptive_name.ipynb` (XX = sequence number)
- Configs: `experiment_name.yaml`
- Checkpoints: `model_epoch_{epoch}.pth` or `best_model.pth`

## Dependencies

Core dependencies are listed in `requirements.txt`:
- PyTorch for deep learning
- NumPy, Pandas for data manipulation
- Scikit-learn for metrics
- Matplotlib, Seaborn for visualization
- Weights & Biases, TensorBoard for experiment tracking

Install with:
```bash
pip install -r requirements.txt
```

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Best Practices for ML Projects](https://github.com/drivendata/cookiecutter-data-science)
- [Git Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows)


