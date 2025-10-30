# Quick Reference Guide

A cheat sheet for common operations in the Echo(I) project.

## Setup

```bash
# Clone and setup
git clone <repo-url>
cd 6.S890-Project
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Common Commands

### Using Makefile

```bash
make help          # Show available commands
make install       # Install dependencies
make clean         # Clean temporary files
make test          # Run tests
make format        # Format code
make lint          # Check code quality
make train         # Train with default config
```

### Training

```bash
# Train with default config
python scripts/train.py

# Train with custom config
python scripts/train.py --config configs/my_config.yaml

# Resume from checkpoint
python scripts/train.py --config configs/my_config.yaml --resume results/checkpoints/model.pth
```

### Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py --checkpoint results/checkpoints/best_model.pth

# Evaluate on specific split
python scripts/evaluate.py --checkpoint model.pth --split val
```

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_sample.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v tests/
```

## Code Snippets

### Load Configuration

```python
from src.utils.config import load_config

config = load_config('configs/default_config.yaml')
print(config['training']['learning_rate'])
```

### Setup Logger

```python
from src.utils.logging import setup_logger

logger = setup_logger('my_experiment', log_file='logs/exp.log')
logger.info('Starting experiment')
```

### Create Dataset

```python
from src.data.dataset import EchoDataset
from src.data.dataloader import get_dataloader

dataset = EchoDataset(data_path='data/raw')
dataloader = get_dataloader(dataset, batch_size=32, shuffle=True)
```

### Initialize Model

```python
from src.models.base_model import BaseModel

config = {'input_dim': 512, 'hidden_dim': 256, 'output_dim': 128}
model = BaseModel(config)
print(f"Model has {model.get_num_params()} parameters")
```

### Training Loop

```python
from src.training.trainer import Trainer
import torch.optim as optim
import torch.nn as nn

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
trainer = Trainer(model, optimizer, criterion, device='cuda')

# Train for one epoch
train_loss = trainer.train_epoch(train_dataloader)

# Validate
val_loss = trainer.validate(val_dataloader)
```

### Compute Metrics

```python
from src.evaluation.metrics import compute_metrics

metrics = compute_metrics(predictions, targets)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
```

## Directory Quick Access

```bash
# Project root
cd /path/to/6.S890-Project

# Navigate to common directories
cd data/raw              # Raw data
cd data/processed        # Processed data
cd src                   # Source code
cd configs               # Configurations
cd notebooks             # Jupyter notebooks
cd experiments           # Experiment results
cd results/checkpoints   # Model checkpoints
```

## Git Workflow

```bash
# Create new branch for feature
git checkout -b feature/my-feature

# Check status
git status

# Stage and commit changes
git add .
git commit -m "Add feature description"

# Push to remote
git push origin feature/my-feature

# Pull latest changes
git pull origin main

# View commit history
git log --oneline
```

## Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Start JupyterLab
jupyter lab

# Convert notebook to script
jupyter nbconvert --to script notebook.ipynb
```

## Python Path Setup

When importing from src in scripts or notebooks:

```python
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Or for notebooks
sys.path.append('../src')
```

## Configuration Template

```yaml
# configs/my_experiment.yaml
data:
  data_dir: "data/raw"
  batch_size: 32
  
model:
  input_dim: 512
  hidden_dim: 256
  
training:
  epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
  
experiment:
  name: "my_experiment"
  seed: 42
  device: "cuda"
```

## Common Issues

### Import Errors
```bash
# Make sure package is installed in development mode
pip install -e .

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### CUDA Out of Memory
```python
# Reduce batch size in config
# Enable gradient checkpointing
# Use mixed precision training
```

### Data Not Found
```bash
# Check data directory structure
ls data/raw/
ls data/processed/

# Verify paths in config file
cat configs/default_config.yaml
```

## Useful Tips

- Use `make format` before committing
- Run `make test` to ensure nothing breaks
- Keep notebooks clean (clear outputs before committing)
- Use meaningful experiment names
- Document hyperparameters in config files
- Save checkpoints regularly
- Log metrics during training
- Version control configs with experiments

## Environment Variables

Create `.env` file (don't commit it):
```bash
WANDB_API_KEY=your_key
CUDA_VISIBLE_DEVICES=0
DATA_DIR=data/raw
```

Load in Python:
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('WANDB_API_KEY')
```

## Helpful Links

- PyTorch Documentation: https://pytorch.org/docs/
- Project README: [README.md](../README.md)
- API Reference: [API.md](API.md)
- Project Structure: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)


