# API Documentation

## Data Module

### Dataset

```python
from src.data.dataset import EchoDataset

dataset = EchoDataset(data_path='data/raw', transform=None)
```

**Parameters:**
- `data_path` (str): Path to the data directory
- `transform` (callable, optional): Transform to apply to samples

### DataLoader

```python
from src.data.dataloader import get_dataloader

dataloader = get_dataloader(
    dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=4
)
```

**Parameters:**
- `dataset`: PyTorch Dataset object
- `batch_size` (int): Batch size for training
- `shuffle` (bool): Whether to shuffle data
- `num_workers` (int): Number of workers for data loading

## Models Module

### Base Model

```python
from src.models.base_model import BaseModel

model = BaseModel(config)
```

**Methods:**
- `forward(x)`: Forward pass
- `get_num_params()`: Get number of parameters

## Representation Module

### Encoder

```python
from src.representation.encoder import RepresentationEncoder

encoder = RepresentationEncoder(
    input_dim=512,
    hidden_dim=256,
    output_dim=128
)
```

**Parameters:**
- `input_dim` (int): Dimension of input features
- `hidden_dim` (int): Dimension of hidden layers
- `output_dim` (int): Dimension of output representations

## Training Module

### Trainer

```python
from src.training.trainer import Trainer

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device='cuda'
)

# Train for one epoch
train_loss = trainer.train_epoch(train_dataloader)

# Validate
val_loss = trainer.validate(val_dataloader)
```

## Evaluation Module

### Metrics

```python
from src.evaluation.metrics import compute_accuracy, compute_metrics

# Compute accuracy
accuracy = compute_accuracy(predictions, targets)

# Compute all metrics
metrics = compute_metrics(predictions, targets)
# Returns: {'accuracy': ..., 'precision': ..., 'recall': ..., 'f1': ...}
```

## Utils Module

### Configuration

```python
from src.utils.config import load_config, save_config

# Load configuration
config = load_config('configs/default_config.yaml')

# Save configuration
save_config(config, 'configs/my_config.yaml')
```

### Logging

```python
from src.utils.logging import setup_logger

logger = setup_logger(
    name='my_logger',
    log_file='logs/my_log.log',
    level=logging.INFO
)

logger.info('Training started')
```


