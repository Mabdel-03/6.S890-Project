# Getting Started with Echo(I)

Welcome to the Echo(I) project! This guide will help you get up and running quickly.

## Prerequisites

Before you begin, ensure you have:
- Python 3.8 or higher installed
- Git installed
- (Recommended) CUDA-capable GPU for faster training
- 16GB+ RAM

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd 6.S890-Project
```

### 2. Create Virtual Environment

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

This will install:
- PyTorch for deep learning
- NumPy, Pandas for data manipulation
- Scikit-learn for metrics
- Visualization libraries
- And more (see `requirements.txt`)

### 4. Verify Installation

```bash
# Run sample tests
pytest tests/test_sample.py

# Check if PyTorch can access GPU (if available)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Understanding the Project Structure

The repository is organized into four main phases:

### 1️⃣ Planning (`planning/`)
Start here to understand the project timeline and milestones.

### 2️⃣ Data (`data/`)
- `data/raw/`: Place your raw data here
- `data/processed/`: Preprocessed data ready for training
- `data/interim/`: Temporary processing steps

### 3️⃣ Representation Learning (`src/representation/`)
Modules for learning feature representations from data.

### 4️⃣ End-to-End Modeling (`src/models/`, `src/training/`, `src/evaluation/`)
Complete pipeline for training and evaluating models.

## Your First Steps

### 1. Explore the Data

Start by exploring your data in Jupyter notebooks:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

Key tasks:
- Load and visualize your data
- Compute basic statistics
- Identify data quality issues
- Plan preprocessing steps

### 2. Configure Your Experiment

Edit the configuration file:

```bash
# Open the config file
nano configs/default_config.yaml
```

Important settings:
- `data.data_dir`: Path to your data
- `data.batch_size`: Batch size for training
- `training.learning_rate`: Learning rate
- `experiment.name`: Name for your experiment

### 3. Implement Data Loading

Edit `src/data/dataset.py` to load your specific data:

```python
class EchoDataset(Dataset):
    def __init__(self, data_path, transform=None):
        # Load your data
        self.data = load_your_data(data_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
```

### 4. Define Your Model

Edit `src/models/base_model.py`:

```python
class YourModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Define your model architecture
        self.encoder = nn.Linear(config['input_dim'], config['hidden_dim'])
        self.decoder = nn.Linear(config['hidden_dim'], config['output_dim'])
        
    def forward(self, x):
        x = self.encoder(x)
        x = torch.relu(x)
        x = self.decoder(x)
        return x
```

### 5. Run Your First Training

```bash
python scripts/train.py --config configs/default_config.yaml
```

Monitor training progress:
- Check console output for loss values
- View TensorBoard logs: `tensorboard --logdir results/logs`
- Checkpoints saved in `results/checkpoints/`

### 6. Evaluate Your Model

```bash
python scripts/evaluate.py \
    --config configs/default_config.yaml \
    --checkpoint results/checkpoints/best_model.pth \
    --split test
```

## Common Workflows

### Development Workflow

1. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Make changes** to code in `src/`

3. **Test your changes**
   ```bash
   make test
   ```

4. **Format code**
   ```bash
   make format
   ```

5. **Commit and push**
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin feature/your-feature
   ```

### Experiment Workflow

1. **Plan experiment** in `planning/project_plan.md`

2. **Create config** in `configs/experiment_name.yaml`

3. **Run training**
   ```bash
   python scripts/train.py --config configs/experiment_name.yaml
   ```

4. **Track results** in `experiments/experiment_name/`

5. **Analyze in notebook**
   ```bash
   jupyter notebook notebooks/
   ```

6. **Document findings** in experiment notes

## Tips for Success

### 🎯 Start Simple
- Begin with a simple baseline model
- Use a small subset of data for initial experiments
- Verify your pipeline works end-to-end
- Then gradually increase complexity

### 📊 Track Everything
- Use meaningful experiment names
- Save all configurations
- Log hyperparameters and metrics
- Keep notes on what works and what doesn't

### 🔄 Iterate Quickly
- Use fast prototyping in notebooks
- Move working code to `src/` modules
- Automate repetitive tasks with scripts
- Use the Makefile for common commands

### 🧪 Test Your Code
- Write tests for critical functions
- Run tests before committing
- Use continuous integration if available

### 📝 Document As You Go
- Comment complex code sections
- Update docstrings
- Keep README current
- Write experiment notes

## Troubleshooting

### Import Errors

If you get import errors:
```bash
# Reinstall in development mode
pip install -e .

# Check Python path
echo $PYTHONPATH
```

### Out of Memory Errors

If training crashes with OOM:
1. Reduce `batch_size` in config
2. Use gradient accumulation
3. Enable mixed precision training
4. Use a smaller model

### Data Loading Issues

If data doesn't load:
1. Check paths in config file
2. Verify data exists: `ls data/raw/`
3. Check file permissions
4. Review dataset implementation

### Slow Training

To speed up training:
1. Use GPU: `device: 'cuda'` in config
2. Increase `num_workers` in dataloader
3. Use data augmentation on GPU
4. Enable mixed precision training

## Next Steps

Once you're comfortable with the basics:

1. **Implement Representation Learning**
   - Design encoder architecture
   - Implement self-supervised learning
   - Evaluate learned representations

2. **Optimize Your Model**
   - Hyperparameter tuning
   - Architecture search
   - Regularization techniques

3. **Advanced Features**
   - Distributed training
   - Model ensembles
   - Advanced visualizations

4. **Deployment**
   - Export trained models
   - Create inference pipeline
   - Optimize for production

## Resources

### Documentation
- [Full Project Structure](PROJECT_STRUCTURE.md)
- [API Reference](API.md)
- [Quick Reference](QUICK_REFERENCE.md)
- [Contributing Guide](CONTRIBUTING.md)

### External Resources
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Machine Learning Best Practices](https://github.com/drivendata/cookiecutter-data-science)
- [Deep Learning Papers](https://paperswithcode.com/)

## Getting Help

If you run into issues:
1. Check the troubleshooting section above
2. Review the documentation
3. Look for similar issues in the repository
4. Open a new issue with details about your problem

## Welcome to the Team! 🎉

You're now ready to start working on Echo(I). Remember:
- Start small and iterate
- Document your work
- Ask questions
- Have fun learning!

Good luck with your project! 🚀


