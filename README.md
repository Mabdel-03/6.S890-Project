# Echo(I) - Learning Hierarchical Influence Propagation in Organizational Response Systems

A deep learning framework for predicting organizational responses to recommendations using game theory, graph neural networks, and natural language processing.

## 📋 Project Overview

This repository contains the implementation for the Echo(I) project as part of MIT's 6.S890 course. The project develops a novel framework that integrates **game-theoretic formalization** with **deep learning architectures** to model and predict organizational response dynamics.

### Core Innovation

We formalize organizational response as a **Hierarchical Partially Observable Stochastic Game (H-POSG)** and learn effective representations from text descriptions to predict:
- Individual agent responses across organizational levels
- Collective organizational adoption outcomes
- Influence propagation through hierarchical authority structures

**Research Question:** *Can deep learning models learn effective representations of organizational dynamics from text to predict responses while respecting game-theoretic properties of hierarchical strategic interaction?*

**Key Documents:**
- [Project Proposal](29102025%20-%20Echo(I)%20-%20Proposal.pdf)
- [Project Roadmap](29102025%20-%20Echo(I)%20-%20Roadmap.pdf)

## 📁 Repository Structure

```
6.S890-Project/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup file
├── .gitignore                     # Git ignore rules
│
├── planning/                      # Project planning and documentation
│   └── project_plan.md           # Detailed project timeline and milestones
│
├── data/                          # Data directory
│   ├── raw/                      # Raw, immutable data
│   ├── processed/                # Cleaned and preprocessed data
│   └── interim/                  # Intermediate data transformations
│
├── notebooks/                     # Jupyter notebooks for exploration
│   └── 01_data_exploration.ipynb # Data analysis and visualization
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── data/                     # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py           # Dataset implementations
│   │   └── dataloader.py        # DataLoader utilities
│   │
│   ├── models/                   # Model architectures
│   │   ├── __init__.py
│   │   └── base_model.py        # Base model class
│   │
│   ├── representation/           # Representation learning modules
│   │   ├── __init__.py
│   │   └── encoder.py           # Encoder implementations
│   │
│   ├── training/                 # Training loops and optimization
│   │   ├── __init__.py
│   │   └── trainer.py           # Training logic
│   │
│   ├── evaluation/               # Evaluation and metrics
│   │   ├── __init__.py
│   │   └── metrics.py           # Metric implementations
│   │
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── config.py            # Configuration management
│       └── logging.py           # Logging utilities
│
├── configs/                      # Configuration files
│   └── default_config.yaml      # Default hyperparameters
│
├── scripts/                      # Executable scripts
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation script
│
├── experiments/                  # Experiment tracking
│   └── .gitkeep
│
├── results/                      # Results and outputs
│   ├── checkpoints/             # Model checkpoints
│   ├── figures/                 # Generated figures
│   └── metrics/                 # Evaluation metrics
│
├── tests/                        # Unit tests
│
└── docs/                         # Additional documentation

```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd 6.S890-Project
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install the package in development mode**
```bash
pip install -e .
```

### Quick Start

1. **Prepare your data**
   - Place raw data in `data/raw/`
   - Run preprocessing scripts

2. **Configure your experiment**
   - Edit `configs/default_config.yaml`
   - Set hyperparameters and paths

3. **Train a model**
```bash
python scripts/train.py --config configs/default_config.yaml
```

4. **Evaluate the model**
```bash
python scripts/evaluate.py --config configs/default_config.yaml --checkpoint results/checkpoints/best_model.pth
```

## 🔬 Project Phases

### Phase 1: Planning
- Project definition and scope
- Literature review
- Timeline and milestone planning
- See `planning/project_plan.md` for details

### Phase 2: Data
- **Synthetic Data Generation:** LLM-based generation of 10k organizations with 750k agent responses
- **Real-World Collection:** 500-1000 cases from business schools, corporate reports
- **Agent-Based Modeling:** 100k rule-based simulations for validation
- Exploratory data analysis and quality validation

**Key Notebooks:**
- `notebooks/01_data_exploration.ipynb`: Organizational data analysis
- `notebooks/02_synthetic_generation.ipynb`: LLM-based data generation

**Key Modules:**
- `src/data/org_dataset.py`: Organizational dataset implementations
- `src/data/graph_utils.py`: Authority graph construction
- `scripts/generate_synthetic_data.py`: LLM-based data generation
- `scripts/run_abm_simulation.py`: Agent-based model simulation

### Phase 3: Representation Learning (Dual Encoders)
- **Recommendation Encoder:** Transform-based encoding of recommendation text
- **Organizational Context Encoder:** Encoding of org structure, culture, history
- **Contrastive Learning:** Align semantically compatible recommendation-org pairs
- Embedding quality evaluation (retrieval accuracy, t-SNE visualization)

**Key Modules:**
- `src/representation/dual_encoder.py`: Dual encoder architecture
- `src/representation/contrastive_loss.py`: Contrastive learning objectives
- `src/representation/text_encoder.py`: BERT/RoBERTa encoders

### Phase 4: End-to-End Modeling (H-POSG)
- **Graph Neural Network:** 4-layer GAT for influence propagation
- **Hierarchical Policy Network:** Agent response prediction with hierarchical conditioning
- **Multi-Objective Training:** Prediction + consistency + equilibrium losses
- Integration and end-to-end optimization

**Key Modules:**
- `src/models/h_posg_model.py`: Full H-POSG architecture
- `src/models/gat_propagation.py`: Graph Attention Network
- `src/models/hierarchical_policy.py`: Hierarchical policy network
- `src/training/multi_agent_trainer.py`: Multi-agent training loop
- `src/evaluation/equilibrium_metrics.py`: Game-theoretic evaluation

## 📊 Experiments

Track experiments using the structure:
```
experiments/
└── experiment_name/
    ├── config.yaml          # Experiment configuration
    ├── logs/                # Training logs
    ├── checkpoints/         # Model checkpoints
    └── results.json         # Evaluation results
```

### Recommended Experiment Tracking

We support multiple experiment tracking tools:
- TensorBoard: `tensorboard --logdir results/logs`
- Weights & Biases: Set `use_wandb: true` in config
- MLflow: (Optional) Configure in training script

## 🧪 Testing

Run tests using pytest:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## 📈 Results

Results are organized as:
- `results/checkpoints/`: Saved model checkpoints
- `results/figures/`: Visualization outputs
- `results/metrics/`: Quantitative evaluation results

## 🛠️ Development

### Code Style

We follow PEP 8 style guidelines. Format code using:
```bash
black src/
isort src/
```

Check code quality:
```bash
flake8 src/
```

### Adding New Features

1. Create a new branch: `git checkout -b feature/your-feature`
2. Implement your feature in the appropriate module
3. Add tests in `tests/`
4. Update documentation
5. Submit a pull request

## 📝 Configuration

The project uses YAML configuration files. Key configuration sections:

- **data**: Data paths and preprocessing settings
- **model**: Model architecture parameters
- **representation**: Representation learning settings
- **training**: Training hyperparameters
- **evaluation**: Evaluation metrics and settings
- **experiment**: Experiment tracking configuration

See `configs/default_config.yaml` for a complete example.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

[Add your license information]

## 👥 Team

[Add team member information]

## 🙏 Acknowledgments

- MIT 6.S890 Course Staff
- [Add other acknowledgments]

## 📚 References

[Add key papers and resources]

## 📧 Contact

For questions or issues, please [open an issue](link-to-issues) or contact [your-email].

---

**Last Updated:** October 30, 2025
