.PHONY: help install clean test format lint train evaluate

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make clean      - Clean temporary files and caches"
	@echo "  make test       - Run tests"
	@echo "  make format     - Format code with black and isort"
	@echo "  make lint       - Run linting checks"
	@echo "  make train      - Train model with default config"
	@echo "  make evaluate   - Evaluate model"

install:
	pip install -r requirements.txt
	pip install -e .

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build dist .pytest_cache .coverage htmlcov

test:
	pytest tests/ -v

format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

lint:
	flake8 src/ scripts/ tests/
	black --check src/ scripts/ tests/
	isort --check src/ scripts/ tests/

train:
	python scripts/train.py --config configs/default_config.yaml

evaluate:
	@echo "Please specify checkpoint path:"
	@echo "python scripts/evaluate.py --config configs/default_config.yaml --checkpoint <path>"


