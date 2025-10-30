# Contributing to Echo(I)

Thank you for your interest in contributing to the Echo(I) project!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone <your-fork-url>`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Commit your changes: `git commit -m "Add your message"`
6. Push to your fork: `git push origin feature/your-feature-name`
7. Create a Pull Request

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Code Style

We follow PEP 8 style guidelines. Please ensure your code:

- Is formatted with `black`
- Has imports sorted with `isort`
- Passes `flake8` linting
- Includes type hints where appropriate
- Is well-documented with docstrings

Run formatting and linting:
```bash
make format
make lint
```

## Testing

All new features should include tests. Run tests with:

```bash
make test
```

Tests should be placed in the `tests/` directory and follow the naming convention `test_*.py`.

## Documentation

- Update docstrings for any modified functions/classes
- Update README.md if adding new features
- Add examples for new functionality

## Pull Request Process

1. Ensure all tests pass
2. Update documentation
3. Describe your changes in the PR description
4. Reference any related issues

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on what is best for the project

## Questions?

Feel free to open an issue for questions or discussions.


