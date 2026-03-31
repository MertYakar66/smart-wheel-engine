# Contributing to Smart Wheel Engine

Thank you for your interest in contributing to Smart Wheel Engine! This document provides guidelines for contributing.

## Getting Started

### Prerequisites

- Python 3.11+
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/MertYakar66/smart-wheel-engine.git
cd smart-wheel-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write clean, documented code
- Follow existing code style (we use Ruff for formatting)
- Add tests for new functionality

### 3. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_option_pricer.py -v

# Run with coverage
pytest tests/ --cov=engine --cov=dashboard
```

### 4. Pre-commit Checks

```bash
# Run pre-commit on all files
pre-commit run --all-files

# Or let it run automatically on commit
git commit -m "Your message"
```

### 5. Submit Pull Request

1. Push your branch to GitHub
2. Open a Pull Request against `main`
3. Describe your changes clearly
4. Wait for CI checks to pass
5. Request review

## Code Guidelines

### Python Style

- Follow PEP 8
- Use type hints for all functions
- Document public functions with docstrings
- Keep functions focused and small

```python
def calculate_option_price(
    spot: float,
    strike: float,
    volatility: float,
    option_type: Literal['call', 'put']
) -> float:
    """
    Calculate option price using Black-Scholes.

    Args:
        spot: Current stock price
        strike: Option strike price
        volatility: Annualized volatility
        option_type: 'call' or 'put'

    Returns:
        Option price
    """
    ...
```

### Testing

- Write tests for all new functionality
- Use descriptive test names
- Test edge cases

```python
class TestOptionPricing:
    def test_atm_call_delta_near_half(self):
        """ATM call delta should be approximately 0.5."""
        ...

    def test_expired_option_returns_intrinsic(self):
        """Expired option should return intrinsic value."""
        ...
```

### Commits

- Use clear, descriptive commit messages
- Reference issues when applicable

```
feat: Add third-order Greeks (Speed, Color, Ultima)

- Implement Speed (dGamma/dS) for gamma sensitivity
- Implement Color (dGamma/dT) for gamma decay
- Implement Ultima (dVolga/dSigma) for vega convexity
- Add comprehensive tests

Closes #123
```

## Project Structure

```
smart-wheel-engine/
├── dashboard/          # Dashboard module
├── engine/             # Core quantitative engine
├── src/                # Source modules
├── tests/              # Test suite
├── utils/              # Utilities
├── docs/               # Documentation
└── config/             # Configuration
```

## Areas for Contribution

### High Priority
- Additional option pricing models
- Performance optimizations
- Documentation improvements
- Test coverage expansion

### Feature Ideas
- Monte Carlo Greeks calculation
- Jump-diffusion pricing models
- Real-time streaming interface
- Additional broker integrations

## Questions?

Open an issue for questions or discussions about contributing.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
