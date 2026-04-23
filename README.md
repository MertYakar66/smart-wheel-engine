# Smart Wheel Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-170%2B%20passing-green.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Institutional-grade quantitative trading engine for options strategies.**

A systematic, data-driven framework for managing short-put and covered-call strategies (the "Wheel Strategy"). The engine evaluates trades using probability estimates and expected-value calculations derived from historical data.

## Features

### Option Pricing
- **European Options**: Black-Scholes-Merton with continuous dividend yield
- **American Options**: Barone-Adesi-Whaley (1987) approximation
- **Implied Volatility**: Newton-Raphson with Brent fallback

### Greeks (All Orders)
| Order | Greeks |
|-------|--------|
| First | Delta, Theta, Vega, Rho |
| Second | Gamma, Vanna, Charm, Volga |
| Third | Speed, Color, Ultima |

### Risk Management
- **VaR/CVaR**: Multi-asset covariance with correlation matrix
- **Stress Testing**: Market crash, vol explosion, rate shock scenarios
- **Position Sizing**: Kelly criterion with fractional sizing
- **Portfolio Greeks**: Aggregate exposure tracking

### Professional Dashboard
- Interactive CLI interface
- Option analysis reports
- Portfolio risk reports
- Real-time Greeks surface generation

## Quick Start

### Installation

```bash
git clone https://github.com/MertYakar66/smart-wheel-engine.git
cd smart-wheel-engine
pip install -r requirements.txt
```

### Basic Usage

```python
from dashboard import QuantDashboard, OptionInput, Position

# Create dashboard
dash = QuantDashboard()

# Price an option
opt = OptionInput(spot=150, strike=145, dte=30, volatility=0.28, option_type='put')
result = dash.price_european(opt)
print(f"Price: ${result['price']:.2f}")
print(f"Delta: {result['delta']:.4f}")

# Full Greeks analysis (including 3rd order)
greeks = dash.analyze_greeks(opt)
print(f"Speed: {greeks['third_order']['speed']:.8f}")
print(f"Color: {greeks['third_order']['color']:.8f}")
```

### Portfolio Risk Analysis

```python
import pandas as pd

# Build portfolio
dash.set_portfolio_value(500_000)
dash.add_position(Position('AAPL', 'put', 170, 45, 0.28, 5, True, 175))
dash.add_position(Position('MSFT', 'put', 400, 45, 0.24, 3, True, 420))

# Set correlation matrix
corr = pd.DataFrame(
    [[1.0, 0.72], [0.72, 1.0]],
    index=['AAPL', 'MSFT'],
    columns=['AAPL', 'MSFT']
)
dash.set_correlation_matrix(corr)

# Calculate VaR
var_result = dash.calculate_var(confidence=0.95)
print(f"95% VaR: ${var_result['var']:,.2f} ({var_result['var_pct']:.2%})")
```

### Interactive Dashboard

```bash
python -m dashboard.quant_dashboard
```

```
╔══════════════════════════════════════════════════════════════╗
║           SMART WHEEL ENGINE - QUANT DASHBOARD               ║
╚══════════════════════════════════════════════════════════════╝

  1. Option Pricing (European)
  2. Option Pricing (American)
  3. Greeks Analysis
  4. Implied Volatility Solver
  5. Portfolio Management
  6. Risk Analysis (VaR/CVaR)
  7. Stress Testing
  8. Position Sizing (Kelly)
  9. Generate Reports
  0. Exit
```

## Project Structure

```
smart-wheel-engine/
├── dashboard/              # Professional trading dashboard
│   ├── __init__.py
│   └── quant_dashboard.py
├── engine/                 # Core quantitative engine
│   ├── option_pricer.py    # BS, American, Greeks
│   ├── risk_manager.py     # VaR, position sizing
│   ├── monte_carlo.py      # Monte Carlo simulations
│   └── ...
├── tests/                  # Comprehensive test suite
│   ├── test_dashboard.py
│   ├── test_advanced_quant.py
│   ├── test_option_pricer.py
│   └── ...
├── utils/                  # Utilities
│   └── security.py         # Security features
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md
│   ├── GOVERNANCE.md
│   ├── GREEKS_UNIT_CONTRACT.md
│   ├── MODEL_CARDS.md
│   └── DATA_SPECIFICATION.md
└── config/                 # Configuration
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_dashboard.py -v
pytest tests/test_advanced_quant.py -v

# With coverage
pytest tests/ --cov=engine --cov=dashboard
```

**Test Coverage**: 170+ tests covering all quantitative functionality.

## Security

See [SECURITY.md](SECURITY.md) for security policy and best practices.

```python
from utils.security import InputValidator, AuditLogger

# Input validation
spot = InputValidator.validate_price(user_input)
symbol = InputValidator.validate_symbol(user_input)

# Audit logging
logger = AuditLogger(log_file="audit.log")
logger.log_trade("BUY", {"symbol": "AAPL", "contracts": 5})
```

## Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit with your settings
vim .env
```

Required environment variables:
- `BROKER_API_KEY`: Your broker API key
- `BROKER_SECRET`: Your broker secret

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture and dataflow |
| [GOVERNANCE.md](docs/GOVERNANCE.md) | Model governance framework |
| [GREEKS_UNIT_CONTRACT.md](docs/GREEKS_UNIT_CONTRACT.md) | Canonical Greeks unit conventions |
| [MODEL_CARDS.md](docs/MODEL_CARDS.md) | Model documentation |
| [DATA_SPECIFICATION.md](docs/DATA_SPECIFICATION.md) | Data architecture and schemas |
| [SECURITY.md](SECURITY.md) | Security policy |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Run tests (`pytest tests/ -v`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing`)
6. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Black-Scholes-Merton model implementation follows Hull (11th Edition)
- American option pricing uses Barone-Adesi & Whaley (1987)
- VaR methodology references Jorion's "Value at Risk"
