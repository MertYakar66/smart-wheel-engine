# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.x.x   | :white_check_mark: |
| 1.x.x   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

1. **DO NOT** create a public GitHub issue for security vulnerabilities
2. Email security concerns to the repository maintainers
3. Include detailed steps to reproduce the vulnerability
4. Allow reasonable time for a fix before public disclosure

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Security Best Practices

When using Smart Wheel Engine in production:

### Configuration

```bash
# Never commit .env files
cp .env.example .env
chmod 600 .env

# Use environment variables for sensitive data
export BROKER_API_KEY="your-key"
export BROKER_SECRET="your-secret"
```

### API Keys and Secrets

1. **Never hardcode credentials** in source files
2. Use environment variables or secure vaults
3. Rotate API keys regularly
4. Use read-only API keys where possible

### Input Validation

The engine provides built-in input validation:

```python
from utils.security import InputValidator

# All inputs are validated
spot = InputValidator.validate_price(user_input, "spot")
symbol = InputValidator.validate_symbol(user_input)
```

### Audit Logging

Enable audit logging for production:

```python
from utils.security import AuditLogger

logger = AuditLogger(log_file="audit.log")
logger.log_trade("BUY", {"symbol": "AAPL", "contracts": 5})
```

### Rate Limiting

Apply rate limiting to external API calls:

```python
from utils.security import RateLimiter, rate_limited

limiter = RateLimiter(max_requests=100, window_seconds=60)

@rate_limited(limiter)
def call_external_api():
    ...
```

## Security Features

### Input Sanitization
- Price/quantity range validation
- Symbol sanitization (alphanumeric only)
- Option type validation
- String length limits

### Secrets Management
- Secure configuration loading from `.env`
- No secrets in logs (automatic redaction)
- Secure hash functions for sensitive data

### Audit Trail
- All trades logged with timestamps
- Risk events tracked
- Access attempts recorded

### Rate Limiting
- Configurable request limits
- Per-operation rate limiting
- Automatic backoff

## Secure Deployment Checklist

- [ ] All API keys stored in environment variables
- [ ] `.env` file has restrictive permissions (600)
- [ ] Audit logging enabled
- [ ] Rate limiting configured
- [ ] Input validation applied to all user inputs
- [ ] HTTPS enforced for all external connections
- [ ] Regular dependency updates scheduled
- [ ] Security monitoring in place

## Dependencies

We regularly update dependencies to patch security vulnerabilities. Run:

```bash
pip install --upgrade -r requirements.txt
pip audit  # Check for known vulnerabilities
```

## Contact

For security concerns, please contact the repository maintainers directly.
