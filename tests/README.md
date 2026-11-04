# Glyph Tests

Unit and integration tests for Glyph.

## Running Tests

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=glyph --cov-report=html

# Run specific test file
pytest tests/test_sae.py
```

## Test Structure

- `test_config.py` - Configuration loading/validation
- `test_dataset.py` - Dataset loading
- `test_sae.py` - SAE model
- `test_sae_trainer.py` - SAE training
- `test_linear_probe.py` - Linear probe
- `test_phewas.py` - PheWAS statistics
- `test_integration.py` - End-to-end integration tests
- `test_glyph.py` - Main Glyph class

Tests will be implemented in Phase 9.
