# Testing and Code Quality

This document describes the testing infrastructure and code quality tools set up for this project.

## Overview

The project uses:
- **pytest** for testing
- **black** for code formatting
- **pylint** for linting
- **isort** for import sorting
- **flake8** for style checking
- **mypy** for type checking
- **pre-commit** for automated checks before commits

## Installation

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

## Running Tests

### Run all tests:
```bash
pytest
```

### Run with verbose output:
```bash
pytest -v
```

### Run specific test file:
```bash
pytest tests/test_main_estimation.py
```

### Run specific test class:
```bash
pytest tests/test_ica.py::TestGenerateICAData
```

### Run specific test:
```bash
pytest tests/test_ica.py::TestGenerateICAData::test_generate_ica_data_output_shapes
```

### Run with coverage:
```bash
pytest --cov=. --cov-report=html
```

### Run tests in parallel (faster):
```bash
pytest -n auto
```

## Code Formatting

### Format all Python files with black:
```bash
black .
```

### Format specific file:
```bash
black main_estimation.py
```

### Check formatting without making changes:
```bash
black --check .
```

## Linting

### Run pylint on all Python files:
```bash
pylint *.py
```

### Run pylint on specific file:
```bash
pylint main_estimation.py
```

### Run flake8:
```bash
flake8 .
```

## Import Sorting

### Sort imports with isort:
```bash
isort .
```

### Check imports without making changes:
```bash
isort --check-only .
```

## Type Checking

### Run mypy:
```bash
mypy *.py
```

## Pre-commit Hooks

Pre-commit hooks automatically run checks before each commit.

### Install hooks (already done):
```bash
pre-commit install
```

### Run hooks manually on all files:
```bash
pre-commit run --all-files
```

### Run specific hook:
```bash
pre-commit run black --all-files
```

### Skip hooks for a commit (not recommended):
```bash
git commit --no-verify
```

## Test Structure

Tests are organized in the `tests/` directory:

- `tests/test_main_estimation.py` - Tests for main estimation methods
- `tests/test_ica.py` - Tests for ICA data generation and estimation
- `tests/test_mcc.py` - Tests for disentanglement metrics
- `tests/test_plot_utils.py` - Tests for plotting utilities

## Writing Tests

Tests use pytest fixtures for setup. Example:

```python
import pytest
import numpy as np

class TestMyFunction:
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        return np.random.randn(100, 10)

    def test_function_output_shape(self, sample_data):
        """Test that function returns correct shape."""
        result = my_function(sample_data)
        assert result.shape == (100,)
```

## Configuration Files

- `pytest.ini` - Pytest configuration
- `.pylintrc` - Pylint configuration
- `pyproject.toml` - Black and isort configuration
- `.pre-commit-config.yaml` - Pre-commit hooks configuration

## CI/CD Integration

The pre-commit hooks can be integrated with CI/CD pipelines. Add this to your CI configuration:

```yaml
- name: Run pre-commit hooks
  run: pre-commit run --all-files
```

## Code Coverage

Generate a coverage report:

```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

## Troubleshooting

### Tests fail with import errors:
Use `python -m pytest` instead of `pytest`:
```bash
python -m pytest tests/
```

### Pre-commit hooks are slow:
Skip certain hooks during development:
```bash
SKIP=pylint,mypy git commit -m "message"
```

### Black and isort conflict:
Both are configured to work together using the "black" profile in isort.

### Pylint is too strict:
The `.pylintrc` file has been configured to be reasonable for scientific computing. You can adjust disabled warnings there.
