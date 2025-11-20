# CI/CD Documentation

This document describes the Continuous Integration and Continuous Deployment (CI/CD) setup for this repository.

## Overview

The repository uses GitHub Actions for automated testing, code quality checks, and continuous integration. The CI/CD pipeline ensures code quality and correctness before merging changes.

## Workflows

### 1. CI Workflow (`.github/workflows/ci.yml`)

**Triggers:**
- Push to `main`, `master`, or `develop` branches
- Pull requests targeting `main`, `master`, or `develop` branches

**Jobs:**

#### Code Quality Checks
- **Black**: Ensures code formatting follows the Black style guide (line length: 120)
- **isort**: Validates import statement ordering
- **flake8**: Performs linting with configured ignore rules
- **pylint**: Additional linting checks (non-blocking)

#### Test Suite
- Runs on Python versions: 3.8, 3.9, 3.10, 3.11
- Executes all tests in the `tests/` directory using pytest
- Generates code coverage reports
- Uploads coverage to Codecov (Python 3.10 only)

#### Integration Tests
- Runs tests marked with `@pytest.mark.integration`
- Non-blocking (continues even if integration tests fail)

### 2. Pre-commit Workflow (`.github/workflows/pre-commit.yml`)

**Triggers:**
- Push to `main`, `master`, or `develop` branches
- Pull requests targeting `main`, `master`, or `develop` branches

**Jobs:**
- Runs all pre-commit hooks defined in `.pre-commit-config.yaml`
- Skips `pytest-check` and `pylint` hooks (covered by main CI workflow)
- Validates code formatting, linting, and other code quality checks
- Uses cached pre-commit environments for faster execution
- Timeout: 15 minutes

### 3. Quick Test Workflow (`.github/workflows/quick-test.yml`)

**Triggers:**
- Push to any branch
- Pull requests to any branch

**Jobs:**
- Fast test execution on Python 3.10 only
- No code quality checks (only tests)
- Uses pip caching for faster execution
- Fails fast on first test failure (`-x` flag)
- Timeout: 30 minutes
- Ideal for quick feedback during development

## Status Badges

The README includes status badges showing:
- CI workflow status
- Pre-commit checks status
- Code style compliance (Black)

## Local Development

### Running Tests Locally

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run only fast tests (exclude slow tests)
pytest -m "not slow"

# Run only integration tests
pytest -m integration
```

### Running Code Quality Checks Locally

```bash
# Format code with black
black --line-length=120 .

# Sort imports
isort --profile black --line-length 120 .

# Run flake8
flake8 . --max-line-length=120

# Run pylint
pylint *.py --rcfile=.pylintrc

# Run all pre-commit hooks
pre-commit run --all-files
```

### Installing Pre-commit Hooks

To automatically run checks before each commit:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

## Caching Strategy

The workflows use caching to speed up CI runs:

- **pip cache**: Caches Python package installations
- **pre-commit cache**: Caches pre-commit hook environments

Cache keys are based on:
- Operating system
- Python version
- Requirements file hashes
- Pre-commit config hash

## Coverage Reporting

Test coverage is automatically calculated and can be uploaded to Codecov. To enable Codecov integration:

1. Sign up at [codecov.io](https://codecov.io)
2. Add the repository to Codecov
3. No token is required for public repositories

## Debugging CI Failures

### Code Quality Failures

If code quality checks fail:
1. Run the failing check locally (see commands above)
2. Fix the issues or auto-format: `black . && isort .`
3. Commit and push the fixes

### Test Failures

If tests fail:
1. Run the specific failing test locally: `pytest tests/test_module.py::test_function -v`
2. Fix the issue
3. Verify all tests pass: `pytest`
4. Commit and push

### Pre-commit Hook Failures

If pre-commit hooks fail:
1. Run `pre-commit run --all-files` locally
2. Review and fix the reported issues
3. Commit and push

## Extending the CI/CD Pipeline

### Adding New Tests

1. Create test files in `tests/` directory following the pattern `test_*.py`
2. Use pytest conventions for test functions: `def test_something():`
3. Mark slow tests: `@pytest.mark.slow`
4. Mark integration tests: `@pytest.mark.integration`

### Adding New Workflows

Create new workflow files in `.github/workflows/` directory:

```yaml
name: My Workflow
on:
  push:
    branches: [main]
jobs:
  my-job:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run my task
        run: echo "Hello"
```

### Modifying Python Versions

To test on different Python versions, edit the matrix in `.github/workflows/ci.yml`:

```yaml
strategy:
  matrix:
    python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
```

## Best Practices

1. **Always run tests locally** before pushing
2. **Install pre-commit hooks** to catch issues early
3. **Write tests for new features** to maintain coverage
4. **Keep dependencies updated** in `requirements.txt` and `requirements-dev.txt`
5. **Use descriptive commit messages** for easier debugging
6. **Review CI failures** before requesting code review

## Troubleshooting

### Common Issues

**Issue**: "Module not found" errors in CI
- **Solution**: Ensure all dependencies are listed in `requirements.txt`

**Issue**: Tests pass locally but fail in CI
- **Solution**: Check Python version compatibility, ensure no local-only dependencies

**Issue**: Pre-commit hooks are slow
- **Solution**: Caching is enabled; subsequent runs should be faster

**Issue**: Pre-commit workflow fails with pytest errors
- **Solution**: The pytest-check hook is now skipped in CI (it's covered by the main test workflow)

**Issue**: Workflow times out
- **Solution**: All workflows now have timeout settings (15-30 minutes). If legitimate work exceeds this, increase timeout in workflow file

**Issue**: Code quality checks fail but you want to proceed
- **Solution**: Code quality checks are non-blocking (continue-on-error: true). Tests must still pass.

**Issue**: Codecov upload fails
- **Solution**: This is non-blocking and won't fail the build; check Codecov dashboard for details

### Workflow Timeout Settings

All workflows have timeout protection:
- Quick Test: 30 minutes
- Code Quality: 20 minutes
- Test Suite: 30 minutes per Python version
- Integration Tests: 20 minutes
- Pre-commit: 15 minutes

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/)
- [pre-commit Documentation](https://pre-commit.com/)
- [Black Code Style](https://black.readthedocs.io/)
- [Codecov Documentation](https://docs.codecov.com/)
