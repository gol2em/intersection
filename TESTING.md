# Testing Guide

## Quick Start

### Install pytest (if not already installed)
```bash
.venv\Scripts\pip.exe install pytest pytest-cov
```

### Run all tests
```bash
.venv\Scripts\pytest.exe tests/
```

Expected output:
```
60 passed, 1 skipped in 1.51s
```

## Common Test Commands

### Basic Testing

```bash
# Run all tests
.venv\Scripts\pytest.exe tests/

# Run with verbose output (shows each test name)
.venv\Scripts\pytest.exe tests/ -v

# Run with very verbose output (shows test names and results)
.venv\Scripts\pytest.exe tests/ -vv

# Run with short traceback (easier to read failures)
.venv\Scripts\pytest.exe tests/ --tb=short

# Run with no output capture (see print statements)
.venv\Scripts\pytest.exe tests/ -s
```

### Running Specific Tests

```bash
# Run specific test file
.venv\Scripts\pytest.exe tests/test_geometry.py

# Run specific test class
.venv\Scripts\pytest.exe tests/test_geometry.py::TestLine2D

# Run specific test method
.venv\Scripts\pytest.exe tests/test_geometry.py::TestLine2D::test_creation_direct

# Run tests matching a pattern
.venv\Scripts\pytest.exe tests/ -k "line"  # Runs all tests with "line" in name
.venv\Scripts\pytest.exe tests/ -k "Line2D"  # Runs all Line2D tests
```

### Coverage Reports

```bash
# Run with coverage report
.venv\Scripts\pytest.exe tests/ --cov=src/intersection

# Run with detailed coverage (shows missing lines)
.venv\Scripts\pytest.exe tests/ --cov=src/intersection --cov-report=term-missing

# Generate HTML coverage report
.venv\Scripts\pytest.exe tests/ --cov=src/intersection --cov-report=html
# Then open htmlcov/index.html in browser

# Generate XML coverage report (for CI/CD)
.venv\Scripts\pytest.exe tests/ --cov=src/intersection --cov-report=xml
```

### Test Selection

```bash
# Run only failed tests from last run
.venv\Scripts\pytest.exe tests/ --lf

# Run failed tests first, then others
.venv\Scripts\pytest.exe tests/ --ff

# Stop after first failure
.venv\Scripts\pytest.exe tests/ -x

# Stop after N failures
.venv\Scripts\pytest.exe tests/ --maxfail=3
```

### Output Control

```bash
# Quiet mode (minimal output)
.venv\Scripts\pytest.exe tests/ -q

# Show local variables in tracebacks
.venv\Scripts\pytest.exe tests/ -l

# Show summary of all test outcomes
.venv\Scripts\pytest.exe tests/ -ra
```

## Test Organization

### Test Files

```
tests/
├── __init__.py              # Makes tests a package
├── conftest.py              # Pytest configuration
├── test_geometry.py         # Tests for geometry classes (30 tests)
├── test_interpolation.py    # Tests for interpolation (11 tests)
└── test_bernstein.py        # Tests for Bernstein conversion (19 tests)
```

### Test Classes

Each test file contains multiple test classes:

**test_geometry.py:**
- `TestLine2D` - Tests for 2D line class
- `TestLine3D` - Tests for 3D line class
- `TestCurve` - Tests for curve class
- `TestSurface` - Tests for surface class

**test_interpolation.py:**
- `TestInterpolateCurve` - Tests for curve interpolation
- `TestInterpolateSurface` - Tests for surface interpolation
- `TestInterpolationAccuracy` - Tests for interpolation accuracy

**test_bernstein.py:**
- `TestPolynomialToBernstein` - Tests for power to Bernstein conversion
- `TestBernsteinToPolynomial` - Tests for Bernstein to power conversion
- `TestEvaluateBernstein` - Tests for Bernstein evaluation
- `TestPolynomial2DToBernstein` - Tests for 2D conversion
- `TestBernsteinProperties` - Tests for mathematical properties
- `TestBernsteinIntegration` - Integration tests

## Writing New Tests

### Basic Test Structure

```python
import pytest
import numpy as np
from src.intersection.geometry import Curve

class TestNewFeature:
    """Tests for new feature"""
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        # Arrange
        curve = Curve(
            x_func=lambda u: u,
            y_func=lambda u: u**2,
            u_range=(0, 1),
            degree=3
        )
        
        # Act
        result = curve.evaluate(0.5)
        
        # Assert
        assert np.allclose(result, [0.5, 0.25])
```

### Testing Exceptions

```python
def test_invalid_input(self):
    """Test that invalid input raises error"""
    with pytest.raises(ValueError, match="error message pattern"):
        Line2D(a=0, b=0, c=1)
```

### Testing Output

```python
def test_verbose_output(self, capsys):
    """Test that verbose mode produces output"""
    curve = Curve(..., verbose=True)
    captured = capsys.readouterr()
    assert "expected text" in captured.out
```

### Parametrized Tests

```python
@pytest.mark.parametrize("degree,expected", [
    (1, 2),
    (2, 3),
    (3, 4),
])
def test_polynomial_degree(self, degree, expected):
    """Test polynomial degree"""
    curve = Curve(..., degree=degree)
    assert len(curve.poly_x.coef) == expected
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: windows-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.13'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest pytest-cov
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src/intersection --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Test Coverage Goals

Current coverage:
- ✅ `geometry.py`: 96%
- ✅ `interpolation.py`: 100%
- ✅ `bernstein.py`: 80%

Goals:
- Core modules: >90% ✅
- All modules: >80%
- Critical paths: 100%

## Debugging Failed Tests

### 1. Run with verbose output
```bash
.venv\Scripts\pytest.exe tests/ -vv --tb=long
```

### 2. Run specific failing test
```bash
.venv\Scripts\pytest.exe tests/test_geometry.py::TestLine2D::test_creation_direct -vv
```

### 3. Add print statements (run with -s)
```python
def test_something(self):
    result = some_function()
    print(f"Result: {result}")  # Will be visible with -s flag
    assert result == expected
```

```bash
.venv\Scripts\pytest.exe tests/ -s
```

### 4. Use pytest's built-in debugger
```bash
.venv\Scripts\pytest.exe tests/ --pdb  # Drop into debugger on failure
```

## Best Practices

### 1. Test Naming
- Use descriptive names: `test_creation_from_two_points`
- Follow pattern: `test_<what_is_being_tested>`
- Add docstrings to explain complex tests

### 2. Test Independence
- Each test should be independent
- Don't rely on test execution order
- Clean up after tests if needed

### 3. Assertions
- Use appropriate tolerances for floating point: `np.isclose(..., atol=1e-10)`
- Test both success and failure cases
- Include edge cases

### 4. Test Organization
- Group related tests in classes
- One test file per module
- Keep tests close to the code they test

### 5. Documentation
- Add docstrings to test classes and methods
- Explain why a test is skipped
- Document expected behavior

## Common Issues

### Issue: Tests not found
**Solution**: Make sure `conftest.py` exists and adds `src` to path

### Issue: Import errors
**Solution**: Check that `__init__.py` exists in `tests/` directory

### Issue: Numerical precision failures
**Solution**: Use appropriate tolerance in `np.isclose()` or `np.allclose()`

### Issue: Tests pass locally but fail in CI
**Solution**: Check for platform-specific issues, random seeds, or file paths

## Performance

### Running Tests Faster

```bash
# Run tests in parallel (requires pytest-xdist)
pip install pytest-xdist
.venv\Scripts\pytest.exe tests/ -n auto

# Run only fast tests (mark slow tests with @pytest.mark.slow)
.venv\Scripts\pytest.exe tests/ -m "not slow"
```

### Profiling Tests

```bash
# Show slowest tests
.venv\Scripts\pytest.exe tests/ --durations=10
```

## Summary

- ✅ 60 tests covering all core functionality
- ✅ Easy to run: `.venv\Scripts\pytest.exe tests/`
- ✅ Great coverage: 96-100% on core modules
- ✅ Well-organized and maintainable
- ✅ Ready for CI/CD integration

For more information, see:
- [Pytest Documentation](https://docs.pytest.org/)
- [PYTEST_TESTS_SUMMARY.md](PYTEST_TESTS_SUMMARY.md) - Detailed test summary

