# Tests

This directory contains test scripts for the intersection computation library.

## Test Files

### Core Functionality Tests

**`test_new_nd_design.py`**
- Tests the n-dimensional framework (Hyperplane, Line, Hypersurface)
- Validates 2D, 3D, and 4D cases
- Run: `uv run python tests/test_new_nd_design.py`

**`test_circle_sphere.py`**
- Tests circle (2D) and sphere (3D) with polynomial degrees 2-10
- Shows interpolation error convergence
- Displays Bernstein basis functions
- Run: `uv run python tests/test_circle_sphere.py`

### Polynomial System Tests

**`test_polynomial_system.py`**
- Tests polynomial system formation for line-hypersurface intersection
- Includes 2D (parabola-line), 3D (paraboloid-line), and circle-line examples
- Validates residual computation
- Run: `uv run python tests/test_polynomial_system.py`

**`test_equation_bernstein_coeffs.py`**
- Verifies equation Bernstein coefficients are computed correctly
- Tests linear combination of hypersurface coefficients
- Validates 2D and 3D cases
- Run: `uv run python tests/test_equation_bernstein_coeffs.py`

**`test_circle_diagonal.py`**
- Unit circle (degree 12) intersecting diagonal line y=x
- Demonstrates high-degree polynomial support
- Generates visualization plot
- Run: `uv run python tests/test_circle_diagonal.py`

## Running Tests

### Run Individual Test
```bash
uv run python tests/test_new_nd_design.py
uv run python tests/test_circle_sphere.py
uv run python tests/test_polynomial_system.py
uv run python tests/test_equation_bernstein_coeffs.py
uv run python tests/test_circle_diagonal.py
```

### Run All Tests
```bash
# Run all test scripts
uv run python tests/test_new_nd_design.py
uv run python tests/test_circle_sphere.py
uv run python tests/test_polynomial_system.py
uv run python tests/test_equation_bernstein_coeffs.py
uv run python tests/test_circle_diagonal.py
```

## Test Organization

These are **script-style tests** that print results to stdout. They are not pytest-style unit tests.

**Advantages**:
- Easy to read and understand
- Show detailed output and intermediate results
- Good for demonstration and debugging
- Can generate visualizations

**Future**: Consider converting to pytest format for automated testing.

## Expected Output

All tests should:
- ✅ Run without errors
- ✅ Show "Test Complete" or similar success message
- ✅ Display residuals ≈ 0 at known intersection points
- ✅ Demonstrate correct Bernstein coefficient computation

## Test Coverage

- ✅ N-dimensional geometry (Hyperplane, Line, Hypersurface)
- ✅ Polynomial interpolation (1D, 2D, 3D)
- ✅ Bernstein basis conversion
- ✅ Polynomial system formation
- ✅ Equation Bernstein coefficients
- ✅ Residual evaluation
- ✅ High-degree polynomials (up to degree 12)
- ⚠️ LP method (not yet implemented)

## Notes

- Tests use the `src.intersection` module
- Some tests generate plots (e.g., `test_circle_diagonal.py`)
- All tests validate against analytical solutions where possible

