# Examples

This directory contains example scripts demonstrating how to use the intersection computation library.

## Example Files

### Basic Examples

**`example_2d.py`**
- Basic 2D line-curve intersection example
- Shows how to create lines and curves
- Demonstrates system formation and evaluation
- Run: `uv run python examples/example_2d.py`

**`example_3d.py`**
- Basic 3D line-surface intersection example
- Shows how to create lines and surfaces
- Demonstrates system formation and evaluation
- Run: `uv run python examples/example_3d.py`

### Advanced Examples

**`example_get_bernstein_coeffs.py`**
- Comprehensive guide to accessing Bernstein coefficients
- Shows 3 different methods to get coefficients
- Demonstrates 2D (curve) and 3D (surface) cases
- Explains coefficient structure and shapes
- Run: `uv run python examples/example_get_bernstein_coeffs.py`

## Quick Start

### Example 1: 2D Circle Intersecting Line

```python
import numpy as np
from src.intersection.geometry import Hyperplane, Line, Hypersurface
from src.intersection.polynomial_system import create_intersection_system, evaluate_system

# Create unit circle
circle = Hypersurface(
    func=lambda u: np.array([np.cos(2*np.pi*u), np.sin(2*np.pi*u)]),
    param_ranges=[(0, 1)],
    ambient_dim=2,
    degree=5
)

# Create line y = x
h1 = Hyperplane(coeffs=[1, -1], d=0)
line = Line([h1])

# Create intersection system
system = create_intersection_system(line, circle)

# Evaluate at a point
residual = evaluate_system(system, 0.125)  # Should be ≈ 0 at intersection
print(f"Residual: {residual}")
```

### Example 2: 3D Surface Intersecting Line

```python
import numpy as np
from src.intersection.geometry import Hyperplane, Line, Hypersurface
from src.intersection.polynomial_system import create_intersection_system, evaluate_system

# Create paraboloid z = x^2 + y^2
surface = Hypersurface(
    func=lambda u, v: np.array([u, v, u**2 + v**2]),
    param_ranges=[(0, 1), (0, 1)],
    ambient_dim=3,
    degree=4
)

# Create line x=0.5, y=0.5
h1 = Hyperplane(coeffs=[1, 0, 0], d=-0.5)
h2 = Hyperplane(coeffs=[0, 1, 0], d=-0.5)
line = Line([h1, h2])

# Create intersection system
system = create_intersection_system(line, surface)

# Evaluate at a point
residuals = evaluate_system(system, 0.5, 0.5)  # Should be ≈ [0, 0] at intersection
print(f"Residuals: {residuals}")
```

### Example 3: Get Bernstein Coefficients

```python
from src.intersection.polynomial_system import (
    create_intersection_system,
    get_equation_bernstein_coeffs
)

# Create system (as above)
system = create_intersection_system(line, hypersurface)

# Get equation Bernstein coefficients
eq_coeffs = get_equation_bernstein_coeffs(system)

# For 2D: eq_coeffs[0] is the single equation
# For 3D: eq_coeffs[0] and eq_coeffs[1] are the two equations

print(f"Equation Bernstein coefficients: {eq_coeffs}")
```

## Running Examples

```bash
# Run basic 2D example
uv run python examples/example_2d.py

# Run basic 3D example
uv run python examples/example_3d.py

# Run Bernstein coefficients example
uv run python examples/example_get_bernstein_coeffs.py
```

## What You'll Learn

- ✅ How to create hyperplanes, lines, and hypersurfaces
- ✅ How to form intersection systems
- ✅ How to evaluate residuals at parameter values
- ✅ How to access Bernstein coefficients
- ✅ Understanding coefficient structure and shapes
- ✅ Working with different dimensions (2D, 3D, nD)

## Next Steps

After running these examples, check out:
- **Documentation**: `../README.md` for project overview
- **Tests**: `../tests/` for more complex examples
- **Design**: `../N_DIMENSIONAL_DESIGN.md` for mathematical details
- **Bernstein Coefficients**: `../EQUATION_BERNSTEIN_COEFFICIENTS.md` for detailed explanation

## Notes

- All examples use the `src.intersection` module
- Examples are self-contained and can be run independently
- Output is printed to stdout for easy understanding

