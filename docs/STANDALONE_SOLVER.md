# Standalone Polynomial System Solver

## Overview

The standalone polynomial system solver provides a clean, simple API for solving systems of polynomial equations using the PP (Projected Polyhedron) method with subdivision and Newton refinement.

**Key Features:**
- ✅ **Standalone** - No need to construct line-hypersurface intersection systems
- ✅ **Arbitrary polynomial systems** - Works with any polynomial system in Bernstein basis
- ✅ **Custom parameter domains** - Support for arbitrary parameter ranges
- ✅ **Automatic Newton refinement** - Optional refinement for high accuracy
- ✅ **Clean API** - Simple, intuitive interface

## Quick Start

```python
import numpy as np
from src.intersection.polynomial_solver import (
    create_polynomial_system,
    solve_polynomial_system
)
from src.intersection.bernstein import polynomial_nd_to_bernstein

# Define polynomial: f(t) = t - 0.5 = 0
power_coeffs = np.array([-0.5, 1.0])
bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)

# Create system
system = create_polynomial_system(
    equation_coeffs=[bern_coeffs],
    param_ranges=[(0.0, 1.0)]
)

# Solve
solutions = solve_polynomial_system(system)
# Returns: [{'t': 0.5}]
```

## API Reference

### `create_polynomial_system()`

Create a polynomial system from Bernstein coefficients.

```python
def create_polynomial_system(
    equation_coeffs: List[np.ndarray],
    param_ranges: List[Tuple[float, float]],
    param_names: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> PolynomialSystem
```

**Parameters:**
- `equation_coeffs`: Bernstein coefficients for each equation
  - For 1D (k=1): each array has shape `(degree+1,)`
  - For 2D (k=2): each array has shape `(degree+1, degree+1)`
  - For kD: each array has shape `(degree+1,)^k`
- `param_ranges`: Parameter ranges `[(min1, max1), (min2, max2), ...]`
- `param_names`: Optional parameter names (default: `['t', 'u', 'v', 'w', 's']`)
- `metadata`: Optional metadata dictionary

**Returns:**
- `PolynomialSystem` object

### `solve_polynomial_system()`

Solve a polynomial system.

```python
def solve_polynomial_system(
    system: PolynomialSystem,
    method: str = 'pp',
    tolerance: float = 1e-6,
    crit: float = 0.8,
    max_depth: int = 30,
    refine: bool = True,
    verbose: bool = False
) -> List[Dict[str, float]]
```

**Parameters:**
- `system`: The polynomial system to solve
- `method`: Solving method - `'pp'`, `'lp'`, or `'hybrid'` (default: `'pp'`)
- `tolerance`: Size threshold for claiming a root (default: `1e-6`)
- `crit`: Critical ratio for subdivision (default: `0.8`)
- `max_depth`: Maximum subdivision depth (default: `30`)
- `refine`: Whether to refine solutions using Newton iteration (default: `True`)
- `verbose`: Print progress (default: `False`)

**Returns:**
- List of solutions, each as a dictionary mapping parameter names to values

### `PolynomialSystem` Class

Dataclass representing a polynomial system.

```python
@dataclass
class PolynomialSystem:
    equation_coeffs: List[np.ndarray]  # Bernstein coefficients
    param_ranges: List[Tuple[float, float]]  # Parameter ranges
    k: int  # Number of parameters
    degree: int  # Polynomial degree
    param_names: Optional[List[str]] = None  # Parameter names
    metadata: Optional[Dict[str, Any]] = None  # Metadata
```

## Examples

### Example 1: 1D Linear Equation

```python
# Solve: f(t) = t - 0.7 = 0
power_coeffs = np.array([-0.7, 1.0])
bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)

system = create_polynomial_system(
    equation_coeffs=[bern_coeffs],
    param_ranges=[(0.0, 1.0)]
)

solutions = solve_polynomial_system(system)
# Returns: [{'t': 0.7}]
```

### Example 2: 1D Quadratic with Two Roots

```python
# Solve: f(t) = (t - 0.3)(t - 0.7) = t^2 - t + 0.21 = 0
power_coeffs = np.array([0.21, -1.0, 1.0])
bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)

system = create_polynomial_system(
    equation_coeffs=[bern_coeffs],
    param_ranges=[(0.0, 1.0)]
)

solutions = solve_polynomial_system(system)
# Returns: [{'t': 0.3}, {'t': 0.7}]
```

### Example 3: 2D System

```python
# Solve system:
#   f1(u,v) = u - 0.3 = 0
#   f2(u,v) = v - 0.7 = 0

# Equation 1: f1(u,v) = u - 0.3
power_coeffs_1 = np.array([[-0.3, 0.0], [1.0, 0.0]])
bern_coeffs_1 = polynomial_nd_to_bernstein(power_coeffs_1, k=2)

# Equation 2: f2(u,v) = v - 0.7
power_coeffs_2 = np.array([[-0.7, 1.0], [0.0, 0.0]])
bern_coeffs_2 = polynomial_nd_to_bernstein(power_coeffs_2, k=2)

system = create_polynomial_system(
    equation_coeffs=[bern_coeffs_1, bern_coeffs_2],
    param_ranges=[(0.0, 1.0), (0.0, 1.0)],
    param_names=['u', 'v']
)

solutions = solve_polynomial_system(system)
# Returns: [{'u': 0.3, 'v': 0.7}]
```

### Example 4: Custom Parameter Domain

```python
# Solve: f(x) = x - 5.0 = 0 for x in [2, 8]

# For domain [2, 8], normalize to [0, 1]:
# x = 2 + 6*s where s in [0, 1]
# f(x) = x - 5 = (2 + 6*s) - 5 = 6*s - 3
power_coeffs = np.array([-3.0, 6.0])
bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)

system = create_polynomial_system(
    equation_coeffs=[bern_coeffs],
    param_ranges=[(2.0, 8.0)],
    param_names=['x']
)

solutions = solve_polynomial_system(system)
# Returns: [{'x': 5.0}]
```

### Example 5: Using PolynomialSystem Class Directly

```python
# Create system using class
power_coeffs = np.array([-0.5, 0.0, 1.0])
bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)

system = PolynomialSystem(
    equation_coeffs=[bern_coeffs],
    param_ranges=[(0.0, 1.0)],
    k=1,
    degree=2,
    param_names=['t'],
    metadata={'description': 'Quadratic equation'}
)

solutions = solve_polynomial_system(system)
# Returns: [{'t': 0.707...}]
```

## Workflow

The solver implements a complete 4-step workflow:

### 1. Normalization to [0, 1]^k

The Bernstein coefficients are already in the correct form for the normalized domain [0, 1]^k. The `param_ranges` specify how to map back to the original domain.

### 2. Subdivision with PP Method

Uses the PP (Projected Polyhedron) method to find all possible roots:
- For each dimension j, extract univariate polynomials from all equations
- Compute convex hull of control points and intersect with x-axis
- Intersect all ranges to get tightest bound for dimension j
- Subdivide recursively until boxes are smaller than tolerance

### 3. Newton Refinement (Optional)

Refines each solution using Newton iteration:
- Multi-dimensional Newton: x_{n+1} = x_n - J(x_n)^{-1} * F(x_n)
- Jacobian computed numerically using finite differences
- Iterates until residual norm < tolerance

### 4. Denormalization to Original Domain

Converts solutions from normalized [0, 1]^k space back to original parameter ranges:
- Linear interpolation: x_orig = min + (max - min) × x_norm
- Returns dictionary with parameter names

## Important Notes

### Bernstein Coefficients and Parameter Domains

When working with custom parameter domains, the Bernstein coefficients should be expressed in terms of the **normalized** parameter in [0, 1].

**Example:**
- Original domain: x ∈ [2, 8]
- Normalized: s ∈ [0, 1] where x = 2 + 6s
- Equation: f(x) = x - 5 becomes f(s) = (2 + 6s) - 5 = 6s - 3
- Bernstein coefficients are for f(s), not f(x)

### Power to Bernstein Conversion

Use `polynomial_nd_to_bernstein()` to convert from power basis to Bernstein basis:

```python
from src.intersection.bernstein import polynomial_nd_to_bernstein

# 1D: f(t) = a0 + a1*t + a2*t^2
power_coeffs = np.array([a0, a1, a2])
bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)

# 2D: f(u,v) = a00 + a10*u + a01*v + a11*u*v
power_coeffs = np.array([[a00, a01], [a10, a11]])
bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=2)
```

## Testing

Comprehensive tests are available in `tests/test_polynomial_solver.py`:

```bash
uv run python -m pytest tests/test_polynomial_solver.py -v
```

**Test Coverage:**
- ✅ 1D linear equation
- ✅ 1D quadratic with two roots
- ✅ Custom parameter domain
- ✅ 2D system
- ✅ System with no solutions
- ✅ PolynomialSystem class usage

## Performance

The PP method provides significant performance improvements:

| Test Case | Simple Method | Enhanced PP | Improvement |
|-----------|---------------|-------------|-------------|
| 1D: f(t) = (t - 0.3)(t - 0.7) | 1.000000 | 0.580000 | 42.0% |
| 1D: f(t) = (t - 0.5)^3 | 1.000000 | 0.666667 | 33.3% |
| 2D: f(u,v) = (u - 0.5)(v - 0.5) | 1.000000 | 0.000000 | 100.0% |

## Comparison with Line-Hypersurface Solver

### Standalone Solver (New)

```python
# Direct polynomial system
power_coeffs = np.array([-0.5, 1.0])
bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)

system = create_polynomial_system(
    equation_coeffs=[bern_coeffs],
    param_ranges=[(0.0, 1.0)]
)

solutions = solve_polynomial_system(system)
```

**Advantages:**
- ✅ Simple, direct API
- ✅ No geometric setup required
- ✅ Works with any polynomial system
- ✅ Clear parameter domain specification

### Line-Hypersurface Solver (Original)

```python
# Requires geometric setup
hypersurface = Hypersurface(
    func=lambda t: np.array([t, t**2 - 0.5]),
    param_ranges=[(0.0, 1.0)],
    ambient_dim=2,
    degree=2
)

line = Line(hyperplanes=[Hyperplane(coeffs=[0.0, 1.0], d=0.0)])

system = create_intersection_system(line, hypersurface)
solutions = solve_polynomial_system(system, method='pp')
```

**Use Cases:**
- Geometric intersection problems
- Line-hypersurface intersections
- When you already have geometric objects

## See Also

- `examples/example_standalone_solver.py` - Complete examples
- `tests/test_polynomial_solver.py` - Test suite
- `src/intersection/polynomial_solver.py` - Implementation
- `docs/PP_METHOD.md` - PP method documentation

