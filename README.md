# Polynomial System Solver

A high-performance solver for polynomial systems using Bernstein basis representation and subdivision methods.

## Features

- **Multiple Solving Methods**:
  - **PP (Projected Polyhedron)**: Fast and reliable for 1D and general problems
  - **LP (Linear Programming)**: Tighter bounds for multi-dimensional systems

- **Automatic Parameter Tuning**:
  - Auto-calculated max depth based on Bezout's theorem
  - Adaptive subdivision strategy

- **Robust Implementation**:
  - Handles 1D to n-dimensional polynomial systems
  - Bernstein basis for numerical stability
  - Newton refinement for high accuracy

## Algorithm Overview

1. **Bernstein Representation**: Convert polynomials to Bernstein basis on [0,1]^k domain
2. **Subdivision**: Recursively subdivide parameter space using bounding methods
3. **Bounding Methods**:
   - **PP Method**: Projects control points and computes convex hull intersection
   - **LP Method**: Solves linear programs to find intersection of convex hulls
4. **Refinement**: Apply Newton iteration for high-precision solutions

## Project Structure

```
intersection/
├── src/intersection/
│   ├── bernstein.py              # Bernstein basis conversion
│   ├── polynomial_solver.py      # Main solver interface
│   ├── subdivision_solver.py     # Subdivision algorithm (PP/LP methods)
│   └── geometry.py               # Geometric primitives
├── examples/
│   ├── usage_example.py          # Comprehensive usage guide
│   ├── benchmark_examples.py     # Performance benchmarks
│   └── visualize_*.py            # Visualization tools
├── tests/
│   └── test_*.py                 # Test suite
└── pyproject.toml
```

## Installation

```bash
# Install dependencies using uv
uv sync
```

## Quick Start

### Example 1: Simple 1D Polynomial

```python
import numpy as np
from intersection.polynomial_solver import create_polynomial_system, solve_polynomial_system
from intersection.bernstein import polynomial_nd_to_bernstein

# Solve: x - 0.5 = 0 on [0, 1]
poly = np.array([-0.5, 1.0])
bernstein_coeffs = polynomial_nd_to_bernstein(poly, k=1)

system = create_polynomial_system(
    equation_coeffs=[bernstein_coeffs],
    param_ranges=[(0.0, 1.0)]
)

solutions = solve_polynomial_system(system, method='pp', tolerance=1e-6)
# Result: [{'t': 0.5}]
```

### Example 2: 2D Circle-Ellipse Intersection

```python
# Solve: x^2 + y^2 - 1 = 0 and x^2/4 + 4*y^2 - 1 = 0 on [0,1]×[0,1]
circle_power = np.array([[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
ellipse_power = np.array([[-1.0, 0.0, 4.0], [0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])

circle_bernstein = polynomial_nd_to_bernstein(circle_power, k=2)
ellipse_bernstein = polynomial_nd_to_bernstein(ellipse_power, k=2)

system = create_polynomial_system(
    equation_coeffs=[circle_bernstein, ellipse_bernstein],
    param_ranges=[(0.0, 1.0), (0.0, 1.0)]
)

# Compare methods
solutions_pp = solve_polynomial_system(system, method='pp', tolerance=1e-6)
solutions_lp = solve_polynomial_system(system, method='lp', tolerance=1e-6)
```

## Running Examples

```bash
# Comprehensive usage examples
uv run python examples/usage_example.py

# Performance benchmarks (PP vs LP)
uv run python examples/benchmark_examples.py

# Run tests
uv run pytest
```

## Method Comparison

### PP (Projected Polyhedron) Method
- ✅ Fast per-box computation
- ✅ Reliable for all dimensions
- ✅ Works well for 1D problems
- ⚠️ May process more boxes in multi-dimensional cases

### LP (Linear Programming) Method
- ✅ Tighter bounds for 2D+ systems (up to 2.6x fewer boxes)
- ✅ Excellent for multi-equation systems
- ⚠️ Slower per-box due to LP overhead
- ⚠️ May have numerical issues for high-degree 1D polynomials

**Recommendation**: Use PP for 1D problems, LP for 2D+ multi-equation systems.

## Benchmark Results

From `examples/benchmark_examples.py`:

| Example | Method | Boxes | Depth | Time | Status |
|---------|--------|-------|-------|------|--------|
| 2D Circle-Ellipse | PP | 13 | 1 | 0.095s | ✅ |
| 2D Circle-Ellipse | **LP** | **5** | **0** | **0.021s** | ✅ **2.6x fewer boxes!** |
| 1D Wilkinson (deg 20) | PP | 81 | 5 | 0.019s | ✅ |
| 1D Wilkinson (deg 20) | LP | 1 | 0 | 0.001s | ❌ Numerical issues |

## Documentation

- `examples/README.md` - Usage examples and visualizations
- `tests/README.md` - Test suite documentation
- `examples/benchmark_results.txt` - Latest benchmark results

## Development

```bash
# Run benchmarks
uv run python examples/benchmark_examples.py

# Run specific example
uv run python examples/usage_example.py

# Run tests
uv run pytest
```
