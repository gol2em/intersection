# Intersection Computation Framework - Project Summary

## Overview

A complete Python framework for computing intersections between straight lines and parametric curves (2D) or surfaces (3D) using polynomial interpolation and Bernstein basis representation.

## Project Structure

```
intersection/
├── src/intersection/          # Core library modules
│   ├── __init__.py           # Main API and high-level functions
│   ├── geometry.py           # Geometric primitives (Line, Curve, Surface)
│   ├── interpolation.py      # Step 1: Polynomial interpolation
│   ├── bernstein.py          # Step 2: Bernstein basis conversion
│   ├── polynomial_system.py  # Step 3: System formation
│   ├── solver.py             # Step 4: Polynomial solver
│   └── utils.py              # Visualization and utilities
├── examples/                  # Example scripts
│   ├── example_2d.py         # 2D intersection examples
│   └── example_3d.py         # 3D intersection examples
├── tests/                     # Test suite
│   └── test_basic.py         # Basic functionality tests
├── pyproject.toml            # Project configuration
├── README.md                 # Project documentation
├── QUICKSTART.md             # Quick start guide
├── PYCHARM_SETUP.md          # PyCharm configuration guide
└── .gitignore                # Git ignore rules
```

## Algorithm Pipeline

The framework implements a 5-step algorithm:

### Step 1: Interpolation
- **Module**: `interpolation.py`
- **Purpose**: Convert parametric functions to polynomial form
- **Method**: Chebyshev node sampling + least squares fitting
- **Output**: Polynomial coefficients in power basis

### Step 2: Bernstein Basis Conversion
- **Module**: `bernstein.py`
- **Purpose**: Convert polynomials to Bernstein basis for numerical stability
- **Method**: Basis transformation using binomial coefficients
- **Output**: Bernstein coefficients

### Step 3: Polynomial System Formation
- **Module**: `polynomial_system.py`
- **Purpose**: Formulate intersection as polynomial equations
- **Method**: 
  - 2D: Eliminate line parameter to get 1 equation in t
  - 3D: Eliminate line parameter to get 2 equations in (u, v)
- **Output**: Polynomial system representation

### Step 4: Solver
- **Module**: `solver.py`
- **Purpose**: Solve polynomial system for parameters
- **Method**: 
  - 2D: Root finding + Newton refinement
  - 3D: Numerical solver (fsolve) with multiple initial guesses
- **Output**: Parameter values (t for 2D, (u,v) for 3D)

### Step 5: Point Conversion
- **Module**: `__init__.py` (in main functions)
- **Purpose**: Convert parameters to actual intersection points
- **Method**: Evaluate parametric functions at solution parameters
- **Output**: Intersection points with validation metrics

## Key Features

### 1. Modular Design
- Each step is independent and can be tested separately
- Clear separation of concerns
- Easy to extend or modify individual components

### 2. Verbose Output
- Every step can print detailed information with `verbose=True`
- Helps validate algorithm correctness
- Useful for debugging and understanding

### 3. Visualization
- 2D plotting with matplotlib
- 3D surface visualization
- Automatic annotation of intersection points

### 4. Numerical Stability
- Chebyshev nodes for interpolation
- Bernstein basis for polynomial representation
- Newton refinement for accuracy

### 5. Flexible API
- High-level functions: `compute_intersections_2d/3d()`
- Low-level access to each step
- Support for custom parametric functions

## Core Classes

### Geometric Primitives

#### `Line2D` / `Line3D`
- Represents straight lines
- Methods: `evaluate(s)`, `distance_to_point(point)`

#### `ParametricCurve`
- Represents 2D parametric curves
- Constructor: `ParametricCurve(x_func, y_func, t_range)`
- Methods: `evaluate(t)`, `sample(n_points)`

#### `ParametricSurface`
- Represents 3D parametric surfaces
- Constructor: `ParametricSurface(x_func, y_func, z_func, u_range, v_range)`
- Methods: `evaluate(u, v)`, `sample(n_u, n_v)`

## Main Functions

### High-Level API

```python
compute_intersections_2d(line, curve, degree=5, verbose=False)
compute_intersections_3d(line, surface, degree=5, verbose=False)
```

### Step-by-Step API

```python
# Step 1
poly_x, poly_y = interpolate_curve(curve, degree, verbose)
poly_x, poly_y, poly_z = interpolate_surface(surface, degree, verbose)

# Step 2
bern_x = polynomial_to_bernstein(poly_x, verbose)

# Step 3
system = create_intersection_system_2d(line, bern_x, bern_y, verbose)
system = create_intersection_system_3d(line, bern_x, bern_y, bern_z, verbose)

# Step 4
solutions = solve_polynomial_system(system, verbose)

# Step 5 (manual)
for sol in solutions:
    point = curve.evaluate(sol['t'])  # or surface.evaluate(sol['u'], sol['v'])
```

### Visualization

```python
visualize_2d(line, curve, intersections, title)
visualize_3d(line, surface, intersections, title)
print_intersection_summary(intersections, dimension)
```

## Dependencies

- **numpy**: Numerical computations and arrays
- **scipy**: Polynomial fitting and numerical solvers
- **matplotlib**: Visualization and plotting

## Installation & Usage

### Install Dependencies
```bash
uv sync
# or
uv pip install numpy scipy matplotlib
```

### Run Examples
```bash
uv run python examples/example_2d.py
uv run python examples/example_3d.py
```

### Run Tests
```bash
uv pip install pytest
uv run pytest tests/
```

## PyCharm Integration

See `PYCHARM_SETUP.md` for detailed instructions on:
1. Configuring Python interpreter to use `.venv`
2. Setting up run configurations
3. Using `uv run` in terminal
4. Debugging and testing

## Example Usage

### 2D Example: Parabola Intersection

```python
from intersection import Line2D, ParametricCurve, compute_intersections_2d

# Parabola: y = x^2
curve = ParametricCurve(lambda t: t, lambda t: t**2, t_range=(0, 2))

# Horizontal line at y = 0.5
line = Line2D(point=(0, 0.5), direction=(1, 0))

# Compute intersections
intersections = compute_intersections_2d(line, curve, degree=5, verbose=True)

# Results
for inter in intersections:
    print(f"t = {inter['parameter']}, point = {inter['point']}")
```

### 3D Example: Sphere Intersection

```python
from intersection import Line3D, ParametricSurface, compute_intersections_3d
import numpy as np

# Unit sphere
surface = ParametricSurface(
    x_func=lambda u, v: np.sin(np.pi*v) * np.cos(2*np.pi*u),
    y_func=lambda u, v: np.sin(np.pi*v) * np.sin(2*np.pi*u),
    z_func=lambda u, v: np.cos(np.pi*v),
    u_range=(0, 1), v_range=(0, 1)
)

# Line through sphere
line = Line3D(point=(-2, 0, 0), direction=(1, 0, 0))

# Compute intersections
intersections = compute_intersections_3d(line, surface, degree=6, verbose=True)
```

## Customization

### Adjusting Accuracy
- Increase `degree` for better accuracy (slower)
- Decrease `degree` for faster computation (less accurate)
- Typical range: 3-12

### Custom Solver
You can implement your own solver by:
1. Creating a function that takes a `system` dict
2. Returning a list of solution dicts with appropriate keys
3. Passing it to the pipeline

### Custom Visualization
Use the utility functions or create your own:
```python
from intersection.utils import visualize_2d
import matplotlib.pyplot as plt

fig, ax = visualize_2d(line, curve, intersections)
# Customize the plot
ax.set_xlim(-1, 3)
plt.show()
```

## Performance Considerations

- **2D**: Very fast, typically < 0.1s
- **3D**: Slower due to 2D polynomial fitting and multi-variable solving
- **Bottlenecks**: 
  - 3D interpolation (tensor product polynomials)
  - 3D solver (multiple initial guesses)
- **Optimization tips**:
  - Use lower degree for initial testing
  - Reduce grid resolution in 3D solver
  - Cache interpolation results if solving multiple times

## Future Enhancements

Potential improvements:
1. GPU acceleration for large-scale problems
2. Adaptive degree selection based on error
3. Support for rational parametric functions
4. Interval arithmetic for guaranteed bounds
5. Parallel solving with multiple initial guesses
6. Export to various file formats (JSON, CSV, etc.)

## Testing

The framework includes basic tests in `tests/test_basic.py`:
- Geometric primitive tests
- Simple intersection tests
- Validation of algorithm steps

Run tests with:
```bash
uv run python tests/test_basic.py
```

## Documentation

- **README.md**: Project overview and structure
- **QUICKSTART.md**: Quick start guide with examples
- **PYCHARM_SETUP.md**: IDE configuration
- **This file**: Comprehensive project summary
- **Docstrings**: All functions and classes have detailed docstrings

## License

This is a custom implementation for computational geometry. Modify and extend as needed for your use case.

## Contact & Support

For questions or issues:
1. Check the examples in `examples/`
2. Review the docstrings in source files
3. Run with `verbose=True` to see algorithm steps
4. Check the test cases for usage patterns

