# Quick Start Guide

Get started with the Intersection Computation library in 5 minutes!

## Prerequisites

- Python 3.9 or higher
- `uv` package manager installed

## Installation

1. **Install dependencies using UV:**

```powershell
# Navigate to project directory
cd D:\lsec\Python\Intersection

# Sync all dependencies
uv sync

# Or install manually
uv pip install numpy scipy matplotlib
```

2. **Verify installation:**

```powershell
uv run python -c "import numpy, scipy, matplotlib; print('All dependencies installed!')"
```

## Running Your First Example

### 2D Example: Line-Parabola Intersection

```powershell
uv run python examples/example_2d.py
```

This will:
- Compute intersections between a line and a parabola
- Print detailed output for each step of the algorithm
- Generate visualization plots
- Save results to PNG files

### 3D Example: Line-Surface Intersection

```powershell
uv run python examples/example_3d.py
```

This demonstrates 3D intersections with various surfaces (plane, paraboloid, sphere, saddle).

## Running Tests

```powershell
# Install pytest first
uv pip install pytest

# Run all tests
uv run pytest tests/

# Run specific test file
uv run python tests/test_basic.py
```

## Understanding the Algorithm Pipeline

The library follows a 5-step pipeline:

### Step 1: Interpolation
Converts parametric functions to polynomial form using Chebyshev nodes.

### Step 2: Bernstein Basis
Converts polynomials to Bernstein basis for numerical stability.

### Step 3: Polynomial System
Formulates the intersection problem as a polynomial system.

### Step 4: Solver
Solves the polynomial system using numerical methods.

### Step 5: Point Conversion
Converts parameter solutions to actual intersection points.

## Basic Usage Example

### 2D Intersection

```python
from intersection import Line2D, ParametricCurve, compute_intersections_2d

# Define a parametric curve (e.g., parabola)
curve = ParametricCurve(
    x_func=lambda t: t,
    y_func=lambda t: t**2,
    t_range=(0, 2)
)

# Define a line
line = Line2D(
    point=(0, 0.5),      # Point on the line
    direction=(1, 0)      # Direction vector
)

# Compute intersections with verbose output
intersections = compute_intersections_2d(
    line, 
    curve, 
    degree=5,      # Polynomial degree for interpolation
    verbose=True   # Print detailed steps
)

# Access results
for i, inter in enumerate(intersections):
    print(f"Intersection {i+1}:")
    print(f"  Parameter t: {inter['parameter']}")
    print(f"  Point: {inter['point']}")
    print(f"  Distance to line: {inter['distance_to_line']}")
```

### 3D Intersection

```python
from intersection import Line3D, ParametricSurface, compute_intersections_3d

# Define a parametric surface (e.g., paraboloid)
surface = ParametricSurface(
    x_func=lambda u, v: u,
    y_func=lambda u, v: v,
    z_func=lambda u, v: u**2 + v**2,
    u_range=(-1, 1),
    v_range=(-1, 1)
)

# Define a line
line = Line3D(
    point=(0, 0, 0),
    direction=(0, 0, 1)
)

# Compute intersections
intersections = compute_intersections_3d(
    line,
    surface,
    degree=5,
    verbose=True
)

# Access results
for i, inter in enumerate(intersections):
    print(f"Intersection {i+1}:")
    print(f"  Parameters (u, v): {inter['parameters']}")
    print(f"  Point: {inter['point']}")
```

## Visualization

### 2D Visualization

```python
from intersection import visualize_2d
import matplotlib.pyplot as plt

# After computing intersections
fig, ax = visualize_2d(line, curve, intersections, 
                       title="My Intersection")
plt.savefig('my_intersection.png')
plt.show()
```

### 3D Visualization

```python
from intersection import visualize_3d
import matplotlib.pyplot as plt

# After computing intersections
fig, ax = visualize_3d(line, surface, intersections,
                       title="My 3D Intersection")
plt.savefig('my_3d_intersection.png')
plt.show()
```

## Customizing the Algorithm

### Adjusting Polynomial Degree

Higher degree = better accuracy but slower computation:

```python
# Low degree (faster, less accurate)
intersections = compute_intersections_2d(line, curve, degree=3)

# High degree (slower, more accurate)
intersections = compute_intersections_2d(line, curve, degree=10)
```

### Accessing Intermediate Results

```python
from intersection import (
    interpolate_curve,
    polynomial_to_bernstein,
    create_intersection_system_2d,
    solve_polynomial_system
)

# Step 1: Interpolate
poly_x, poly_y = interpolate_curve(curve, degree=5, verbose=True)

# Step 2: Convert to Bernstein
bern_x = polynomial_to_bernstein(poly_x, verbose=True)
bern_y = polynomial_to_bernstein(poly_y, verbose=True)

# Step 3: Create system
system = create_intersection_system_2d(line, bern_x, bern_y, verbose=True)

# Step 4: Solve
solutions = solve_polynomial_system(system, verbose=True)

# Step 5: Convert to points
for sol in solutions:
    t = sol['t']
    point = curve.evaluate(t)
    print(f"t={t}, point={point}")
```

## Common Parametric Curves

### Circle
```python
import numpy as np

curve = ParametricCurve(
    x_func=lambda t: np.cos(2 * np.pi * t),
    y_func=lambda t: np.sin(2 * np.pi * t),
    t_range=(0, 1)
)
```

### Ellipse
```python
curve = ParametricCurve(
    x_func=lambda t: 2 * np.cos(2 * np.pi * t),
    y_func=lambda t: np.sin(2 * np.pi * t),
    t_range=(0, 1)
)
```

### Spiral
```python
curve = ParametricCurve(
    x_func=lambda t: t * np.cos(4 * np.pi * t),
    y_func=lambda t: t * np.sin(4 * np.pi * t),
    t_range=(0, 2)
)
```

## Common Parametric Surfaces

### Sphere
```python
surface = ParametricSurface(
    x_func=lambda u, v: np.sin(np.pi*v) * np.cos(2*np.pi*u),
    y_func=lambda u, v: np.sin(np.pi*v) * np.sin(2*np.pi*u),
    z_func=lambda u, v: np.cos(np.pi*v),
    u_range=(0, 1),
    v_range=(0, 1)
)
```

### Torus
```python
R, r = 2, 1  # Major and minor radius

surface = ParametricSurface(
    x_func=lambda u, v: (R + r*np.cos(2*np.pi*v)) * np.cos(2*np.pi*u),
    y_func=lambda u, v: (R + r*np.cos(2*np.pi*v)) * np.sin(2*np.pi*u),
    z_func=lambda u, v: r * np.sin(2*np.pi*v),
    u_range=(0, 1),
    v_range=(0, 1)
)
```

## Next Steps

1. **Explore the examples:** Run all examples in `examples/` directory
2. **Read the code:** Check the implementation in `src/intersection/`
3. **Customize:** Modify examples for your specific use case
4. **Test:** Add your own test cases in `tests/`
5. **Configure PyCharm:** See `PYCHARM_SETUP.md` for IDE integration

## Troubleshooting

### No intersections found
- Check that the line actually intersects the curve/surface
- Try increasing the polynomial degree
- Verify the parameter ranges are correct

### Poor accuracy
- Increase polynomial degree (try 8-12)
- Check that the curve/surface is smooth
- Verify the parametric functions are correct

### Slow computation
- Decrease polynomial degree
- Simplify the parametric functions
- For 3D, reduce the grid resolution in the solver

## Getting Help

- Check the examples in `examples/`
- Read the module docstrings
- Review the algorithm steps with `verbose=True`
- Examine the test cases in `tests/`

