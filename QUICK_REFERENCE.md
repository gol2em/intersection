# Quick Reference Guide - New Design

## Installation

```bash
# Install dependencies
.venv\Scripts\pip.exe install numpy scipy matplotlib

# Or if uv is available
uv sync
```

## Import

```python
from src.intersection.geometry import Line2D, Line3D, Curve, Surface
```

## Line2D - Implicit Form: ax + by + c = 0

### Creation

```python
# Method 1: Direct (implicit form)
line = Line2D(a=1, b=-1, c=0)  # x - y = 0

# Method 2: From two points
line = Line2D.from_two_points((0, 0), (1, 1))

# Method 3: From point and direction
line = Line2D.from_point_and_direction(point=(0, 0), direction=(1, 1))
```

### Methods

```python
# Evaluate y for given x
y = line.evaluate_y(x=2.0)

# Evaluate x for given y
x = line.evaluate_x(y=3.0)

# Distance from point to line
dist = line.distance_to_point((1, 2))

# String representation
print(line)  # Line2D(1.000x + -1.000y + 0.000 = 0)
```

## Line3D - Two Planes

### Creation

```python
# Method 1: Direct (two planes)
line = Line3D(
    a1=0, b1=1, c1=0, d1=0,  # Plane 1: y = 0
    a2=0, b2=0, c2=1, d2=0   # Plane 2: z = 0
)

# Method 2: From point and direction
line = Line3D.from_point_and_direction(
    point=(0, 0, 0),
    direction=(1, 0, 0)
)
```

### Methods

```python
# Get direction vector
direction = line.get_direction()  # Returns normalized numpy array

# Get a point on the line
point = line.get_point()  # Returns numpy array [x, y, z]

# Distance from point to line
dist = line.distance_to_point((1, 2, 3))

# String representation
print(line)  # Shows both plane equations
```

## Curve - 2D Parametric Curve

### Creation (Automatic Interpolation!)

```python
curve = Curve(
    x_func=lambda u: u,           # x(u) function
    y_func=lambda u: u**2,        # y(u) function
    u_range=(0, 1),               # Parameter range
    degree=5,                     # Polynomial degree (default: 5)
    verbose=True                  # Show processing (default: False)
)
```

**What happens automatically:**
1. ✅ Samples at Chebyshev nodes
2. ✅ Fits polynomials
3. ✅ Converts to Bernstein basis
4. ✅ Stores all results

### Attributes

```python
# Polynomial objects (numpy.polynomial.Polynomial)
curve.poly_x
curve.poly_y

# Bernstein coefficients (numpy arrays)
curve.bernstein_x
curve.bernstein_y

# Original functions
curve.x_func
curve.y_func

# Parameters
curve.u_range
curve.degree
```

### Methods

```python
# Evaluate at parameter u (uses original functions)
point = curve.evaluate(u=0.5)  # Returns [x, y]

# Sample n points
points = curve.sample(n_points=100)  # Returns (n, 2) array

# String representation
print(curve)  # Curve(u_range=(0, 1), degree=5)
```

## Surface - 3D Parametric Surface

### Creation (Automatic Interpolation!)

```python
import numpy as np

surface = Surface(
    x_func=lambda u, v: u,        # x(u,v) function
    y_func=lambda u, v: v,        # y(u,v) function
    z_func=lambda u, v: u**2 + v**2,  # z(u,v) function
    u_range=(0, 1),               # First parameter range
    v_range=(0, 1),               # Second parameter range
    degree=5,                     # Polynomial degree (default: 5)
    verbose=True                  # Show processing (default: False)
)
```

**What happens automatically:**
1. ✅ Samples at Chebyshev grid
2. ✅ Fits 2D tensor product polynomials
3. ✅ Converts to Bernstein basis
4. ✅ Stores all results

### Attributes

```python
# 2D polynomial coefficient matrices
surface.poly_x  # Shape: (degree+1, degree+1)
surface.poly_y
surface.poly_z

# Bernstein coefficient matrices
surface.bernstein_x
surface.bernstein_y
surface.bernstein_z

# Original functions
surface.x_func
surface.y_func
surface.z_func

# Parameters
surface.u_range
surface.v_range
surface.degree
```

### Methods

```python
# Evaluate at parameters (u, v) (uses original functions)
point = surface.evaluate(u=0.5, v=0.5)  # Returns [x, y, z]

# Sample grid of points
X, Y, Z = surface.sample(n_u=30, n_v=30)  # Returns meshgrid arrays

# String representation
print(surface)  # Surface(u_range=(0, 1), v_range=(0, 1), degree=5)
```

## Common Parametric Functions

### 2D Curves

```python
# Parabola: y = x^2
curve = Curve(
    x_func=lambda u: u,
    y_func=lambda u: u**2,
    u_range=(0, 2)
)

# Circle: x^2 + y^2 = 1
curve = Curve(
    x_func=lambda u: np.cos(2*np.pi*u),
    y_func=lambda u: np.sin(2*np.pi*u),
    u_range=(0, 1)
)

# Sine wave
curve = Curve(
    x_func=lambda u: u,
    y_func=lambda u: np.sin(2*np.pi*u),
    u_range=(0, 2)
)

# Ellipse
curve = Curve(
    x_func=lambda u: 2*np.cos(2*np.pi*u),
    y_func=lambda u: np.sin(2*np.pi*u),
    u_range=(0, 1)
)
```

### 3D Surfaces

```python
# Plane: z = x + y
surface = Surface(
    x_func=lambda u, v: u,
    y_func=lambda u, v: v,
    z_func=lambda u, v: u + v,
    u_range=(0, 1),
    v_range=(0, 1)
)

# Paraboloid: z = x^2 + y^2
surface = Surface(
    x_func=lambda u, v: u,
    y_func=lambda u, v: v,
    z_func=lambda u, v: u**2 + v**2,
    u_range=(-1, 1),
    v_range=(-1, 1)
)

# Sphere: x^2 + y^2 + z^2 = 1
surface = Surface(
    x_func=lambda u, v: np.sin(np.pi*v) * np.cos(2*np.pi*u),
    y_func=lambda u, v: np.sin(np.pi*v) * np.sin(2*np.pi*u),
    z_func=lambda u, v: np.cos(np.pi*v),
    u_range=(0, 1),
    v_range=(0, 1)
)

# Saddle: z = x^2 - y^2
surface = Surface(
    x_func=lambda u, v: u,
    y_func=lambda u, v: v,
    z_func=lambda u, v: u**2 - v**2,
    u_range=(-1, 1),
    v_range=(-1, 1)
)
```

## Tips

### Choosing Polynomial Degree

- **Low degree (3-5)**: Faster, less accurate, good for simple curves
- **Medium degree (5-8)**: Balanced, good for most cases
- **High degree (8-12)**: Slower, more accurate, for complex curves

### Verbose Mode

Always use `verbose=True` when:
- Testing new parametric functions
- Debugging interpolation issues
- Verifying algorithm correctness
- Learning how the system works

### Parameter Ranges

- Use `u_range` and `v_range` to control which part of the curve/surface to use
- Ranges don't have to be (0, 1) - use any interval
- Smaller ranges = better interpolation accuracy

## Complete Example

```python
import numpy as np
from src.intersection.geometry import Line2D, Curve

# Create a parabola
curve = Curve(
    x_func=lambda u: u,
    y_func=lambda u: u**2,
    u_range=(0, 2),
    degree=5,
    verbose=True
)

# Create a horizontal line at y = 0.5
line = Line2D.from_point_and_direction(
    point=(0, 0.5),
    direction=(1, 0)
)

# Access interpolation results
print(f"\nPolynomial x(u): {curve.poly_x}")
print(f"Polynomial y(u): {curve.poly_y}")
print(f"Bernstein coefficients (x): {curve.bernstein_x}")
print(f"Bernstein coefficients (y): {curve.bernstein_y}")

# Evaluate curve at specific point
point = curve.evaluate(0.5)
print(f"\nCurve at u=0.5: {point}")

# Sample curve
points = curve.sample(100)
print(f"Sampled {len(points)} points")

# Check line
print(f"\nLine: {line}")
print(f"Distance from (1, 0) to line: {line.distance_to_point((1, 0))}")
```

## Testing

Run the test file to verify everything works:

```bash
.venv\Scripts\python.exe test_new_design.py
```

This will test all four classes with verbose output showing the automatic processing.

