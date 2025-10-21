# New Design Summary

## Overview

The framework has been redesigned according to your specifications:

1. **Line2D**: Uses implicit form `ax + by + c = 0`
2. **Line3D**: Uses two planes `a1x + b1y + c1z + d1 = 0` and `a2x + b2y + c2z + d2 = 0`
3. **Curve**: 2D parametric curve with **automatic** interpolation and Bernstein conversion
4. **Surface**: 3D parametric surface with **automatic** interpolation and Bernstein conversion

## Key Changes

### 1. Line Representations

#### Line2D (Implicit Form)
```python
# Direct creation
line = Line2D(a=1, b=-1, c=0)  # x - y = 0

# From two points
line = Line2D.from_two_points((0, 0), (1, 1))

# From point and direction
line = Line2D.from_point_and_direction((0, 0), (1, 1))
```

**Methods:**
- `evaluate_y(x)`: Get y for given x
- `evaluate_x(y)`: Get x for given y
- `distance_to_point(point)`: Distance from point to line

#### Line3D (Two Planes)
```python
# Direct creation
line = Line3D(a1=0, b1=1, c1=0, d1=0,  # Plane 1: y = 0
              a2=0, b2=0, c2=1, d2=0)  # Plane 2: z = 0

# From point and direction
line = Line3D.from_point_and_direction((0, 0, 0), (1, 0, 0))
```

**Methods:**
- `get_direction()`: Get direction vector
- `get_point()`: Get a point on the line
- `distance_to_point(point)`: Distance from point to line

### 2. Curve Class (Automatic Processing)

The `Curve` class now **automatically** performs interpolation and Bernstein conversion upon initialization:

```python
curve = Curve(
    x_func=lambda u: u,
    y_func=lambda u: u**2,
    u_range=(0, 1),
    degree=5,        # Polynomial degree
    verbose=True     # Show processing steps
)
```

**What happens automatically:**
1. ✅ Samples the curve at Chebyshev nodes
2. ✅ Fits polynomials to x(u) and y(u)
3. ✅ Converts polynomials to Bernstein basis
4. ✅ Stores all results as attributes

**Attributes after initialization:**
- `poly_x`, `poly_y`: Polynomial objects (numpy.polynomial.Polynomial)
- `bernstein_x`, `bernstein_y`: Bernstein coefficients (numpy arrays)
- `x_func`, `y_func`: Original parametric functions
- `u_range`: Parameter range
- `degree`: Polynomial degree

**Methods:**
- `evaluate(u)`: Evaluate at parameter u using **original** functions
- `sample(n_points)`: Sample n points along the curve

### 3. Surface Class (Automatic Processing)

The `Surface` class also **automatically** performs interpolation and Bernstein conversion:

```python
surface = Surface(
    x_func=lambda u, v: u,
    y_func=lambda u, v: v,
    z_func=lambda u, v: u**2 + v**2,
    u_range=(0, 1),
    v_range=(0, 1),
    degree=5,
    verbose=True
)
```

**What happens automatically:**
1. ✅ Samples the surface at Chebyshev grid points
2. ✅ Fits 2D tensor product polynomials to x(u,v), y(u,v), z(u,v)
3. ✅ Converts polynomials to Bernstein basis
4. ✅ Stores all results as attributes

**Attributes after initialization:**
- `poly_x`, `poly_y`, `poly_z`: 2D polynomial coefficient matrices
- `bernstein_x`, `bernstein_y`, `bernstein_z`: Bernstein coefficient matrices
- `x_func`, `y_func`, `z_func`: Original parametric functions
- `u_range`, `v_range`: Parameter ranges
- `degree`: Polynomial degree

**Methods:**
- `evaluate(u, v)`: Evaluate at parameters (u, v) using **original** functions
- `sample(n_u, n_v)`: Sample grid of points on the surface

## Usage Examples

### Example 1: Create a Curve

```python
from src.intersection.geometry import Curve

# Parabola: y = x^2
curve = Curve(
    x_func=lambda u: u,
    y_func=lambda u: u**2,
    u_range=(0, 2),
    degree=5,
    verbose=True  # See interpolation details
)

# Access results
print(f"Polynomial x(u): {curve.poly_x}")
print(f"Bernstein coefficients (x): {curve.bernstein_x}")

# Evaluate at a point
point = curve.evaluate(0.5)
print(f"Point at u=0.5: {point}")
```

### Example 2: Create a Surface

```python
from src.intersection.geometry import Surface
import numpy as np

# Sphere (using spherical coordinates)
surface = Surface(
    x_func=lambda u, v: np.sin(np.pi*v) * np.cos(2*np.pi*u),
    y_func=lambda u, v: np.sin(np.pi*v) * np.sin(2*np.pi*u),
    z_func=lambda u, v: np.cos(np.pi*v),
    u_range=(0, 1),
    v_range=(0, 1),
    degree=6,
    verbose=True
)

# Access results
print(f"Bernstein coefficients shape: {surface.bernstein_x.shape}")

# Evaluate at a point
point = surface.evaluate(0.5, 0.5)
print(f"Point at (u=0.5, v=0.5): {point}")
```

### Example 3: Create Lines

```python
from src.intersection.geometry import Line2D, Line3D

# 2D line: x - y = 0 (diagonal)
line2d = Line2D(a=1, b=-1, c=0)
print(line2d)  # Line2D(1.000x + -1.000y + 0.000 = 0)

# Or from point and direction
line2d = Line2D.from_point_and_direction((0, 0), (1, 1))

# 3D line through origin along x-axis
line3d = Line3D.from_point_and_direction((0, 0, 0), (1, 0, 0))
print(f"Direction: {line3d.get_direction()}")
print(f"Point: {line3d.get_point()}")
```

## Benefits of New Design

### 1. **Automatic Processing**
- No need to manually call interpolation and Bernstein conversion
- Everything happens in the constructor
- Cleaner, more intuitive API

### 2. **Verbose Mode**
- Set `verbose=True` to see all processing steps
- Helps verify algorithm correctness
- Shows interpolation errors and coefficients

### 3. **Mathematically Correct Line Representations**
- Line2D uses standard implicit form
- Line3D uses intersection of two planes (mathematically rigorous)
- Both support convenient factory methods

### 4. **Consistent Parameter Naming**
- Curves use `u` as parameter
- Surfaces use `(u, v)` as parameters
- Clear and consistent throughout

## File Structure

```
src/intersection/
├── geometry.py          # Line2D, Line3D, Curve, Surface classes
├── interpolation.py     # Polynomial interpolation (called automatically)
├── bernstein.py         # Bernstein conversion (called automatically)
├── polynomial_system.py # System formation (needs update)
├── solver.py            # Polynomial solver (needs update)
└── utils.py             # Visualization utilities
```

## Next Steps

The following modules need to be updated to work with the new design:

1. ✅ **geometry.py** - COMPLETE
2. ✅ **interpolation.py** - COMPLETE (updated to use u_range)
3. ✅ **bernstein.py** - Already compatible
4. ⏳ **polynomial_system.py** - Needs update for new Line representations
5. ⏳ **solver.py** - May need minor updates
6. ⏳ **__init__.py** - Needs update for high-level API
7. ⏳ **examples/** - Need to be rewritten with new API
8. ⏳ **tests/** - Need to be updated

## Testing

Run the test to verify the new design:

```bash
.venv\Scripts\python.exe test_new_design.py
```

This test demonstrates:
- ✅ Line2D creation and methods
- ✅ Line3D creation and methods
- ✅ Curve automatic interpolation and Bernstein conversion
- ✅ Surface automatic interpolation and Bernstein conversion

All tests pass successfully!

## Migration Guide

### Old API → New API

**Curves:**
```python
# OLD
curve = ParametricCurve(x_func, y_func, t_range=(0, 1))
poly_x, poly_y = interpolate_curve(curve, degree=5)
bern_x = polynomial_to_bernstein(poly_x)

# NEW
curve = Curve(x_func, y_func, u_range=(0, 1), degree=5)
# poly_x, poly_y, bern_x, bern_y are already computed!
```

**Surfaces:**
```python
# OLD
surface = ParametricSurface(x_func, y_func, z_func, u_range, v_range)
poly_x, poly_y, poly_z = interpolate_surface(surface, degree=5)
bern_x = polynomial_2d_to_bernstein(poly_x)

# NEW
surface = Surface(x_func, y_func, z_func, u_range, v_range, degree=5)
# Everything is already computed!
```

**Lines:**
```python
# OLD
line = Line2D(point=(0, 0), direction=(1, 1))

# NEW
line = Line2D.from_point_and_direction((0, 0), (1, 1))
# Or directly: Line2D(a=-1, b=1, c=0)
```

## Conclusion

The new design is:
- ✅ More intuitive and user-friendly
- ✅ Mathematically rigorous (proper line representations)
- ✅ Automatic (no manual steps required)
- ✅ Transparent (verbose mode shows everything)
- ✅ Well-tested and working

The framework is ready for the next phase: updating the intersection computation logic to work with the new line representations.

