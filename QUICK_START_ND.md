# Quick Start: n-Dimensional Intersection Framework

## Installation

```bash
# Already installed with uv
uv sync
```

## Basic Usage

### Import

```python
from src.intersection import Hyperplane, Line, Hypersurface
import numpy as np
```

### Create a Hyperplane

```python
# 2D: x - y = 0
h = Hyperplane(coeffs=[1, -1], d=0)

# 3D: x + 2y - z + 3 = 0
h = Hyperplane(coeffs=[1, 2, -1], d=3)

# 4D: x₁ + x₂ + x₃ + x₄ = 1
h = Hyperplane(coeffs=[1, 1, 1, 1], d=-1)
```

### Create a Line

```python
# 2D line (1 hyperplane)
line_2d = Line([
    Hyperplane([1, -1], 0)  # x = y
])

# 3D line (2 hyperplanes)
line_3d = Line([
    Hyperplane([1, 0, 0], -1),  # x = 1
    Hyperplane([0, 1, 0], -2)   # y = 2
])

# 4D line (3 hyperplanes)
line_4d = Line([
    Hyperplane([1, 0, 0, 0], 0),  # x₁ = 0
    Hyperplane([0, 1, 0, 0], 0),  # x₂ = 0
    Hyperplane([0, 0, 1, 0], 0)   # x₃ = 0
])
```

### Create a Hypersurface

#### 2D Curve (1 param → 2D)

```python
# Parabola: y = x²
curve = Hypersurface(
    func=lambda u: np.array([u, u**2]),
    param_ranges=[(0, 1)],
    ambient_dim=2,
    degree=5
)

# Evaluate at u=0.5
point = curve.evaluate(0.5)  # [0.5, 0.25]

# Sample 50 points
points = curve.sample(50)
```

#### 3D Surface (2 params → 3D)

```python
# Paraboloid: z = x² + y²
surface = Hypersurface(
    func=lambda u, v: np.array([u, v, u**2 + v**2]),
    param_ranges=[(0, 1), (0, 1)],
    ambient_dim=3,
    degree=5
)

# Evaluate at (u,v) = (0.5, 0.5)
point = surface.evaluate(0.5, 0.5)  # [0.5, 0.5, 0.5]

# Sample 20x20 grid
points = surface.sample(20)
```

#### 4D Hypersurface (3 params → 4D)

```python
# 3D manifold in 4D space
hypersurface_4d = Hypersurface(
    func=lambda u, v, w: np.array([u, v, w, u*v*w]),
    param_ranges=[(0, 1), (0, 1), (0, 1)],
    ambient_dim=4,
    degree=5
)

# Evaluate at (u,v,w) = (0.5, 0.5, 0.5)
point = hypersurface_4d.evaluate(0.5, 0.5, 0.5)  # [0.5, 0.5, 0.5, 0.125]
```

## Verbose Mode

See all interpolation and Bernstein conversion details:

```python
curve = Hypersurface(
    func=lambda u: np.array([u, u**2]),
    param_ranges=[(0, 1)],
    ambient_dim=2,
    degree=5,
    verbose=True  # Enable verbose output
)
```

Output shows:
- Chebyshev nodes used
- Polynomial coefficients
- Bernstein coefficients
- Interpolation errors

## Common Patterns

### Circle

```python
circle = Hypersurface(
    func=lambda t: np.array([np.cos(2*np.pi*t), np.sin(2*np.pi*t)]),
    param_ranges=[(0, 1)],
    ambient_dim=2,
    degree=10
)
```

### Sphere

```python
sphere = Hypersurface(
    func=lambda u, v: np.array([
        np.cos(2*np.pi*u) * np.sin(np.pi*v),
        np.sin(2*np.pi*u) * np.sin(np.pi*v),
        np.cos(np.pi*v)
    ]),
    param_ranges=[(0, 1), (0, 1)],
    ambient_dim=3,
    degree=10
)
```

### Helix in 3D

```python
helix = Hypersurface(
    func=lambda t: np.array([
        np.cos(2*np.pi*t),
        np.sin(2*np.pi*t),
        t
    ]),
    param_ranges=[(0, 1)],
    ambient_dim=3,
    degree=10
)
```

### Torus in 3D

```python
R, r = 2.0, 1.0  # Major and minor radius
torus = Hypersurface(
    func=lambda u, v: np.array([
        (R + r*np.cos(2*np.pi*v)) * np.cos(2*np.pi*u),
        (R + r*np.cos(2*np.pi*v)) * np.sin(2*np.pi*u),
        r * np.sin(2*np.pi*v)
    ]),
    param_ranges=[(0, 1), (0, 1)],
    ambient_dim=3,
    degree=10
)
```

## Accessing Internals

### Polynomial Coefficients

```python
curve = Hypersurface(...)

# List of polynomials (for k=1) or coefficient arrays (for k>=2)
polynomials = curve.polynomials

# For 2D curve (k=1):
poly_x = curve.polynomials[0]  # Polynomial object
poly_y = curve.polynomials[1]  # Polynomial object

# For 3D surface (k=2):
poly_x = curve.polynomials[0]  # 2D coefficient array
poly_y = curve.polynomials[1]  # 2D coefficient array
poly_z = curve.polynomials[2]  # 2D coefficient array
```

### Bernstein Coefficients

```python
# List of Bernstein coefficient arrays
bernstein_coeffs = curve.bernstein_coeffs

# For 2D curve (k=1):
bern_x = curve.bernstein_coeffs[0]  # 1D array
bern_y = curve.bernstein_coeffs[1]  # 1D array

# For 3D surface (k=2):
bern_x = curve.bernstein_coeffs[0]  # 2D array
bern_y = curve.bernstein_coeffs[1]  # 2D array
bern_z = curve.bernstein_coeffs[2]  # 2D array
```

## Error Handling

### Invalid Hyperplane

```python
# This will raise ValueError
h = Hyperplane([0, 0], 0)  # All coefficients zero
```

### Invalid Line

```python
# Wrong number of hyperplanes
line = Line([h1])  # Need 2 hyperplanes for 3D, got 1

# Parallel hyperplanes (not independent)
h1 = Hyperplane([1, 0, 0], 0)
h2 = Hyperplane([2, 0, 0], 1)  # Parallel to h1
line = Line([h1, h2])  # Raises ValueError
```

### Invalid Hypersurface

```python
# Wrong number of parameters
hypersurface = Hypersurface(
    func=lambda u: np.array([u, u**2]),
    param_ranges=[(0, 1), (0, 1)],  # 2 ranges
    ambient_dim=2  # But k should be n-1 = 1
)  # Raises ValueError

# Function returns wrong dimension
hypersurface = Hypersurface(
    func=lambda u: np.array([u]),  # Returns 1D
    param_ranges=[(0, 1)],
    ambient_dim=2  # Expects 2D
)  # Raises ValueError on evaluate()
```

## Tips

### 1. Choose Appropriate Degree

- **Low degree (3-5)**: Fast, less accurate
- **Medium degree (5-10)**: Good balance
- **High degree (10-20)**: Slow, very accurate

```python
# For smooth functions, degree 5-10 is usually sufficient
curve = Hypersurface(..., degree=5)

# For functions with high curvature, use higher degree
curve = Hypersurface(..., degree=15)
```

### 2. Parameter Ranges

Can use any range, not just [0, 1]:

```python
curve = Hypersurface(
    func=lambda t: np.array([t, t**2]),
    param_ranges=[(-1, 1)],  # From -1 to 1
    ambient_dim=2,
    degree=5
)
```

### 3. Sampling

For visualization, sample enough points:

```python
# For curves: 50-100 points
points = curve.sample(100)

# For surfaces: 20x20 to 50x50 grid
points = surface.sample(50)
```

## Complete Example

```python
from src.intersection import Hyperplane, Line, Hypersurface
import numpy as np

# Create a line in 2D: y = x
line = Line([
    Hyperplane([1, -1], 0)
])

# Create a parabola: y = x²
curve = Hypersurface(
    func=lambda u: np.array([u, u**2]),
    param_ranges=[(0, 2)],
    ambient_dim=2,
    degree=5,
    verbose=True
)

# Sample the curve
points = curve.sample(100)

# Evaluate at specific point
point = curve.evaluate(1.0)
print(f"Point at u=1.0: {point}")

# Access Bernstein coefficients
print(f"Bernstein coefficients (x): {curve.bernstein_coeffs[0]}")
print(f"Bernstein coefficients (y): {curve.bernstein_coeffs[1]}")
```

## Next Steps

1. See `N_DIMENSIONAL_DESIGN.md` for detailed design documentation
2. See `test_new_nd_design.py` for more examples
3. Implement intersection algorithm (polynomial_system.py, solver.py)
4. Create visualization tools for n-D objects

## Summary

The n-dimensional framework provides:
- ✅ Simple, consistent API
- ✅ Works for any dimension n ≥ 2
- ✅ Automatic interpolation and Bernstein conversion
- ✅ Verbose mode for debugging
- ✅ Proper error handling
- ✅ Extensible design

Start with 2D/3D examples, then extend to higher dimensions as needed!

