# n-Dimensional Intersection Framework

## Overview

The project has been completely reworked to use a generalized n-dimensional framework. The new design supports:

- **Hyperplanes** in n-dimensional space
- **Lines** in n-dimensional space (intersection of n-1 hyperplanes)
- **Hypersurfaces**: (n-1)-dimensional parametric manifolds in n-dimensional space

## Key Design Principles

### 1. **Single Callable for Parametric Functions**
Instead of separate functions for each coordinate, the parametric function returns all components:

```python
# OLD (separate functions):
curve = Curve(
    x_func=lambda u: u,
    y_func=lambda u: u**2,
    ...
)

# NEW (single function):
curve = Hypersurface(
    func=lambda u: np.array([u, u**2]),
    ...
)
```

### 2. **Minimal Methods**
Hyperplane and Line classes only have validation logic - no extra methods like distance calculation, evaluation, etc.

### 3. **No Backward Compatibility**
Clean slate - removed all old Line2D, Line3D, Curve, Surface classes.

### 4. **Hypersurface Class**
Specifically for (n-1)-dimensional manifolds in n-dimensional space (k = n-1), which is what's needed for intersection computation.

## Class Hierarchy

```
Hyperplane
  ├─ Represents: a₁x₁ + a₂x₂ + ... + aₙxₙ + d = 0
  └─ Methods: __init__, __repr__

Line
  ├─ Represents: Intersection of (n-1) hyperplanes
  └─ Methods: __init__, __repr__

Hypersurface
  ├─ Represents: (n-1) parameters → n-dimensional space
  ├─ Automatic: Interpolation + Bernstein conversion
  └─ Methods: __init__, evaluate, sample, __repr__
```

## Usage Examples

### 2D Case: Curve in 2D Space

```python
from src.intersection import Hyperplane, Line, Hypersurface
import numpy as np

# Line in 2D (1 hyperplane)
h1 = Hyperplane(coeffs=[1, -1], d=0)  # x - y = 0
line_2d = Line([h1])

# Curve in 2D (1 param → 2D)
curve = Hypersurface(
    func=lambda u: np.array([u, u**2]),
    param_ranges=[(0, 1)],
    ambient_dim=2,
    degree=5,
    verbose=True
)

# Evaluate
point = curve.evaluate(0.5)  # Returns [0.5, 0.25]
```

### 3D Case: Surface in 3D Space

```python
# Line in 3D (2 hyperplanes)
h1 = Hyperplane(coeffs=[1, 0, 0], d=-1)  # x = 1
h2 = Hyperplane(coeffs=[0, 1, 0], d=-2)  # y = 2
line_3d = Line([h1, h2])

# Surface in 3D (2 params → 3D)
surface = Hypersurface(
    func=lambda u, v: np.array([u, v, u**2 + v**2]),
    param_ranges=[(0, 1), (0, 1)],
    ambient_dim=3,
    degree=5,
    verbose=True
)

# Evaluate
point = surface.evaluate(0.5, 0.5)  # Returns [0.5, 0.5, 0.5]
```

### 4D Case: Hypersurface in 4D Space

```python
# Line in 4D (3 hyperplanes)
h1 = Hyperplane(coeffs=[1, 0, 0, 0], d=0)  # x₁ = 0
h2 = Hyperplane(coeffs=[0, 1, 0, 0], d=0)  # x₂ = 0
h3 = Hyperplane(coeffs=[0, 0, 1, 0], d=0)  # x₃ = 0
line_4d = Line([h1, h2, h3])

# Hypersurface in 4D (3 params → 4D)
hypersurface_4d = Hypersurface(
    func=lambda u, v, w: np.array([u, v, w, u*v*w]),
    param_ranges=[(0, 1), (0, 1), (0, 1)],
    ambient_dim=4,
    degree=5,
    verbose=True
)

# Evaluate
point = hypersurface_4d.evaluate(0.5, 0.5, 0.5)  # Returns [0.5, 0.5, 0.5, 0.125]
```

## Module Structure

### `geometry.py`

**Hyperplane Class:**
- `__init__(coeffs, d)`: Initialize hyperplane
- Validates that coefficients are not all zero

**Line Class:**
- `__init__(hyperplanes)`: Initialize from list of hyperplanes
- Validates:
  - All hyperplanes have same dimension
  - Exactly n-1 hyperplanes for n-D space
  - Hyperplanes are linearly independent

**Hypersurface Class:**
- `__init__(func, param_ranges, ambient_dim, degree, verbose)`: Initialize with automatic processing
- `evaluate(*params)`: Evaluate at parameter values
- `sample(n_samples)`: Sample points on hypersurface
- Automatically performs:
  1. Polynomial interpolation using Chebyshev nodes
  2. Conversion to Bernstein basis

### `interpolation.py`

**Main Function:**
- `interpolate_hypersurface(hypersurface, degree, verbose)`: Interpolate parametric hypersurface

**Internal Functions:**
- `_interpolate_curve(hypersurface, degree, verbose)`: For k=1 (curves)
- `_interpolate_surface(hypersurface, degree, verbose)`: For k=2 (surfaces)
- `_interpolate_general(hypersurface, degree, verbose)`: For k≥3 (general case)
- `_fit_2d_polynomial(...)`: Fit 2D tensor product polynomial
- `_fit_kd_polynomial(...)`: Fit k-D tensor product polynomial

**Returns:**
- For k=1: List of `Polynomial` objects
- For k≥2: List of coefficient tensors (numpy arrays)

### `bernstein.py`

**Main Function:**
- `polynomial_nd_to_bernstein(poly, k, verbose)`: Convert k-D polynomial to Bernstein basis

**Internal Functions:**
- `_polynomial_1d_to_bernstein(poly, verbose)`: For k=1
- `_polynomial_2d_to_bernstein(poly_coeffs, verbose)`: For k=2
- `_polynomial_kd_to_bernstein(poly_coeffs, k, verbose)`: For k≥3

**Evaluation Functions:**
- `evaluate_bernstein_1d(bernstein_coeffs, t)`: Evaluate 1D Bernstein polynomial
- `evaluate_bernstein_2d(bernstein_coeffs, u, v)`: Evaluate 2D Bernstein polynomial
- `evaluate_bernstein_kd(bernstein_coeffs, *params)`: Evaluate k-D Bernstein polynomial

## Mathematical Background

### Hyperplane in n-D
A hyperplane in n-dimensional space is defined by:
```
a₁x₁ + a₂x₂ + ... + aₙxₙ + d = 0
```

### Line in n-D
A line in n-dimensional space is the intersection of (n-1) hyperplanes:
```
H₁: a₁₁x₁ + a₁₂x₂ + ... + a₁ₙxₙ + d₁ = 0
H₂: a₂₁x₁ + a₂₂x₂ + ... + a₂ₙxₙ + d₂ = 0
...
Hₙ₋₁: a(n-1)₁x₁ + a(n-1)₂x₂ + ... + a(n-1)ₙxₙ + dₙ₋₁ = 0
```

### Hypersurface in n-D
A (n-1)-dimensional hypersurface in n-dimensional space is defined parametrically:
```
f: ℝ^(n-1) → ℝ^n
(u₁, u₂, ..., u_(n-1)) ↦ (x₁, x₂, ..., xₙ)
```

Examples:
- **n=2**: Curve (1 param → 2D)
- **n=3**: Surface (2 params → 3D)
- **n=4**: 3D manifold (3 params → 4D)

### Tensor Product Polynomials
For k parameters, we use tensor product polynomials:
```
P(u₁, u₂, ..., uₖ) = ∑ᵢ₁∑ᵢ₂...∑ᵢₖ cᵢ₁ᵢ₂...ᵢₖ u₁^i₁ u₂^i₂ ... uₖ^iₖ
```

### Bernstein Basis
The Bernstein basis in k dimensions is:
```
Bᵢ₁ᵢ₂...ᵢₖ(u₁, u₂, ..., uₖ) = Bᵢ₁^n₁(u₁) · Bᵢ₂^n₂(u₂) · ... · Bᵢₖ^nₖ(uₖ)
```
where `Bᵢ^n(u) = C(n,i) · u^i · (1-u)^(n-i)` is the 1D Bernstein basis.

## Advantages of New Design

### 1. **Dimension Agnostic**
Works for any dimension n ≥ 2:
- 2D: Curves
- 3D: Surfaces
- 4D+: Hypersurfaces

### 2. **Cleaner API**
Single function returns all components:
```python
func=lambda u, v: np.array([u, v, u**2 + v**2])
```
vs. old:
```python
x_func=lambda u, v: u
y_func=lambda u, v: v
z_func=lambda u, v: u**2 + v**2
```

### 3. **Consistent Naming**
- `Hyperplane`: n-1 dimensional object in n-D space
- `Line`: 1-dimensional object (intersection of n-1 hyperplanes)
- `Hypersurface`: n-1 dimensional parametric manifold

### 4. **Extensible**
Easy to add:
- Higher dimensions
- Different interpolation methods
- Different basis functions

### 5. **Mathematically Rigorous**
Proper representation of geometric objects:
- Lines as hyperplane intersections
- Hypersurfaces as parametric manifolds

## Testing

Run the test file to verify all dimensions work:

```bash
.venv\Scripts\python.exe test_new_nd_design.py
```

Tests include:
- ✅ Hyperplane in 2D
- ✅ Line in 2D (1 hyperplane)
- ✅ Line in 3D (2 hyperplanes)
- ✅ Hypersurface 1→2D (curve)
- ✅ Hypersurface 2→3D (surface)
- ✅ Hypersurface 3→4D (3D manifold in 4D space)
- ✅ Line in 4D (3 hyperplanes)

## Next Steps

1. **Update polynomial_system.py** - Generalize for n-D
2. **Update solver.py** - Handle n-D systems
3. **Create new tests** - pytest tests for new classes
4. **Create examples** - Demonstrate n-D intersections
5. **Update documentation** - Complete API reference

## Migration from Old Design

### Old Classes → New Classes

| Old | New | Notes |
|-----|-----|-------|
| `Line2D(a, b, c)` | `Line([Hyperplane([a, b], c)])` | Single hyperplane |
| `Line3D(a1, b1, c1, d1, a2, b2, c2, d2)` | `Line([Hyperplane([a1, b1, c1], d1), Hyperplane([a2, b2, c2], d2)])` | Two hyperplanes |
| `Curve(x_func, y_func, ...)` | `Hypersurface(lambda u: np.array([x_func(u), y_func(u)]), ...)` | Single function |
| `Surface(x_func, y_func, z_func, ...)` | `Hypersurface(lambda u, v: np.array([x_func(u,v), y_func(u,v), z_func(u,v)]), ...)` | Single function |

### Old Files (Backed Up)

- `geometry_old.py`
- `interpolation_old.py`
- `bernstein_old.py`

## Summary

The new n-dimensional framework provides:
- ✅ **Generalized design** for any dimension
- ✅ **Cleaner API** with single callable
- ✅ **Minimal classes** with only essential methods
- ✅ **Automatic processing** (interpolation + Bernstein)
- ✅ **Mathematically rigorous** representations
- ✅ **Fully tested** for dimensions 2, 3, and 4
- ✅ **Extensible** for future enhancements

The framework is ready for implementing the intersection algorithm!

