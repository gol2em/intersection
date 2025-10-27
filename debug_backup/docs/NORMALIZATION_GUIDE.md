# Parameter Domain Normalization Guide

## Overview

The LP and PP methods from the research papers assume that the parameter domain is normalized to **[0,1]^k**. This guide explains how to normalize arbitrary parameter ranges using affine transformations.

## Why Normalization is Required

### LP and PP Methods Assumption

The Linear Programming (LP) and Projected Polyhedron (PP) methods from Sherbrooke & Patrikalakis (1993) work with Bernstein polynomials, which are naturally defined on the unit interval [0,1].

**Key Properties:**
- Bernstein basis functions B_i^n(t) are defined for t ∈ [0,1]
- Convex hull property holds on [0,1]
- LP constraints are formulated for normalized domain
- Subdivision algorithms assume [0,1]^k domain

### Example Problem

Consider a unit circle parameterized as:
```
(x, y) = (cos(u), sin(u))  where u ∈ [-π, π]
```

**Problem:** The parameter range [-π, π] is not [0,1]!

**Solution:** Use affine transformation to map [-π, π] → [0,1]

## Affine Transformation

### Mathematical Formulation

For parameter i with range [a_i, b_i], the affine transformation is:

```
u_normalized = (u_original - a_i) / (b_i - a_i)
u_original = (b_i - a_i) * u_normalized + a_i
```

Where:
- **Scale:** `scale_i = b_i - a_i`
- **Offset:** `offset_i = a_i`

### Multi-Parameter Case

For k parameters with ranges [(a_1, b_1), (a_2, b_2), ..., (a_k, b_k)]:

```
u_normalized[i] = (u_original[i] - a_i) / (b_i - a_i)  for i = 1, ..., k
```

Each parameter is normalized **independently**.

## Implementation

### Using the Normalization Utility

```python
from src.intersection.normalization import normalize_hypersurface

# Create hypersurface with arbitrary parameter ranges
circle = Hypersurface(
    func=lambda u: np.array([np.cos(u), np.sin(u)]),
    param_ranges=[(-np.pi, np.pi)],  # Original range
    ambient_dim=2,
    degree=8
)

# Normalize to [0,1]
circle_normalized, transform_info = normalize_hypersurface(circle, verbose=True)

# Now circle_normalized has param_ranges = [(0, 1)]
```

### Transform Info Dictionary

The `transform_info` dictionary contains:

```python
{
    'original_ranges': [(-π, π)],           # Original parameter ranges
    'normalized_ranges': [(0, 1)],          # Normalized ranges (always [0,1]^k)
    'scales': [2π],                         # Scale factors
    'offsets': [-π],                        # Offset values
    'forward': function,                    # Map normalized → original
    'inverse': function,                    # Map original → normalized
}
```

### Forward and Inverse Transformations

```python
# Map normalized parameter to original
u_original = transform_info['forward'](0.5)  # Returns 0.0 (middle of [-π, π])

# Map original parameter to normalized
u_normalized = transform_info['inverse'](0.0)  # Returns 0.5
```

## Workflow for LP/PP Methods

### Step 1: Create Original Hypersurface

```python
from src.intersection.geometry import Hypersurface

# Example: Circle with u ∈ [-π, π]
circle = Hypersurface(
    func=lambda u: np.array([np.cos(u), np.sin(u)]),
    param_ranges=[(-np.pi, np.pi)],
    ambient_dim=2,
    degree=8
)
```

### Step 2: Normalize Parameter Domain

```python
from src.intersection.normalization import normalize_hypersurface

circle_normalized, transform_info = normalize_hypersurface(circle)
# Now circle_normalized has param_ranges = [(0, 1)]
```

### Step 3: Create Intersection System

```python
from src.intersection.geometry import Line, Hyperplane
from src.intersection.polynomial_system import create_intersection_system

# Create line
line = Line([Hyperplane(coeffs=[1, -1], d=0)])  # y = x

# Create system with NORMALIZED hypersurface
system = create_intersection_system(line, circle_normalized)
```

### Step 4: Solve Using LP/PP Method

```python
from src.intersection.solver import solve_polynomial_system

# Solve using LP method (parameter domain is [0,1])
solutions_normalized = solve_polynomial_system(
    system,
    method='lp',
    tolerance=1e-6,
    max_depth=20,
    verbose=True
)

# Solutions are in normalized space: t ∈ [0,1]
# Example: [{'t': 0.125}, {'t': 0.625}]
```

### Step 5: Denormalize Solutions

```python
from src.intersection.normalization import denormalize_solutions

# Convert solutions back to original parameter space
solutions_original = denormalize_solutions(solutions_normalized, transform_info)

# Now solutions are in original space: t ∈ [-π, π]
# Example: [{'t': -2.356}, {'t': 0.785}]
```

### Step 6: Verify Solutions

```python
# Verify using original hypersurface
for sol in solutions_original:
    t = sol['t']
    point = circle.evaluate(t)
    print(f"t = {t:.4f}, point = ({point[0]:.4f}, {point[1]:.4f})")
```

## Examples

### Example 1: Unit Circle [-π, π] → [0, 1]

```python
# Original parameterization
circle = Hypersurface(
    func=lambda u: np.array([np.cos(u), np.sin(u)]),
    param_ranges=[(-np.pi, np.pi)],
    ambient_dim=2,
    degree=8
)

# Normalize
circle_normalized, transform = normalize_hypersurface(circle)

# Transformation
# u_original = 2π * u_normalized - π
# u_normalized = (u_original + π) / (2π)

# Test points
transform['forward'](0.0)   # Returns -π
transform['forward'](0.5)   # Returns 0
transform['forward'](1.0)   # Returns π
```

### Example 2: 3D Surface with 2 Parameters

```python
# Original parameterization
surface = Hypersurface(
    func=lambda u, v: np.array([u, v, u**2 + v**2]),
    param_ranges=[(-2, 2), (-1, 1)],  # u ∈ [-2,2], v ∈ [-1,1]
    ambient_dim=3,
    degree=4
)

# Normalize
surface_normalized, transform = normalize_hypersurface(surface)

# Transformation
# u_original = 4 * u_normalized - 2
# v_original = 2 * v_normalized - 1

# Test points
transform['forward'](0.0, 0.0)   # Returns (-2, -1)
transform['forward'](0.5, 0.5)   # Returns (0, 0)
transform['forward'](1.0, 1.0)   # Returns (2, 1)
```

## Verification

### Verify Normalization Preserves Geometry

```python
from src.intersection.normalization import verify_normalization

passed = verify_normalization(
    hypersurface_original,
    hypersurface_normalized,
    transform_info,
    n_test_points=20,
    verbose=True
)

# Checks that evaluating at corresponding points gives same result
```

### Manual Verification

```python
# Pick a normalized parameter
u_norm = 0.3

# Get corresponding original parameter
u_orig = transform_info['forward'](u_norm)

# Evaluate both hypersurfaces
point_norm = hypersurface_normalized.evaluate(u_norm)
point_orig = hypersurface_original.evaluate(u_orig)

# Should be identical
assert np.allclose(point_norm, point_orig)
```

## Key Properties

### 1. Geometry Preservation

✅ **The normalized hypersurface represents the same geometric object**

The affine transformation only changes the parameterization, not the geometry.

### 2. Bernstein Coefficients

⚠️ **Bernstein coefficients will be different**

Because the parameterization changes, the Bernstein coefficients will be different. However, they still represent the same geometric curve/surface.

### 3. Independence

✅ **Each parameter is normalized independently**

For multi-parameter hypersurfaces, each parameter dimension is transformed separately.

### 4. Invertibility

✅ **The transformation is invertible**

You can always convert between normalized and original parameter spaces.

## Common Use Cases

### Use Case 1: Circle with Non-Standard Range

```python
# Circle parameterized on [0, 2π]
circle = Hypersurface(
    func=lambda u: np.array([np.cos(u), np.sin(u)]),
    param_ranges=[(0, 2*np.pi)],
    ambient_dim=2,
    degree=8
)

# Normalize to [0, 1]
circle_norm, transform = normalize_hypersurface(circle)
```

### Use Case 2: Curve Segment

```python
# Parabola segment on [2, 5]
parabola = Hypersurface(
    func=lambda u: np.array([u, u**2]),
    param_ranges=[(2, 5)],
    ambient_dim=2,
    degree=5
)

# Normalize to [0, 1]
parabola_norm, transform = normalize_hypersurface(parabola)
```

### Use Case 3: Surface with Different Ranges

```python
# Surface with u ∈ [-10, 10], v ∈ [0, 5]
surface = Hypersurface(
    func=lambda u, v: np.array([u, v, np.sin(u) * np.cos(v)]),
    param_ranges=[(-10, 10), (0, 5)],
    ambient_dim=3,
    degree=6
)

# Normalize to [0, 1] × [0, 1]
surface_norm, transform = normalize_hypersurface(surface)
```

## Summary

1. ✅ **LP/PP methods require normalized domain [0,1]^k**
2. ✅ **Use `normalize_hypersurface()` to normalize**
3. ✅ **Solve in normalized space**
4. ✅ **Use `denormalize_solutions()` to convert back**
5. ✅ **Geometry is preserved, only parameterization changes**

## Next Steps

- Implement LP method using normalized hypersurfaces
- Implement PP method using normalized hypersurfaces
- Test with various parameter ranges
- Verify solutions in original parameter space

