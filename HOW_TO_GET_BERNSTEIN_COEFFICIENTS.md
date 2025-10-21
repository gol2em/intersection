# How to Get Bernstein Coefficients from the Intersection System

## Quick Answer

```python
# Create the intersection system
system = create_intersection_system(line, hypersurface)

# Get Bernstein coefficients
bernstein_coeffs = system['bernstein_coeffs']
```

## Detailed Explanation

### System Structure

When you call `create_intersection_system(line, hypersurface)`, it returns a dictionary with this structure:

```python
system = {
    'n': 2,                          # Ambient dimension
    'k': 1,                          # Number of parameters
    'line': Line object,             # The line
    'hypersurface': Hypersurface,    # The hypersurface
    'equations': [...],              # Equation specifications
    'bernstein_coeffs': [...],       # ← BERNSTEIN COEFFICIENTS HERE
    'param_ranges': [(0, 1)],        # Parameter ranges
    'degree': 12                     # Polynomial degree
}
```

### Accessing Bernstein Coefficients

#### Method 1: From System Dictionary (Recommended)

```python
# Get all Bernstein coefficients
bernstein_coeffs = system['bernstein_coeffs']

# This is a LIST of numpy arrays, one for each coordinate
# For 2D: [bern_x, bern_y]
# For 3D: [bern_x, bern_y, bern_z]
# For nD: [bern_x1, bern_x2, ..., bern_xn]
```

#### Method 2: From Hypersurface Object

```python
# Get from the hypersurface directly
bernstein_coeffs = system['hypersurface'].bernstein_coeffs
```

Both methods give you the **same result**.

## Examples

### Example 1: 2D Circle (Curve in 2D Space)

```python
from src.intersection.geometry import Hyperplane, Line, Hypersurface
from src.intersection.polynomial_system import create_intersection_system
import numpy as np

# Create unit circle with degree 12
circle = Hypersurface(
    func=lambda u: np.array([np.cos(2*np.pi*u), np.sin(2*np.pi*u)]),
    param_ranges=[(0, 1)],
    ambient_dim=2,
    degree=12
)

# Create line y = x
h1 = Hyperplane(coeffs=[1, -1], d=0)
line = Line([h1])

# Create system
system = create_intersection_system(line, circle)

# Get Bernstein coefficients
bernstein_coeffs = system['bernstein_coeffs']

# Access individual coordinates
bern_x = bernstein_coeffs[0]  # x(u) coefficients
bern_y = bernstein_coeffs[1]  # y(u) coefficients

print(f"x(u) coefficients shape: {bern_x.shape}")  # (13,) for degree 12
print(f"y(u) coefficients shape: {bern_y.shape}")  # (13,) for degree 12
print(f"x(u) coefficients: {bern_x}")
print(f"y(u) coefficients: {bern_y}")
```

**Output**:
```
x(u) coefficients shape: (13,)
y(u) coefficients shape: (13,)
x(u) coefficients: [-1.00000000e+00 -5.61977345e-16  7.36833897e-02 ...]
y(u) coefficients: [-9.54637911e-16 -2.59890470e-01  2.04216908e-16 ...]
```

### Example 2: 3D Surface (Surface in 3D Space)

```python
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

# Create system
system = create_intersection_system(line, surface)

# Get Bernstein coefficients
bernstein_coeffs = system['bernstein_coeffs']

# Access individual coordinates
bern_x = bernstein_coeffs[0]  # x(u,v) coefficients
bern_y = bernstein_coeffs[1]  # y(u,v) coefficients
bern_z = bernstein_coeffs[2]  # z(u,v) coefficients

print(f"x(u,v) coefficients shape: {bern_x.shape}")  # (5, 5) for degree 4
print(f"y(u,v) coefficients shape: {bern_y.shape}")  # (5, 5) for degree 4
print(f"z(u,v) coefficients shape: {bern_z.shape}")  # (5, 5) for degree 4
```

**Output**:
```
x(u,v) coefficients shape: (5, 5)
y(u,v) coefficients shape: (5, 5)
z(u,v) coefficients shape: (5, 5)
```

## Understanding the Shape

The shape of Bernstein coefficient arrays depends on:
- **k**: Number of parameters (dimension of parameter space)
- **degree**: Polynomial degree

| k (parameters) | Shape of each coefficient array | Example |
|----------------|----------------------------------|---------|
| k=1 (curve)    | `(degree+1,)`                   | `(13,)` for degree 12 |
| k=2 (surface)  | `(degree+1, degree+1)`          | `(5, 5)` for degree 4 |
| k=3            | `(degree+1, degree+1, degree+1)`| `(4, 4, 4)` for degree 3 |

**Number of arrays**: Always equals **n** (ambient dimension)
- 2D space: 2 arrays (x, y)
- 3D space: 3 arrays (x, y, z)
- nD space: n arrays (x₁, x₂, ..., xₙ)

## Complete Reference

### Get All Information from System

```python
# Create system
system = create_intersection_system(line, hypersurface)

# Bernstein coefficients
bernstein_coeffs = system['bernstein_coeffs']  # List of n arrays

# Individual coordinates (2D example)
bern_x = system['bernstein_coeffs'][0]
bern_y = system['bernstein_coeffs'][1]

# Individual coordinates (3D example)
bern_x = system['bernstein_coeffs'][0]
bern_y = system['bernstein_coeffs'][1]
bern_z = system['bernstein_coeffs'][2]

# Dimensions
n = system['n']                    # Ambient dimension (2, 3, 4, ...)
k = system['k']                    # Number of parameters (n-1)
degree = system['degree']          # Polynomial degree

# Other useful info
param_ranges = system['param_ranges']  # List of (min, max) tuples
line = system['line']              # Line object
hypersurface = system['hypersurface']  # Hypersurface object
equations = system['equations']    # Equation specifications
```

### Coefficient Array Indexing

For **1D case** (curve, k=1):
```python
bern_x = system['bernstein_coeffs'][0]  # Shape: (degree+1,)
# Access coefficient i:
coeff_i = bern_x[i]
```

For **2D case** (surface, k=2):
```python
bern_x = system['bernstein_coeffs'][0]  # Shape: (degree+1, degree+1)
# Access coefficient (i, j):
coeff_ij = bern_x[i, j]
```

For **3D case** (k=3):
```python
bern_x = system['bernstein_coeffs'][0]  # Shape: (degree+1, degree+1, degree+1)
# Access coefficient (i, j, k):
coeff_ijk = bern_x[i, j, k]
```

## Why Bernstein Coefficients?

The Bernstein coefficients are stored because:

1. **Convex Hull Property**: The polynomial curve/surface lies within the convex hull of its Bernstein control points
2. **Numerical Stability**: Bernstein basis is more stable than power basis
3. **LP Method**: The LP method from the 1993 paper works directly with Bernstein coefficients
4. **Subdivision**: de Casteljau's algorithm uses Bernstein coefficients for subdivision

## Common Use Cases

### Use Case 1: Inspect Coefficients

```python
system = create_intersection_system(line, hypersurface)
bern_x = system['bernstein_coeffs'][0]
print(f"Number of coefficients: {bern_x.size}")
print(f"Min coefficient: {bern_x.min()}")
print(f"Max coefficient: {bern_x.max()}")
```

### Use Case 2: Pass to LP Solver

```python
system = create_intersection_system(line, hypersurface)

# The LP solver will use these coefficients
bernstein_coeffs = system['bernstein_coeffs']
degree = system['degree']
param_ranges = system['param_ranges']

# Call LP solver (to be implemented)
# solutions = solve_system_lp(bernstein_coeffs, degree, param_ranges)
```

### Use Case 3: Compute Bounding Box

```python
system = create_intersection_system(line, hypersurface)

# For each coordinate, the Bernstein coefficients give bounds
for i, bern in enumerate(system['bernstein_coeffs']):
    min_val = bern.min()
    max_val = bern.max()
    print(f"Coordinate {i+1}: [{min_val:.6f}, {max_val:.6f}]")
```

## Summary

**Quick Access**:
```python
system = create_intersection_system(line, hypersurface)
bernstein_coeffs = system['bernstein_coeffs']
```

**Structure**:
- `bernstein_coeffs` is a **list** of **n** numpy arrays
- Each array has shape `(degree+1,)^k` where k is the number of parameters
- For 2D circle (k=1, n=2, degree=12): 2 arrays of shape `(13,)`
- For 3D surface (k=2, n=3, degree=4): 3 arrays of shape `(5, 5)`

**Access Individual Coordinates**:
```python
bern_x = system['bernstein_coeffs'][0]
bern_y = system['bernstein_coeffs'][1]
bern_z = system['bernstein_coeffs'][2]  # For 3D
```

That's it! The Bernstein coefficients are readily available in the system dictionary. 🎯

