# Equation Bernstein Coefficients - Corrected Implementation

## Summary

The system now correctly stores the Bernstein coefficients of the **polynomial equations** (after applying hyperplane constraints), not just the raw hypersurface coefficients.

## What Changed

### Before (Incorrect)
The system only stored the hypersurface Bernstein coefficients:
```python
system['bernstein_coeffs'] = [bern_x, bern_y, bern_z, ...]  # Just coordinates
```

### After (Correct) ✅
The system now stores **both**:
1. **Hypersurface coefficients** (coordinates)
2. **Equation coefficients** (after applying hyperplane constraints)

```python
system['hypersurface_bernstein_coeffs'] = [bern_x, bern_y, bern_z, ...]  # Coordinates
system['equation_bernstein_coeffs'] = [eq1_bern, eq2_bern, ...]  # Equations
```

## Mathematical Formulation

### Hyperplane Equation
For hyperplane `H_i: a_i1*x_1 + a_i2*x_2 + ... + a_in*x_n + d_i = 0`

### Hypersurface Representation
Each coordinate has Bernstein representation:
- `x_1(u) = Σ bern_x1[α] * B_α(u)`
- `x_2(u) = Σ bern_x2[α] * B_α(u)`
- ...
- `x_n(u) = Σ bern_xn[α] * B_α(u)`

Where `B_α(u)` are Bernstein basis functions and `α` is a multi-index.

### Equation Polynomial
Substituting into the hyperplane equation:
```
p_i(u) = a_i1*x_1(u) + a_i2*x_2(u) + ... + a_in*x_n(u) + d_i
       = a_i1*Σ bern_x1[α]*B_α(u) + a_i2*Σ bern_x2[α]*B_α(u) + ... + d_i
       = Σ (a_i1*bern_x1[α] + a_i2*bern_x2[α] + ... + a_in*bern_xn[α] + d_i) * B_α(u)
```

### Equation Bernstein Coefficients
```
eq_bern_i[α] = a_i1*bern_x1[α] + a_i2*bern_x2[α] + ... + a_in*bern_xn[α] + d_i
```

This is a **linear combination** of the hypersurface Bernstein coefficients!

## Examples

### Example 1: 2D Circle with Line y=x

**Hyperplane**: `x - y = 0` (coefficients: `[1, -1]`, constant: `0`)

**Hypersurface** (unit circle):
- `x(u)` has Bernstein coefficients: `bern_x`
- `y(u)` has Bernstein coefficients: `bern_y`

**Equation polynomial**: `p(u) = x(u) - y(u) + 0`

**Equation Bernstein coefficients**:
```python
eq_bern = 1*bern_x + (-1)*bern_y + 0
        = bern_x - bern_y
```

**Verification**:
```
bern_x = [-0.968, 0.000, 0.406, 0.000, -0.420, -0.000]
bern_y = [ 0.000, -0.604, -0.000, 0.441,  0.000, -1.495]
eq_bern = [-0.968, 0.604, 0.406, -0.441, -0.420, 1.495]  ✅
```

### Example 2: 3D Paraboloid with Line x=0.5, y=0.5

**Hyperplanes**:
- `H1: x - 0.5 = 0` (coefficients: `[1, 0, 0]`, constant: `-0.5`)
- `H2: y - 0.5 = 0` (coefficients: `[0, 1, 0]`, constant: `-0.5`)

**Hypersurface** (paraboloid `z = x² + y²`):
- `x(u,v)` has Bernstein coefficients: `bern_x` (shape: 4×4)
- `y(u,v)` has Bernstein coefficients: `bern_y` (shape: 4×4)
- `z(u,v)` has Bernstein coefficients: `bern_z` (shape: 4×4)

**Equation polynomials**:
- `p1(u,v) = x(u,v) - 0.5`
- `p2(u,v) = y(u,v) - 0.5`

**Equation Bernstein coefficients**:
```python
eq1_bern = 1*bern_x + 0*bern_y + 0*bern_z + (-0.5)
         = bern_x - 0.5

eq2_bern = 0*bern_x + 1*bern_y + 0*bern_z + (-0.5)
         = bern_y - 0.5
```

**Verification**: ✅ Both match exactly!

## How to Access

### Method 1: Direct Access
```python
system = create_intersection_system(line, hypersurface)

# Get equation Bernstein coefficients
eq_coeffs = system['equation_bernstein_coeffs']

# For 2D (1 equation):
eq_bern = eq_coeffs[0]

# For 3D (2 equations):
eq1_bern = eq_coeffs[0]
eq2_bern = eq_coeffs[1]
```

### Method 2: Helper Function
```python
from src.intersection.polynomial_system import get_equation_bernstein_coeffs

system = create_intersection_system(line, hypersurface)
eq_coeffs = get_equation_bernstein_coeffs(system)
```

### Method 3: From Equation Specifications
```python
system = create_intersection_system(line, hypersurface)

# Each equation has its Bernstein coefficients
for i, eq in enumerate(system['equations']):
    eq_bern = eq['bernstein_coeffs']
    print(f"Equation {i+1} Bernstein coefficients: {eq_bern}")
```

## System Dictionary Structure

```python
system = {
    'n': 2,                          # Ambient dimension
    'k': 1,                          # Number of parameters
    'line': Line object,
    'hypersurface': Hypersurface object,
    'equations': [                   # List of equation specifications
        {
            'hyperplane_index': 0,
            'hyperplane_coeffs': [1.0, -1.0],  # [a_i1, ..., a_in]
            'hyperplane_d': 0.0,                # d_i
            'bernstein_coeffs': array([...]),   # ← EQUATION COEFFICIENTS
            'description': "H1: 1.0*x1(u) + -1.0*x2(u) + 0.0 = 0"
        },
        ...
    ],
    'equation_bernstein_coeffs': [   # ← EQUATION COEFFICIENTS (list)
        array([...]),  # Equation 1
        array([...]),  # Equation 2
        ...
    ],
    'hypersurface_bernstein_coeffs': [  # ← HYPERSURFACE COEFFICIENTS
        array([...]),  # x1(u)
        array([...]),  # x2(u)
        ...
    ],
    'param_ranges': [(0, 1)],
    'degree': 5
}
```

## Why This Matters for LP Method

The LP method from the 1993 Sherbrooke & Patrikalakis paper works with **equation Bernstein coefficients** to:

1. **Compute Bounding Boxes**: The convex hull property of Bernstein basis means:
   ```
   min(eq_bern) ≤ p(u) ≤ max(eq_bern)
   ```
   This gives bounds on the equation polynomial values.

2. **Formulate LP Problem**: The LP method uses these bounds to find tight bounding boxes for the solution region.

3. **Subdivision**: When subdividing the parameter space, we need to subdivide the **equation polynomials**, not just the coordinate polynomials.

4. **Convergence**: The LP method achieves quadratic convergence by working directly with equation Bernstein coefficients.

## Key Properties

### Linear Combination
The equation Bernstein coefficients are a **linear combination** of hypersurface coefficients:
```python
eq_bern = sum(a_ij * bern_xj for j in range(n)) + d_i
```

### Shape Preservation
The equation coefficients have the **same shape** as the hypersurface coefficients:
- 1D (curve): `(degree+1,)`
- 2D (surface): `(degree+1, degree+1)`
- 3D: `(degree+1, degree+1, degree+1)`

### Convex Hull Property
Since Bernstein basis has the convex hull property:
```
min(eq_bern) ≤ p(u) ≤ max(eq_bern)  for all u in parameter domain
```

This is crucial for the LP method's bounding box computation!

## Verification

The test `test_equation_bernstein_coeffs.py` verifies:

✅ **2D Circle**: `eq_bern = bern_x - bern_y` (exact match)
✅ **3D Surface**: `eq1_bern = bern_x - 0.5`, `eq2_bern = bern_y - 0.5` (exact match)
✅ **Residuals at Intersections**: All residuals are 0.0 at known intersection points

## Summary

**Before**: System only stored hypersurface Bernstein coefficients
**After**: System stores **both** hypersurface AND equation Bernstein coefficients ✅

**Access**:
```python
# Equation coefficients (what you need for LP method)
eq_coeffs = system['equation_bernstein_coeffs']

# Hypersurface coefficients (for reference)
hyp_coeffs = system['hypersurface_bernstein_coeffs']
```

**Formula**:
```python
eq_bern[i] = sum(a_ij * bern_xj for j in range(n)) + d_i
```

**Ready for LP Method**: Yes! The equation Bernstein coefficients are now available for bounding box computation and subdivision. 🎯

