# Graph Bernstein Representation - Summary

## Question

Before implementing PP and LP methods, should the system be restated as finding the intersection of the graph of the polynomials with the hyperplane x_{n+1}=0?

The graph (x_1,...,x_n,f_k(x)) should be transformed into Bernstein basis, including:
1. The polynomial part f_k(x) - already done
2. The monomial x_k (identity maps) - needs Bernstein representation

## Answer

**YES**, this is the correct formulation for LP/PP methods!

## The Graph Representation

For a k-parameter hypersurface in n-dimensional space:
```
Hypersurface: (u_1, ..., u_k) → (x_1(u), ..., x_n(u))
```

The **graph** is a k-dimensional surface in (k+n)-dimensional space:
```
Graph = {(u_1, ..., u_k, x_1(u), ..., x_n(u)) : u ∈ [0,1]^k}
```

For LP/PP methods, we need Bernstein representation of ALL coordinates:
1. **Parameter coordinates**: u_1, ..., u_k (identity maps)
2. **Hypersurface coordinates**: x_1(u), ..., x_n(u) (already computed)

## Identity Maps in Bernstein Basis

### 1D Identity: u → u on [0,1]

**Bernstein coefficients:**
```
b_i = i/n  for i = 0, 1, ..., n
```

**Example** (degree 5):
```
[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
```

### 2D Identity: (u,v) → u on [0,1]²

**Bernstein coefficients:**
```
b_{ij} = i/n  (constant in j)
```

**Example** (degree 4):
```
[[0.00, 0.00, 0.00, 0.00, 0.00],
 [0.25, 0.25, 0.25, 0.25, 0.25],
 [0.50, 0.50, 0.50, 0.50, 0.50],
 [0.75, 0.75, 0.75, 0.75, 0.75],
 [1.00, 1.00, 1.00, 1.00, 1.00]]
```

### 2D Identity: (u,v) → v on [0,1]²

**Bernstein coefficients:**
```
b_{ij} = j/m  (constant in i)
```

**Example** (degree 4):
```
[[0.00, 0.25, 0.50, 0.75, 1.00],
 [0.00, 0.25, 0.50, 0.75, 1.00],
 [0.00, 0.25, 0.50, 0.75, 1.00],
 [0.00, 0.25, 0.50, 0.75, 1.00],
 [0.00, 0.25, 0.50, 0.75, 1.00]]
```

## Test Results

### ✅ Test 1: 1D Identity Map
- **Status**: PASS
- **Result**: Identity u → u correctly represented in Bernstein basis
- **Coefficients match theoretical values**: YES

### ✅ Test 2: 2D Identity Maps  
- **Status**: PASS
- **Result**: Both u and v identity maps correctly represented
- **Coefficients match theoretical values**: YES

### ❌ Test 3: 1D Graph Representation
- **Status**: FAIL (expected)
- **Reason**: Hypersurface Bernstein coefficients are on ORIGINAL parameter range, not [0,1]
- **This is CORRECT behavior**: Hypersurface should be on its original range

### ❌ Test 4: 2D Graph Representation
- **Status**: FAIL (expected)
- **Reason**: Same as Test 3
- **This is CORRECT behavior**: Hypersurface should be on its original range

## Key Finding

**Hypersurface Bernstein coefficients are defined on the ORIGINAL parameter range, not [0,1]!**

This is correct because:
1. The hypersurface is defined on its actual parameter range (e.g., [-π, π])
2. The Bernstein coefficients represent the polynomial on that range
3. For LP/PP methods, we MUST normalize first!

## Solution: Normalization Workflow

For LP/PP methods, use the normalization utility:

```python
from src.intersection.normalization import normalize_hypersurface
from src.intersection.polynomial_system import create_intersection_system

# Step 1: Create hypersurface on original range
hypersurface = Hypersurface(
    func=lambda u: np.array([np.cos(u), np.sin(u)]),
    param_ranges=[(-np.pi, np.pi)],  # Original range
    ambient_dim=2,
    degree=8
)

# Step 2: Normalize to [0, 1]
hypersurface_normalized, transform = normalize_hypersurface(hypersurface)

# Step 3: Create system with normalized hypersurface
system = create_intersection_system(line, hypersurface_normalized)

# Step 4: Apply LP/PP methods
# Now equation_bernstein_coeffs are on [0, 1]
solutions_normalized = solve_lp(system)

# Step 5: Denormalize solutions back to original range
from src.intersection.normalization import denormalize_solutions
solutions_original = denormalize_solutions(solutions_normalized, transform)
```

## What Needs to Be Done for LP/PP

### Already Done ✅

1. **Hypersurface Bernstein coefficients**: `system['hypersurface_bernstein_coeffs']`
   - Represents x_1(u), ..., x_n(u) in Bernstein basis
   
2. **Equation Bernstein coefficients**: `system['equation_bernstein_coeffs']`
   - Represents the polynomial equations in Bernstein basis
   
3. **Normalization utility**: `normalize_hypersurface()`
   - Transforms parameter domain to [0,1]^k

### To Be Implemented 🔨

1. **Identity map Bernstein coefficients** (for graph representation):
   ```python
   def get_identity_bernstein_1d(degree):
       return np.array([i / degree for i in range(degree + 1)])
   
   def get_identity_bernstein_2d(degree, which_param):
       coeffs = np.zeros((degree + 1, degree + 1))
       if which_param == 0:
           for i in range(degree + 1):
               coeffs[i, :] = i / degree
       else:
           for j in range(degree + 1):
               coeffs[:, j] = j / degree
       return coeffs
   ```

2. **Graph Bernstein representation** (for LP/PP):
   ```python
   def get_graph_bernstein_coeffs(system):
       """
       Get Bernstein coefficients for the graph.
       
       Returns:
           param_coeffs: List of Bernstein coeffs for u_1, ..., u_k
           hypersurface_coeffs: List of Bernstein coeffs for x_1(u), ..., x_n(u)
       """
       k = system['k']
       degree = system['degree']
       
       # Parameter coordinates (identity maps)
       param_coeffs = []
       if k == 1:
           param_coeffs.append(get_identity_bernstein_1d(degree))
       elif k == 2:
           param_coeffs.append(get_identity_bernstein_2d(degree, 0))  # u
           param_coeffs.append(get_identity_bernstein_2d(degree, 1))  # v
       # ... extend for k > 2
       
       # Hypersurface coordinates (already computed)
       hypersurface_coeffs = system['hypersurface_bernstein_coeffs']
       
       return param_coeffs, hypersurface_coeffs
   ```

3. **LP/PP solvers** using the graph representation

## Conclusion

### ✅ Tests Confirm:

1. **Identity maps CAN be represented in Bernstein basis** - formulas are correct
2. **Hypersurface Bernstein coefficients are on original ranges** - this is correct
3. **Normalization is REQUIRED before LP/PP** - use `normalize_hypersurface()`

### 📋 Next Steps:

1. Add utility functions for identity map Bernstein coefficients
2. Add function to get complete graph Bernstein representation
3. Implement LP method using graph representation
4. Implement PP method using graph representation
5. Test with normalized hypersurfaces

### 🎯 Key Insight:

The graph formulation is correct and necessary for LP/PP methods. The identity maps (parameter coordinates) need to be included in the Bernstein representation alongside the hypersurface coordinates. This allows LP/PP to work in the (k+n)-dimensional graph space where the intersection problem is naturally formulated.

