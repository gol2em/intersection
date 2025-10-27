# Box and Domain Tracking - Summary

## Your Question

"I need to keep track of the original domain to convert the final result back. And during subdivision of LP/PP, I also need to keep track of the original domain since de Casteljau algorithm keeps the new domain [0,1]. How do I do that?"

## Answer: Box Class with Multi-Level Domain Tracking

### ✅ Solution Implemented

Created `src/intersection/box.py` with a `Box` class that tracks transformations at multiple levels:

```
Original Domain → Normalized [0,1]^k → Current Box → Bernstein [0,1]^k
    (e.g., [-π,π])      (normalization)    (subdivision)   (de Casteljau)
```

### Domain Hierarchy

1. **Original Domain**: Actual parameter ranges of hypersurface
   - Example: `u ∈ [-π, π]`
   
2. **Normalized Domain**: Always `[0,1]^k` after normalization
   - Example: `t ∈ [0, 1]` where `t = (u + π)/(2π)`
   
3. **Current Box**: Sub-box of `[0,1]^k` after subdivision
   - Example: `t ∈ [0.25, 0.5]` (a sub-box)
   
4. **Bernstein Domain**: Always `[0,1]^k` for de Casteljau
   - Example: `s ∈ [0, 1]` where `s = (t - 0.25)/(0.5 - 0.25)`

### Key Transformations

The `Box` class provides methods for all transformations:

```python
# Create box with normalization transform
box = Box(
    k=1,
    ranges=[(0.0, 1.0)],  # Normalized space
    normalization_transform=transform  # From normalize_hypersurface()
)

# Transformations
bernstein_params = box.box_to_bernstein(*box_params)
box_params = box.bernstein_to_box(*bernstein_params)
original_params = box.normalized_to_original(*normalized_params)
normalized_params = box.original_to_normalized(*original_params)

# Composition: Bernstein → Original (most important for LP/PP)
original_params = box.bernstein_to_original(*bernstein_params)
```

### Subdivision Workflow

```python
# Initial box on [0, 1]
box = Box(k=1, ranges=[(0.0, 1.0)], normalization_transform=transform)

# Subdivide
left_box, right_box = box.subdivide(axis=0, split_point=0.5)

# Left box: [0, 0.5] in normalized space
# Right box: [0.5, 1.0] in normalized space

# de Casteljau gives coefficients on [0, 1]^k
# Use left_box.bernstein_to_original() to map back to original space
```

### Test Results

✅ **All Box tests PASS** (`tests/test_box.py`):
- Basic box operations
- Subdivision
- Normalization transforms
- 1D and 2D subdivision with normalization
- All domain transformations work correctly

### LP/PP Workflow

```python
# Step 1: Create hypersurface on original domain
hypersurface = Hypersurface(
    func=lambda u: np.array([np.cos(u), np.sin(u)]),
    param_ranges=[(-np.pi, np.pi)],
    ambient_dim=2,
    degree=8
)

# Step 2: Normalize to [0, 1]
hypersurface_normalized, transform = normalize_hypersurface(hypersurface)

# Step 3: Create initial box
box = Box(k=1, ranges=[(0.0, 1.0)], normalization_transform=transform)

# Step 4: LP/PP subdivision loop
queue = [(box, bernstein_coeffs)]
while queue:
    current_box, coeffs = queue.pop()
    
    # Check if box contains solution (using PP or LP bounds)
    if not contains_solution(coeffs):
        continue
    
    # If small enough, record solution
    if current_box.get_volume() < tolerance:
        # Map center from Bernstein to original space
        center_bernstein = np.array([0.5] * k)
        center_original = current_box.bernstein_to_original(*center_bernstein)
        solutions.append(center_original)
        continue
    
    # Subdivide
    left_box, right_box = current_box.subdivide(axis=0)
    
    # Apply de Casteljau to get coefficients for sub-boxes
    left_coeffs = de_casteljau_subdivide(coeffs, left_box)
    right_coeffs = de_casteljau_subdivide(coeffs, right_box)
    
    queue.append((left_box, left_coeffs))
    queue.append((right_box, right_coeffs))

# Solutions are already in original space!
```

## Critical Bug Discovered

### ❌ Power-to-Bernstein Conversion is BROKEN

While testing the Box class, we discovered that `_polynomial_1d_to_bernstein()` in `src/intersection/bernstein.py` is **completely broken**:

**Test Results:**
```
p(t) = t^2:
  Expected Bernstein: [0, 0.5, 1]
  Got:                [0, 0, 1]  ❌ WRONG!

p(t) = (1-t)^2:
  Expected Bernstein: [1, 0, 0]
  Got:                [1, -1, 1]  ❌ WRONG!
```

**Impact:**
- All Bernstein coefficients computed by the system are INCORRECT
- LP/PP methods will NOT work until this is fixed
- Graph representation tests fail because of this bug

**Root Cause:**
The conversion matrix implementation in lines 58-68 of `src/intersection/bernstein.py` is incorrect.

**Correct Formula:**
For power basis `p(t) = Σ a_i t^i` to Bernstein basis `p(t) = Σ b_j B_j^n(t)`:

```
b_j = Σ_{i=j}^{n} a_i * C(i,j) * C(n-i, n-j) / C(n,j)
```

Where `C(n,k)` is the binomial coefficient.

## Next Steps

### 1. Fix Power-to-Bernstein Conversion (CRITICAL)

**Priority: HIGHEST**

The conversion function must be fixed before any LP/PP work can proceed.

**Action Items:**
- Research correct conversion formula
- Implement correct conversion matrix
- Test with known polynomials:
  - `t` → `[0, 1]`
  - `t^2` → `[0, 0.5, 1]`
  - `(1-t)^2` → `[1, 0, 0]`
  - `2t(1-t)` → `[0, 1, 0]`

### 2. Update Graph Tests

Once conversion is fixed, Test 5 should pass, showing the correct workflow:
- Normalize hypersurface
- Create Box with transform
- Get Bernstein coefficients
- Evaluate using Box transformations
- Subdivide and track domains

### 3. Implement de Casteljau Subdivision

**Status:** Not started (blocked by conversion bug)

**Requirements:**
- `de_casteljau_subdivide_1d(coeffs, box)`: Subdivide 1D Bernstein polynomial
- `de_casteljau_subdivide_2d(coeffs, box)`: Subdivide 2D tensor product
- `de_casteljau_subdivide_kd(coeffs, box)`: General k-D case

### 4. Implement LP/PP Methods

**Status:** Not started (blocked by conversion bug)

**Requirements:**
- Use Box class for domain tracking
- Use de Casteljau for subdivision
- Return solutions in original space using `box.bernstein_to_original()`

## Files Created

### ✅ Implemented and Tested

- `src/intersection/box.py` - Box class with multi-level domain tracking
- `tests/test_box.py` - Comprehensive tests (ALL PASS)

### 📋 Diagnostic Tests

- `tests/test_graph_bernstein.py` - Graph representation tests
  - Tests 1-2: PASS (identity maps)
  - Tests 3-4: FAIL (expected - needs normalization)
  - Test 5: FAIL (blocked by conversion bug)

- `tests/test_bernstein_conversion.py` - Power-to-Bernstein diagnostic
  - Shows conversion is broken

- `tests/test_hypersurface_identity.py` - Hypersurface diagnostic
  - Shows interpolation works but conversion fails

### 📄 Documentation

- `BOX_AND_DOMAIN_TRACKING_SUMMARY.md` - This file
- `GRAPH_BERNSTEIN_SUMMARY.md` - Graph representation explanation

## Conclusion

### ✅ Your Question is ANSWERED

**How to track domains during LP/PP subdivision?**

Use the `Box` class:
1. Create box with normalization transform
2. Subdivide to get sub-boxes
3. Use `box.bernstein_to_original()` to map solutions back
4. de Casteljau works on [0,1]^k, Box handles all transformations

### ❌ Critical Blocker

**Power-to-Bernstein conversion is broken** and must be fixed before LP/PP can be implemented.

### 🎯 Immediate Action

**Fix `_polynomial_1d_to_bernstein()` in `src/intersection/bernstein.py`**

This is the only blocker preventing LP/PP implementation. Once fixed:
- Graph tests will pass
- Bernstein coefficients will be correct
- LP/PP methods can be implemented
- de Casteljau subdivision can be added

The Box class is ready and tested. The workflow is clear. Only the conversion bug stands in the way.

