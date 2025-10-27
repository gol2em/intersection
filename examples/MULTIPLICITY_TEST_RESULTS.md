# Root Multiplicity Test Results

## Summary

This document summarizes the results of testing the polynomial solver's ability to handle roots with varying multiplicities.

## Test Setup

- **Polynomial**: `(x - 0.5)^m = 0` where `m` is the multiplicity
- **Domain**: `[0, 1]`
- **Expected root**: `x = 0.5` (single root with multiplicity `m`)
- **Tolerance**: `1e-6`
- **Max depth**: Auto-calculated as `ceil(d * log_2(m)) + 5` where `d=1` (dimension)
- **Timeout**: 5 seconds per test

## Results

| Multiplicity | Status | Solutions Found | Near Root | Max Depth | Boxes Processed | Time (s) |
|--------------|--------|-----------------|-----------|-----------|-----------------|----------|
| 1            | ✅ PASS | 1               | 1         | 0         | 1               | 0.084    |
| 2            | ⚠️ DUPES | 8               | 8         | 4         | 47              | 0.004    |
| 3            | ⚠️ NEAR  | 128             | 4         | 7         | 273             | 0.021    |
| 4            | ❌ FAIL  | 128             | 0         | 7         | 273             | 0.022    |
| 5            | ❌ FAIL  | 256             | 0         | 8         | 529             | 0.047    |
| 6            | ❌ FAIL  | 32              | 0         | 8         | 77              | 0.007    |
| 7            | ❌ FAIL  | 64              | 0         | 8         | 139             | 0.012    |
| 8            | ❌ FAIL  | 64              | 0         | 8         | 137             | 0.013    |
| 9            | ❌ FAIL  | 128             | 0         | 9         | 263             | 0.029    |
| 10           | ❌ FAIL  | 256             | 0         | 9         | 517             | 0.065    |

## Analysis

### Multiplicity 1 (Simple Root)
- **Status**: ✅ **PASS**
- The solver correctly identifies the simple root at x = 0.5
- No subdivision needed (depth = 0)
- Optimal performance

### Multiplicity 2 (Double Root)
- **Status**: ⚠️ **DUPLICATES**
- Finds 8 duplicate solutions all near x = 0.5
- The polynomial is very flat near the root, causing multiple boxes to appear to contain roots
- Max depth = 4 (auto-calculated: ceil(1 * log2(2)) + 5 = 6, but only reached depth 4)

### Multiplicity 3 (Triple Root)
- **Status**: ⚠️ **NEAR**
- Finds 128 solutions, only 4 are within 10x tolerance of the true root
- The polynomial is extremely flat, making it difficult to isolate the root
- Max depth = 7 (auto-calculated: ceil(1 * log2(3)) + 5 = 7)

### Multiplicity 4+ (High Multiplicity)
- **Status**: ❌ **FAIL**
- Cannot find the root within tolerance
- The polynomial becomes so flat that the PP bounds cannot effectively prune boxes
- The solver reaches max depth without finding a solution

## Key Findings

1. **Maximum Multiplicity Handled**: **1** (simple roots only)
   - The solver reliably handles simple roots

2. **First Failure**: **Multiplicity 4**
   - At multiplicity 4 and above, the solver cannot find the root

3. **First Duplicates**: **Multiplicity 2**
   - Starting at multiplicity 2, the solver finds duplicate solutions

4. **Auto Max Depth Calculation**:
   - Formula: `ceil(d * log_2(m)) + 5`
   - For 1D polynomials: `ceil(log_2(m)) + 5`
   - Examples:
     - m=2: depth = 6
     - m=4: depth = 7
     - m=8: depth = 8
     - m=10: depth = 9
   - This prevents excessive subdivision and keeps runtime reasonable

5. **Performance**:
   - All tests complete within 0.1 seconds
   - No timeouts with auto max depth
   - The auto max depth effectively limits the search space

## Theoretical Background

### Why High Multiplicity is Difficult

For a root of multiplicity `m`, the polynomial and its first `(m-1)` derivatives all vanish at the root:
- `f(x) = 0`
- `f'(x) = 0`
- `f''(x) = 0`
- ...
- `f^(m-1)(x) = 0`

This makes the polynomial extremely flat near the root, which causes problems for subdivision methods:

1. **Bernstein Bounds**: The convex hull of Bernstein coefficients becomes very tight to zero over large regions
2. **Subdivision Ineffective**: Subdividing doesn't significantly improve the bounds
3. **Multiple Boxes**: Many boxes appear to contain roots (all are actually the same root)

### Bezout's Theorem and Max Depth

For a polynomial system with maximum degree `m` in dimension `d`:
- **Maximum number of roots**: `m^d` (by Bezout's theorem)
- **Bisection strategy**: Each bisection along one dimension doubles the number of boxes
- **Depth needed**: To separate `m^d` roots, we need at most `log_2(m^d) = d * log_2(m)` bisections

The auto max depth calculation uses this theoretical bound plus a buffer of 5 levels for numerical issues.

## Recommendations

1. **For Simple Roots**: The solver works excellently
2. **For Multiple Roots**: 
   - Multiplicity 2: Expect duplicates, use duplicate removal
   - Multiplicity 3+: The solver will likely fail
   - Consider using specialized methods for high-multiplicity roots (e.g., GCD-based methods)
3. **Max Depth**: Use automatic calculation (`max_depth=None`) for optimal performance
4. **Tolerance**: For high-multiplicity roots, larger tolerance may help but will reduce accuracy

## Conclusion

The subdivision-based polynomial solver with PP bounding is:
- ✅ **Excellent** for simple roots (multiplicity 1)
- ⚠️ **Limited** for double roots (multiplicity 2) - finds duplicates
- ❌ **Not suitable** for high-multiplicity roots (multiplicity 3+)

The automatic max depth calculation based on `ceil(d * log_2(m)) + 5` effectively prevents excessive subdivision while allowing the solver to find all theoretically possible simple roots.

