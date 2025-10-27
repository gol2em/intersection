# Tolerance Classification for PP Method

## Overview

This document classifies all tolerance parameters used in the PP method and subdivision solver. Understanding these tolerances is critical for managing numerical precision and performance.

## Tolerance Hierarchy

```
User-Level Tolerances (what you control)
├── tolerance (solution_tolerance)
│   └── Controls: When to report a solution
│
├── crit (critical_ratio)
│   └── Controls: When to subdivide vs. tighten
│
└── max_depth
    └── Controls: Maximum recursion depth

Internal Tolerances (implementation details)
├── subdivision_tolerance
│   └── Controls: Numerical precision in PP method
│
└── convex_hull_tolerance
    └── Controls: Zero-crossing detection
```

---

## 1. User-Level Tolerances

### 1.1 `tolerance` (Solution Tolerance)
**Default**: `1e-6`  
**Location**: `SolverConfig.tolerance`, `solve_polynomial_system(tolerance=...)`  
**Purpose**: Determines when a box is small enough to report as containing a solution  
**Units**: Absolute size in normalized [0,1]^k space  

**How it works**:
- After PP method tightens bounds to [t_min, t_max], check if `(t_max - t_min) < tolerance`
- If yes → report midpoint as solution
- If no → subdivide further

**Relationship to coefficient magnitude**:
- ❌ **WRONG**: `tolerance` should NOT depend on coefficient magnitude
- ✅ **CORRECT**: `tolerance` controls solution precision in parameter space
- Coefficients of 10^-6 are fine with tolerance=10^-6 because:
  - Coefficients are in function value space
  - Tolerance is in parameter space
  - These are independent!

**Recommended values**:
- High precision: `1e-8` to `1e-10`
- Standard: `1e-6` (default)
- Fast/approximate: `1e-4` to `1e-3`

**Example**:
```python
# Find roots to within 1 micron in parameter space
solutions = solve_polynomial_system(system, tolerance=1e-6)
```

---

### 1.2 `crit` (Critical Ratio)
**Default**: `0.8`  
**Location**: `SolverConfig.crit`, `solve_polynomial_system(crit=...)`  
**Purpose**: Determines when to subdivide a dimension  
**Units**: Ratio (0.0 to 1.0)  

**How it works**:
- After PP method finds bounds [t_min, t_max] for a dimension
- Compute size: `size = t_max - t_min`
- If `size > crit` → subdivide this dimension
- If `size ≤ crit` → don't subdivide (PP tightening is working well)

**Interpretation**:
- `crit = 0.8` means "subdivide if PP didn't reduce the box by at least 20%"
- `crit = 0.5` means "subdivide if PP didn't reduce the box by at least 50%"
- `crit = 1.0` means "always subdivide" (disables PP tightening benefit)
- `crit = 0.0` means "never subdivide" (infinite PP tightening loop)

**Recommended values**:
- Aggressive subdivision: `0.9` to `1.0`
- Balanced (default): `0.8`
- Trust PP method: `0.5` to `0.7`

**Example**:
```python
# Subdivide only if PP didn't reduce box by at least 50%
solutions = solve_polynomial_system(system, crit=0.5)
```

---

### 1.3 `max_depth`
**Default**: `30`  
**Location**: `SolverConfig.max_depth`, `solve_polynomial_system(max_depth=...)`  
**Purpose**: Prevents infinite recursion  
**Units**: Integer (subdivision depth)  

**How it works**:
- Each subdivision increases depth by 1
- If `depth >= max_depth` → stop subdividing, report current box center as solution
- Acts as a safety valve

**Relationship to tolerance**:
- For tolerance `ε` in 1D, need depth ≈ `log₂(1/ε)`
- For `ε = 1e-6`: depth ≈ 20
- For `ε = 1e-10`: depth ≈ 33

**Recommended values**:
- Fast/approximate: `20`
- Standard: `30` (default)
- High precision: `40` to `50`

**Example**:
```python
# Limit to 20 subdivisions for speed
solutions = solve_polynomial_system(system, max_depth=20)
```

---

## 2. Internal Tolerances

### 2.1 `subdivision_tolerance`
**Default**: `1e-10`  
**Location**: `SolverConfig.subdivision_tolerance`  
**Purpose**: Numerical tolerance for PP method convex hull intersection  
**Units**: Absolute value in function space  

**How it works**:
- Passed to `find_root_box_pp_nd(tolerance=subdivision_tolerance)`
- Used to determine if convex hull crosses x-axis
- If all control points satisfy `|b_i| > subdivision_tolerance` with same sign → no roots

**Why it's separate from `tolerance`**:
- `tolerance` is in parameter space (controls solution precision)
- `subdivision_tolerance` is in function value space (controls zero detection)
- These are different units!

**Critical insight**:
- ✅ **Coefficients of 10^-6 work fine** with `subdivision_tolerance = 1e-10`
- The tolerance checks if `|coefficient| < 1e-10` to consider it "zero"
- Coefficients of 10^-6 are 10,000× larger than the tolerance
- This is perfectly safe with double precision (epsilon ≈ 2.2e-16)

**Recommended values**:
- Standard: `1e-10` (default)
- High precision: `1e-12` to `1e-14`
- Never go below `1e-14` (approaching double precision limit)

---

### 2.2 `convex_hull_tolerance` (in `intersect_convex_hull_with_x_axis`)
**Default**: `1e-10`  
**Location**: `convex_hull.py`, parameter `tolerance`  
**Purpose**: Determines if a point is "on" the x-axis  
**Units**: Absolute value in function space  

**How it works**:
- When checking if convex hull crosses x-axis
- Point is "on" x-axis if `|y| < tolerance`
- Used to detect sign changes

**Same as `subdivision_tolerance`**:
- These are the same tolerance, just passed through
- Both control zero-crossing detection in function value space

---

## 3. Why 10^-6 Coefficients Work Fine

### The Confusion
❌ **Wrong thinking**: "Coefficients are 10^-6, so I need tolerance ≥ 10^-6"

✅ **Correct thinking**: "Coefficients are in function value space, tolerance is in parameter space"

### The Math
Given polynomial with Bernstein coefficients `b_i ≈ 10^-6`:

1. **Function value space**: Coefficients are `b_i ≈ 10^-6`
2. **Zero detection**: Check if `|b_i| < 1e-10` → NO (10^-6 >> 10^-10)
3. **Convex hull**: Compute hull of points `(i/n, b_i)`
4. **Intersection**: Find where hull crosses y=0
5. **Parameter space**: Get bounds `[t_min, t_max]` in [0,1]
6. **Solution check**: Is `(t_max - t_min) < tolerance`? → This is in parameter space!

### The Problem (Why 20-root test failed)
The issue is NOT that coefficients are too small. The issue is:

1. **No pruning**: Convex hull contains x-axis everywhere
   - All coefficients have mixed signs
   - Hull is very flat (coefficients ≈ 10^-6)
   - Hull crosses x-axis across entire [0,1] domain
   - PP method returns [0, 1] → no tightening!

2. **Exponential explosion**: Without pruning
   - Every box subdivides into 2 children
   - No boxes are eliminated
   - Depth 20 → 2^20 = 1,048,576 boxes!

### The Solution
**Option 1**: Increase `subdivision_tolerance` to match coefficient scale
```python
# If coefficients are ~10^-6, use tolerance ~10^-7
config = SolverConfig(
    tolerance=1e-6,              # Solution precision (parameter space)
    subdivision_tolerance=1e-7,  # Zero detection (function value space)
)
```

**Option 2**: Normalize coefficients before solving
```python
# Scale coefficients to have max magnitude ≈ 1
max_coeff = np.max(np.abs(bern_coeffs))
normalized_coeffs = bern_coeffs / max_coeff
```

**Option 3**: Use different method for high-degree polynomials
- Eigenvalue method (companion matrix)
- Sturm sequences
- Interval Newton method

---

## 4. Recommended Settings

### For Low-Degree Polynomials (degree ≤ 5)
```python
solve_polynomial_system(
    system,
    tolerance=1e-6,              # Standard precision
    crit=0.8,                    # Balanced
    max_depth=30,                # Plenty of depth
    subdivision_tolerance=1e-10  # Default (internal)
)
```

### For Medium-Degree Polynomials (degree 6-10)
```python
solve_polynomial_system(
    system,
    tolerance=1e-5,              # Slightly relaxed
    crit=0.7,                    # Trust PP more
    max_depth=40,                # More depth
    subdivision_tolerance=1e-9   # Slightly relaxed
)
```

### For High-Degree Polynomials (degree > 10)
```python
# Option 1: Relax tolerances significantly
solve_polynomial_system(
    system,
    tolerance=1e-4,              # Relaxed precision
    crit=0.5,                    # Trust PP heavily
    max_depth=50,                # Maximum depth
    subdivision_tolerance=1e-7   # Match coefficient scale
)

# Option 2: Use different method (recommended)
# Use eigenvalue method or other specialized root finder
```

---

## 5. Quick Reference Table

| Tolerance | Default | Units | Controls | Adjust When |
|-----------|---------|-------|----------|-------------|
| `tolerance` | 1e-6 | Parameter space | Solution precision | Need more/less precision |
| `crit` | 0.8 | Ratio | Subdivision trigger | PP not tightening well |
| `max_depth` | 30 | Integer | Recursion limit | Hitting depth limit |
| `subdivision_tolerance` | 1e-10 | Function space | Zero detection | Coefficients very small |

---

## 6. Debugging Checklist

If solver is slow or failing:

1. ✅ Check coefficient magnitude: `np.max(np.abs(coeffs))`
   - If < 1e-8 → increase `subdivision_tolerance` or normalize

2. ✅ Check pruning rate: Look for "pruned X boxes" in output
   - If 0% pruned → PP method not working, adjust `subdivision_tolerance`

3. ✅ Check depth: Look for "max depth reached"
   - If hitting limit → increase `max_depth` or relax `tolerance`

4. ✅ Check box count: Look for "processed X boxes"
   - If > 10,000 → problem is too hard for PP method

5. ✅ Check polynomial degree
   - If > 10 → consider different method

