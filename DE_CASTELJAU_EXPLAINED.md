# de Casteljau Subdivision Algorithm

## Overview

The de Casteljau algorithm is a recursive method for:
1. **Evaluating** Bernstein polynomials at a point
2. **Subdividing** Bernstein polynomials into smaller pieces

It's the foundation for subdivision methods in LP/PP algorithms.

## Mathematical Background

### Bernstein Polynomial

A polynomial of degree n in Bernstein form on [0,1]:

```
p(t) = Σ(i=0 to n) b_i * B_i^n(t)
```

Where:
- `b_i` are Bernstein coefficients
- `B_i^n(t) = C(n,i) * t^i * (1-t)^(n-i)` are Bernstein basis functions

### Key Property: Recursive Definition

Bernstein basis functions satisfy:

```
B_i^n(t) = (1-t) * B_i^(n-1)(t) + t * B_(i-1)^(n-1)(t)
```

This leads to the de Casteljau algorithm!

## The Algorithm (1D Case)

### Evaluation at Point t

Given Bernstein coefficients `b_0, b_1, ..., b_n`, evaluate `p(t)`:

**Algorithm:**
```
1. Start with coefficients: b_i^0 = b_i  (i = 0, 1, ..., n)

2. For j = 1 to n:
     For i = 0 to n-j:
       b_i^j = (1-t) * b_i^(j-1) + t * b_(i+1)^(j-1)

3. Result: p(t) = b_0^n
```

**Geometric Interpretation:**
- Each step performs linear interpolation between adjacent coefficients
- After n steps, you get the polynomial value

### Example: Degree 2 Polynomial

Given: `b_0 = 1, b_1 = 3, b_2 = 2`, evaluate at `t = 0.5`

```
Level 0:  b_0^0 = 1    b_1^0 = 3    b_2^0 = 2

Level 1:  b_0^1 = (1-0.5)*1 + 0.5*3 = 2
          b_1^1 = (1-0.5)*3 + 0.5*2 = 2.5

Level 2:  b_0^2 = (1-0.5)*2 + 0.5*2.5 = 2.25

Result: p(0.5) = 2.25
```

### Subdivision at Point t

The de Casteljau algorithm also gives us the Bernstein coefficients for the subdivided pieces!

**Left piece [0, t]:**
```
Coefficients: b_0^0, b_0^1, b_0^2, ..., b_0^n
```

**Right piece [t, 1]:**
```
Coefficients: b_0^n, b_1^(n-1), b_2^(n-2), ..., b_n^0
```

### Example: Subdivide at t = 0.5

Given: `b_0 = 1, b_1 = 3, b_2 = 2`

**de Casteljau pyramid:**
```
        b_0^2 = 2.25
       /            \
    b_0^1 = 2      b_1^1 = 2.5
   /        \      /         \
b_0^0 = 1  b_1^0 = 3  b_2^0 = 2
```

**Left piece [0, 0.5]:**
- Coefficients: `[1, 2, 2.25]`
- These are the LEFT diagonal: `b_0^0, b_0^1, b_0^2`

**Right piece [0.5, 1]:**
- Coefficients: `[2.25, 2.5, 2]`
- These are the RIGHT diagonal: `b_0^2, b_1^1, b_2^0`

## Why This Works

### Key Insight: Renormalization

When we subdivide at `t`, the left piece represents `p(t)` on `[0, t]`, but the new coefficients are **for the Bernstein basis on [0,1]**!

Mathematically:
- Original: `p(t)` on `[0, 1]`
- Left piece: `p(t*s)` on `[0, 1]` where `s ∈ [0, 1]`
- Right piece: `p(t + (1-t)*s)` on `[0, 1]` where `s ∈ [0, 1]`

This is **exactly** the renormalization we need for LP/PP methods!

## Subdivision to Arbitrary Interval [a, b]

To get Bernstein coefficients on `[a, b] ⊂ [0, 1]`:

**Two-step process:**
1. Subdivide at `t = a` to get right piece `[a, 1]`
2. Subdivide that piece at `t' = (b-a)/(1-a)` to get left piece `[a, b]`

**Example:** Get coefficients on `[0.25, 0.75]`

```
Step 1: Subdivide at t = 0.25
  → Get right piece [0.25, 1]

Step 2: Subdivide at t' = (0.75-0.25)/(1-0.25) = 0.5/0.75 = 2/3
  → Get left piece [0.25, 0.75]
```

## Implementation (1D)

### Evaluation

```python
def de_casteljau_eval(coeffs, t):
    """Evaluate Bernstein polynomial at t using de Casteljau."""
    n = len(coeffs) - 1
    b = coeffs.copy()
    
    for j in range(1, n + 1):
        for i in range(n - j + 1):
            b[i] = (1 - t) * b[i] + t * b[i + 1]
    
    return b[0]
```

### Subdivision at t

```python
def de_casteljau_subdivide(coeffs, t):
    """
    Subdivide Bernstein polynomial at t.
    
    Returns:
        left_coeffs: Coefficients on [0, t]
        right_coeffs: Coefficients on [t, 1]
    """
    n = len(coeffs) - 1
    
    # Build de Casteljau pyramid
    pyramid = [coeffs.copy()]
    
    for j in range(1, n + 1):
        level = []
        prev_level = pyramid[-1]
        for i in range(n - j + 1):
            val = (1 - t) * prev_level[i] + t * prev_level[i + 1]
            level.append(val)
        pyramid.append(level)
    
    # Extract left coefficients (left diagonal)
    left_coeffs = [pyramid[j][0] for j in range(n + 1)]
    
    # Extract right coefficients (right diagonal)
    right_coeffs = [pyramid[n - j][j] for j in range(n + 1)]
    
    return left_coeffs, right_coeffs
```

### Subdivision to [a, b]

```python
def de_casteljau_interval(coeffs, a, b):
    """
    Get Bernstein coefficients on interval [a, b] ⊂ [0, 1].
    """
    # Step 1: Subdivide at a, take right piece
    _, right_coeffs = de_casteljau_subdivide(coeffs, a)
    
    # Step 2: Subdivide at (b-a)/(1-a), take left piece
    t_relative = (b - a) / (1 - a) if a < 1 else 0
    left_coeffs, _ = de_casteljau_subdivide(right_coeffs, t_relative)
    
    return left_coeffs
```

## Multi-Dimensional Case (2D)

For tensor product Bernstein polynomials:

```
p(u, v) = Σ Σ b_ij * B_i^n(u) * B_j^m(v)
```

**Subdivision strategy:**
1. Apply de Casteljau along u-direction for each v-slice
2. Apply de Casteljau along v-direction for each u-slice

### Implementation (2D)

```python
def de_casteljau_2d(coeffs_2d, u_interval, v_interval):
    """
    Subdivide 2D Bernstein polynomial.
    
    coeffs_2d: (n+1) × (m+1) array
    u_interval: (u_a, u_b)
    v_interval: (v_a, v_b)
    """
    n_u, n_v = coeffs_2d.shape
    
    # Step 1: Subdivide along u-direction
    temp = np.zeros_like(coeffs_2d)
    for j in range(n_v):
        temp[:, j] = de_casteljau_interval(coeffs_2d[:, j], u_interval[0], u_interval[1])
    
    # Step 2: Subdivide along v-direction
    result = np.zeros_like(coeffs_2d)
    for i in range(n_u):
        result[i, :] = de_casteljau_interval(temp[i, :], v_interval[0], v_interval[1])
    
    return result
```

## Connection to LP/PP Methods

### Why de Casteljau is Perfect for Subdivision

1. **Automatic Renormalization**
   - Subdivided coefficients are on [0,1]
   - No explicit renormalization needed!

2. **Preserves Convex Hull Property**
   - Subdivided coefficients still bound the polynomial
   - LP/PP methods can use them directly

3. **Numerically Stable**
   - Only uses convex combinations
   - No matrix inversions or polynomial root finding

### Workflow in LP/PP

```
Initial: Bernstein coeffs on [0, 1]
    ↓
Apply LP/PP to get bounds
    ↓
Subdivide at t = 0.5
    ↓
de Casteljau gives:
  - Left coeffs on [0, 0.5] (implicitly renormalized to [0,1])
  - Right coeffs on [0.5, 1] (implicitly renormalized to [0,1])
    ↓
Apply LP/PP to each sub-box
    ↓
Recurse...
```

## Visual Example

### Original Polynomial on [0, 1]

```
Coefficients: [1, 3, 2]

     3 •
      /  \
     /    \
    /      \
   /        \
  •          • 2
 1
```

### After Subdivision at t = 0.5

**Left piece [0, 0.5]:**
```
Coefficients: [1, 2, 2.25]

     2.25 •
         /
        /
       /
      • 2
     /
    /
   • 1
```

**Right piece [0.5, 1]:**
```
Coefficients: [2.25, 2.5, 2]

     2.5 •
        /  \
       /    \
      /      \
     •        • 2
   2.25
```

Notice: Both pieces have coefficients that represent the polynomial on [0,1] (renormalized)!

## Summary

### What de Casteljau Does

1. ✅ **Evaluates** Bernstein polynomials efficiently
2. ✅ **Subdivides** polynomials into smaller pieces
3. ✅ **Renormalizes** automatically to [0,1]
4. ✅ **Preserves** convex hull property
5. ✅ **Numerically stable** (only convex combinations)

### Why It's Perfect for LP/PP

- No explicit renormalization needed
- Subdivided coefficients ready for LP/PP
- Works in any dimension (1D, 2D, kD)
- Efficient and stable

### Key Formulas

**Evaluation:**
```
b_i^j = (1-t) * b_i^(j-1) + t * b_(i+1)^(j-1)
```

**Subdivision at t:**
- Left: `[b_0^0, b_0^1, ..., b_0^n]`
- Right: `[b_0^n, b_1^(n-1), ..., b_n^0]`

**Multi-dimensional:**
- Apply de Casteljau along each dimension separately

