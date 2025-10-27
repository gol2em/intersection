# Projected Polyhedron (PP) Algorithm

## Overview

The **Projected Polyhedron (PP)** method is a subdivision algorithm for finding roots of polynomial systems represented in Bernstein form. It was developed by Sherbrooke and Patrikalakis (1993).

**Key Idea:** Use the **convex hull property** of Bernstein polynomials to bound the range of the polynomial over a box, then subdivide boxes that cannot be excluded.

## Mathematical Foundation

### Bernstein Polynomial on [0,1]

A polynomial in Bernstein form on [0,1]:

```
p(t) = Σ(i=0 to n) b_i · B_i^n(t)
```

Where:
- `b_i` are Bernstein coefficients
- `B_i^n(t) = C(n,i) · t^i · (1-t)^(n-i)` are Bernstein basis functions

### Convex Hull Property

**Theorem:** For t ∈ [0, 1]:

```
min(b_0, b_1, ..., b_n) ≤ p(t) ≤ max(b_0, b_1, ..., b_n)
```

**This is the foundation of the PP method!**

The polynomial is bounded by the **convex hull** of its Bernstein coefficients.

## The PP Algorithm (1D Case)

### Problem

Find all roots of `p(t) = 0` for t ∈ [0, 1], where p is given in Bernstein form.

### Algorithm

```
Input: Bernstein coefficients b_0, b_1, ..., b_n
Output: List of intervals containing roots

1. Initialize: queue = [(0, 1, coeffs)]

2. While queue is not empty:
   
   a. Pop (a, b, coeffs) from queue
   
   b. Compute bounds using PP method:
      p_min = min(coeffs)
      p_max = max(coeffs)
   
   c. Exclusion test:
      - If p_min > 0 or p_max < 0:
        → No root in [a, b], discard box
      - If 0 ∈ [p_min, p_max]:
        → Possible root, continue
   
   d. Convergence test:
      - If (b - a) < tolerance:
        → Root found in [a, b], add to solutions
      - Else:
        → Subdivide
   
   e. Subdivision:
      - Subdivide at t = (a + b) / 2
      - Use de Casteljau to get:
        * left_coeffs on [a, (a+b)/2]
        * right_coeffs on [(a+b)/2, b]
      - Add both to queue

3. Return solutions
```

### Key Steps Explained

#### Step 1: Bounding

```python
p_min = min(coeffs)
p_max = max(coeffs)
```

**Interpretation:** The polynomial p(t) is guaranteed to lie in [p_min, p_max] for all t ∈ [0, 1].

#### Step 2: Exclusion

```python
if p_min > 0:
    # Polynomial is always positive → no roots
    discard_box()

if p_max < 0:
    # Polynomial is always negative → no roots
    discard_box()
```

**This is the power of PP:** We can exclude boxes without evaluating the polynomial!

#### Step 3: Subdivision

```python
# Subdivide at midpoint
t_mid = 0.5

# Use de Casteljau to get renormalized coefficients
left_coeffs, right_coeffs = de_casteljau_subdivide(coeffs, t_mid)

# Recurse on both sub-boxes
process(left_coeffs)   # Represents [a, (a+b)/2]
process(right_coeffs)  # Represents [(a+b)/2, b]
```

**Key:** de Casteljau gives coefficients that are automatically renormalized to [0, 1]!

## Multi-Dimensional Case (k Variables)

### Problem

Find all solutions to the system:

```
f_1(u_1, ..., u_k) = 0
f_2(u_1, ..., u_k) = 0
...
f_m(u_1, ..., u_k) = 0
```

Where each f_i is a polynomial in Bernstein form on [0, 1]^k.

### Algorithm

```
Input: Bernstein coefficients for each equation
Output: List of boxes containing solutions

1. Initialize: queue = [Box([0,1]^k, all_coeffs)]

2. While queue is not empty:
   
   a. Pop (box, coeffs) from queue
   
   b. For each equation i:
      - Compute bounds: [p_min_i, p_max_i] = [min(coeffs_i), max(coeffs_i)]
   
   c. Exclusion test:
      - If ANY equation has p_min_i > 0 or p_max_i < 0:
        → No solution in box, discard
      - Else:
        → Possible solution, continue
   
   d. Convergence test:
      - If box.width < tolerance:
        → Solution found, add to solutions
      - Else:
        → Subdivide
   
   e. Subdivision:
      - Choose dimension to subdivide (e.g., widest dimension)
      - Subdivide box at midpoint
      - Use de Casteljau to get coefficients for sub-boxes
      - Add sub-boxes to queue

3. Return solutions
```

### Key Differences from 1D

1. **Multiple equations:** Must check ALL equations for exclusion
2. **Multi-dimensional subdivision:** Choose which dimension to subdivide
3. **Tensor product Bernstein:** Coefficients are multi-dimensional arrays

## Example (1D)

### Problem

Find roots of `p(t) = t^2 - 0.5` on [0, 1].

**Bernstein form (degree 2):**
```
p(t) = b_0·B_0^2(t) + b_1·B_1^2(t) + b_2·B_2^2(t)
```

**Coefficients:** `b_0 = -0.5, b_1 = 0, b_2 = 0.5`

### Iteration 1: Box [0, 1]

```
Coeffs: [-0.5, 0, 0.5]
Bounds: p_min = -0.5, p_max = 0.5
Exclusion: 0 ∈ [-0.5, 0.5] → Cannot exclude
Subdivide at t = 0.5
```

**de Casteljau subdivision:**
- Left [0, 0.5]: coeffs = [-0.5, -0.25, -0.125]
- Right [0.5, 1]: coeffs = [-0.125, 0.125, 0.5]

### Iteration 2a: Box [0, 0.5]

```
Coeffs: [-0.5, -0.25, -0.125]
Bounds: p_min = -0.5, p_max = -0.125
Exclusion: p_max < 0 → All negative, no roots
DISCARD
```

### Iteration 2b: Box [0.5, 1]

```
Coeffs: [-0.125, 0.125, 0.5]
Bounds: p_min = -0.125, p_max = 0.5
Exclusion: 0 ∈ [-0.125, 0.5] → Cannot exclude
Subdivide at t = 0.5 (relative to [0.5, 1])
```

**Continue subdividing until convergence...**

**Final result:** Root at t ≈ 0.707 (which is √0.5)

## Comparison: PP vs LP

### PP Method (Simpler)

**Bounding:**
```
p_min = min(coeffs)
p_max = max(coeffs)
```

**Pros:**
- ✅ Very simple to implement
- ✅ Fast computation (just min/max)
- ✅ No external dependencies

**Cons:**
- ❌ Loose bounds (not tight)
- ❌ Slower convergence
- ❌ More subdivisions needed

### LP Method (Better)

**Bounding:**
```
Solve linear program:
  minimize/maximize: Σ b_i · λ_i
  subject to: Σ λ_i = 1, λ_i ≥ 0
```

**Pros:**
- ✅ Tighter bounds
- ✅ Faster convergence (quadratic)
- ✅ Fewer subdivisions

**Cons:**
- ❌ More complex to implement
- ❌ Requires LP solver
- ❌ Slower per iteration

### When to Use Which?

| Scenario | Recommended Method |
|----------|-------------------|
| Quick prototype | PP |
| Simple problems | PP |
| High accuracy needed | LP |
| Many variables | LP |
| Production code | LP |

## Implementation Pseudocode

### 1D PP Solver

```python
def solve_pp_1d(coeffs, tolerance=1e-6, max_depth=20):
    """
    Find roots of polynomial using PP method.
    
    Parameters
    ----------
    coeffs : array
        Bernstein coefficients on [0, 1]
    tolerance : float
        Convergence tolerance
    max_depth : int
        Maximum subdivision depth
        
    Returns
    -------
    list of intervals containing roots
    """
    solutions = []
    queue = [(0.0, 1.0, coeffs, 0)]  # (a, b, coeffs, depth)
    
    while queue:
        a, b, c, depth = queue.pop(0)
        
        # Compute bounds using PP
        p_min = np.min(c)
        p_max = np.max(c)
        
        # Exclusion test
        if p_min > 0 or p_max < 0:
            continue  # No root in this box
        
        # Convergence test
        if (b - a) < tolerance or depth >= max_depth:
            solutions.append((a, b))
            continue
        
        # Subdivide
        left_coeffs, right_coeffs = de_casteljau_subdivide(c, 0.5)
        
        # Add sub-boxes to queue
        mid = (a + b) / 2
        queue.append((a, mid, left_coeffs, depth + 1))
        queue.append((mid, b, right_coeffs, depth + 1))
    
    return solutions
```

### Multi-Dimensional PP Solver

```python
def solve_pp_kd(equation_coeffs_list, tolerance=1e-6, max_depth=20):
    """
    Find solutions to polynomial system using PP method.
    
    Parameters
    ----------
    equation_coeffs_list : list of arrays
        Bernstein coefficients for each equation
    tolerance : float
        Convergence tolerance
    max_depth : int
        Maximum subdivision depth
        
    Returns
    -------
    list of boxes containing solutions
    """
    k = len(equation_coeffs_list[0].shape)  # Number of variables
    solutions = []
    
    # Initial box: [0, 1]^k
    initial_box = Box(ranges=[(0.0, 1.0) for _ in range(k)])
    queue = [(initial_box, equation_coeffs_list, 0)]
    
    while queue:
        box, coeffs_list, depth = queue.pop(0)
        
        # Check all equations
        can_exclude = False
        for coeffs in coeffs_list:
            p_min = np.min(coeffs)
            p_max = np.max(coeffs)
            
            # Exclusion test
            if p_min > 0 or p_max < 0:
                can_exclude = True
                break
        
        if can_exclude:
            continue  # No solution in this box
        
        # Convergence test
        if box.width() < tolerance or depth >= max_depth:
            solutions.append(box)
            continue
        
        # Subdivide along widest dimension
        dim = box.widest_dimension()
        sub_boxes = box.subdivide(dim)
        
        for sub_box in sub_boxes:
            # Get coefficients for sub-box using de Casteljau
            sub_coeffs_list = []
            for coeffs in coeffs_list:
                sub_coeffs = de_casteljau_kd(coeffs, sub_box, box)
                sub_coeffs_list.append(sub_coeffs)
            
            queue.append((sub_box, sub_coeffs_list, depth + 1))
    
    return solutions
```

## Key Properties

### Convergence

**PP method has LINEAR convergence:**
- Box width decreases by factor of 2 at each subdivision
- Number of iterations ∝ log(1/tolerance)

**Compare to LP method:**
- LP has QUADRATIC convergence
- Much faster for high accuracy

### Completeness

**PP method is COMPLETE:**
- Finds ALL roots in the domain
- No roots are missed (assuming sufficient depth)

### Robustness

**PP method is ROBUST:**
- Works for any polynomial in Bernstein form
- No numerical issues (only min/max operations)
- No matrix inversions or root finding

## Summary

### What is PP?

The **Projected Polyhedron** method uses the **convex hull property** of Bernstein polynomials to:
1. **Bound** the polynomial range using min/max of coefficients
2. **Exclude** boxes where the polynomial cannot be zero
3. **Subdivide** remaining boxes using de Casteljau
4. **Converge** to all roots

### Key Formula

```
min(b_0, ..., b_n) ≤ p(t) ≤ max(b_0, ..., b_n)  for all t ∈ [0, 1]
```

### Algorithm Flow

```
Start with [0, 1]^k
    ↓
Compute bounds: [min(coeffs), max(coeffs)]
    ↓
Can exclude? (0 not in bounds)
    ↓ No
Converged? (box small enough)
    ↓ No
Subdivide using de Casteljau
    ↓
Recurse on sub-boxes
```

### Advantages

✅ Simple to implement  
✅ No external dependencies  
✅ Numerically stable  
✅ Complete (finds all roots)  
✅ Works in any dimension  

### Disadvantages

❌ Loose bounds (not tight)  
❌ Linear convergence (slow)  
❌ Many subdivisions needed  

### When to Use

- Quick prototyping
- Simple problems
- When LP solver not available
- When simplicity > speed

