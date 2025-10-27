# Normalization Strategy for Subdivision Methods

## The Problem

When using subdivision methods (LP, PP), we recursively divide the parameter domain into smaller boxes. Each sub-box needs to be analyzed, but the Bernstein basis and LP/PP methods assume the domain is [0,1]^k.

**Question:** When should we normalize?

## Two Approaches

### Approach 1: Normalize Once at the Beginning (WRONG ❌)

```
Initial domain: [a, b]
    ↓
Normalize to [0, 1]
    ↓
Subdivide: [0, 0.5] and [0.5, 1]
    ↓
Apply LP/PP to each sub-box
    ↓ PROBLEM!
Sub-boxes are NOT [0,1] - they are [0, 0.5] and [0.5, 1]
LP/PP methods assume [0,1]!
```

**Problem:** After subdivision, sub-boxes are NOT [0,1]^k anymore!

### Approach 2: Normalize at Each Subdivision Level (CORRECT ✅)

```
Initial domain: [a, b]
    ↓
Normalize to [0, 1] (Level 0)
    ↓
Subdivide: [0, 0.5] and [0.5, 1]
    ↓
For each sub-box:
    Renormalize [0, 0.5] → [0, 1]  (Level 1)
    Renormalize [0.5, 1] → [0, 1]  (Level 1)
    ↓
Apply LP/PP to normalized sub-boxes
    ↓
Subdivide again and renormalize (Level 2)
```

**Correct:** Each sub-box is renormalized to [0,1]^k before applying LP/PP!

## Why Renormalization is Needed

### Bernstein Basis Definition

Bernstein basis functions are defined on [0,1]:

```
B_i^n(t) = C(n,i) * t^i * (1-t)^(n-i)  for t ∈ [0,1]
```

### Convex Hull Property

The convex hull property holds on [0,1]:

```
min(b_i) ≤ p(t) ≤ max(b_i)  for t ∈ [0,1]
```

If the domain is [0, 0.5], this property doesn't hold directly!

### LP Constraints

LP constraints are formulated assuming t ∈ [0,1]:

```
Minimize/Maximize t
Subject to: 0 ≤ t ≤ 1
            Bernstein constraints
```

## Implementation Strategy

### Option A: Renormalize Bernstein Coefficients (RECOMMENDED ✅)

**Idea:** Use de Casteljau algorithm to compute Bernstein coefficients on sub-box, then renormalize.

```python
def subdivide_box(box, eq_coeffs, depth):
    # box.ranges might be [(0.25, 0.5), (0.3, 0.7)]
    
    # Step 1: Subdivide Bernstein coefficients using de Casteljau
    # This gives coefficients on the sub-box
    sub_eq_coeffs = de_casteljau_subdivide(eq_coeffs, box)
    
    # Step 2: These coefficients are already for the sub-box!
    # de Casteljau automatically gives us the right coefficients
    
    # Step 3: Apply LP/PP using these coefficients
    # The coefficients represent the polynomial on the sub-box
    # mapped to [0,1]
    bounds = lp_bounding(box, sub_eq_coeffs)
```

**Key Insight:** de Casteljau subdivision automatically gives us Bernstein coefficients that represent the polynomial on the sub-box, as if the sub-box were [0,1]^k!

### Option B: Explicit Renormalization (ALTERNATIVE)

**Idea:** Explicitly create a new normalized hypersurface for each sub-box.

```python
def subdivide_box(box, hypersurface, depth):
    # box.ranges might be [(0.25, 0.5), (0.3, 0.7)]
    
    # Step 1: Create a restricted hypersurface on sub-box
    def restricted_func(*params):
        # params are in sub-box range
        return hypersurface.func(*params)
    
    sub_hypersurface = Hypersurface(
        func=restricted_func,
        param_ranges=box.ranges,  # e.g., [(0.25, 0.5), (0.3, 0.7)]
        ambient_dim=hypersurface.n,
        degree=hypersurface.degree
    )
    
    # Step 2: Normalize sub-hypersurface to [0,1]^k
    sub_normalized, sub_transform = normalize_hypersurface(sub_hypersurface)
    
    # Step 3: Create system and apply LP/PP
    system = create_intersection_system(line, sub_normalized)
    solutions = lp_solve(system)
    
    # Step 4: Denormalize solutions
    solutions_in_subbox = denormalize_solutions(solutions, sub_transform)
```

**Problem:** This is VERY expensive - we're re-interpolating and re-computing Bernstein coefficients at every subdivision level!

## Recommended Approach: de Casteljau + Implicit Normalization

### The Elegant Solution

**Key Insight:** de Casteljau subdivision gives us exactly what we need!

When we subdivide Bernstein coefficients from domain [a,b] to sub-domain [c,d] ⊂ [a,b], the de Casteljau algorithm gives us new Bernstein coefficients that represent the polynomial on [c,d] **as if [c,d] were mapped to [0,1]**.

### Mathematical Explanation

Given Bernstein coefficients `b_i` for polynomial `p(t)` on [0,1]:

1. **Subdivide to [0, 0.5]:**
   - de Casteljau gives new coefficients `b_i^L`
   - These represent `p(t)` on [0, 0.5]
   - But interpreted as Bernstein basis on [0,1]!
   - Mathematically: `b_i^L` are Bernstein coefficients of `p(2t)` on [0,1]

2. **Subdivide to [0.5, 1]:**
   - de Casteljau gives new coefficients `b_i^R`
   - These represent `p(t)` on [0.5, 1]
   - But interpreted as Bernstein basis on [0,1]!
   - Mathematically: `b_i^R` are Bernstein coefficients of `p(2t-1)` on [0,1]

### Implementation

```python
class LPSolver:
    def _subdivide_box(self, box, eq_coeffs, depth):
        """
        Recursively subdivide parameter box.
        
        box: Current parameter box (might be sub-box of [0,1]^k)
        eq_coeffs: Bernstein coefficients on current box
                   (already normalized to [0,1]^k implicitly)
        """
        # Step 1: Apply LP method
        # eq_coeffs are interpreted as Bernstein basis on [0,1]^k
        bounds = self._lp_bounding(box, eq_coeffs)
        
        # Step 2: Exclusion test
        if self._can_exclude(eq_coeffs):
            return  # No root in this box
        
        # Step 3: Convergence check
        if self._has_converged(box, depth):
            # Found solution in box
            # box.center() gives solution in ORIGINAL parameter space
            self.solutions.append(box.center())
            return
        
        # Step 4: Subdivide
        sub_boxes = self._subdivide(box)
        
        for sub_box in sub_boxes:
            # Step 5: Compute Bernstein coefficients on sub-box
            # de Casteljau gives coefficients that are implicitly
            # normalized to [0,1]^k
            sub_eq_coeffs = self._de_casteljau_subdivide(
                eq_coeffs, 
                sub_box, 
                box
            )
            
            # Step 6: Recurse
            # sub_eq_coeffs are now Bernstein coefficients on sub_box
            # but interpreted as if sub_box were [0,1]^k
            self._subdivide_box(sub_box, sub_eq_coeffs, depth + 1)
```

### Key Points

1. **Initial normalization:** Normalize once at the beginning to [0,1]^k
2. **de Casteljau subdivision:** Gives coefficients on sub-box
3. **Implicit renormalization:** Coefficients are interpreted as Bernstein basis on [0,1]^k
4. **No explicit renormalization needed:** de Casteljau does it automatically!
5. **Track actual box ranges:** Keep track of actual parameter ranges for final solution

## Detailed Workflow

### Level 0: Initial

```
Original hypersurface: u ∈ [-π, π]
    ↓
Normalize to [0, 1]
    ↓
Bernstein coefficients: b_0, b_1, ..., b_n
Box: [0, 1]
```

### Level 1: First Subdivision

```
Subdivide at u = 0.5
    ↓
Left box: [0, 0.5]
    de Casteljau → b_0^L, b_1^L, ..., b_n^L
    These are Bernstein coeffs on [0, 0.5]
    Interpreted as Bernstein basis on [0, 1]
    
Right box: [0.5, 1]
    de Casteljau → b_0^R, b_1^R, ..., b_n^R
    These are Bernstein coeffs on [0.5, 1]
    Interpreted as Bernstein basis on [0, 1]
```

### Level 2: Second Subdivision (Left Box)

```
Left box [0, 0.5] subdivided at u = 0.25
    ↓
Left-left box: [0, 0.25]
    de Casteljau on b^L → b_0^LL, b_1^LL, ..., b_n^LL
    Interpreted as Bernstein basis on [0, 1]
    
Left-right box: [0.25, 0.5]
    de Casteljau on b^L → b_0^LR, b_1^LR, ..., b_n^LR
    Interpreted as Bernstein basis on [0, 1]
```

## Box Tracking

### Box Class Implementation

```python
class Box:
    def __init__(self, ranges, parent_ranges=None):
        """
        ranges: Current box ranges in ORIGINAL parameter space
        parent_ranges: Parent box ranges (for de Casteljau)
        """
        self.ranges = ranges  # e.g., [(0.25, 0.5)]
        self.parent_ranges = parent_ranges or ranges
    
    def center(self):
        """Return center in ORIGINAL parameter space."""
        return {f'u{i}': (r[0] + r[1]) / 2 for i, r in enumerate(self.ranges)}
    
    def to_parent_coords(self, *normalized_params):
        """
        Map from [0,1]^k (Bernstein space) to parent box coordinates.
        
        normalized_params: Parameters in [0,1]^k
        Returns: Parameters in parent box range
        """
        result = []
        for i, t in enumerate(normalized_params):
            a, b = self.ranges[i]
            u = a + t * (b - a)
            result.append(u)
        return result
```

## Summary

### Answer to Your Question

**When should normalization be done?**

1. ✅ **Once at the beginning:** Normalize original hypersurface to [0,1]^k
2. ✅ **Implicitly at each subdivision:** de Casteljau automatically provides coefficients that are "renormalized"
3. ❌ **NOT explicitly at each level:** Don't create new hypersurfaces and re-interpolate!

### The Magic of de Casteljau

de Casteljau subdivision is **exactly** the renormalization we need:
- Input: Bernstein coefficients on [a,b]
- Output: Bernstein coefficients on [c,d] ⊂ [a,b]
- Interpretation: Output coefficients are for Bernstein basis on [0,1]
- Effect: Automatic renormalization!

### Implementation Checklist

- [x] Normalize hypersurface once at beginning
- [x] Implement de Casteljau subdivision
- [ ] Track box ranges in original parameter space
- [ ] Use de Casteljau coefficients directly with LP/PP
- [ ] Return solutions in original parameter space

### Next Steps

1. Implement de Casteljau subdivision (1D, 2D, kD)
2. Implement Box class with proper range tracking
3. Implement LP/PP methods using subdivided coefficients
4. Test that solutions are in correct parameter space

