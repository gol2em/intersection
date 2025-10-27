# PP Method Visualization Guide

## Overview

The PP (Projected Polyhedron) method visualization shows every step of the polynomial solving process, from power series to Bernstein form, convex hull computation, and subdivision.

## Generated Visualizations

Run the visualization script:
```bash
uv run python examples/visualize_pp_method.py
```

This generates 7 PNG files showing the complete PP method workflow.

## Step-by-Step Breakdown

### Example Polynomial

For demonstration, we use a cubic polynomial with 3 roots:
- **Polynomial**: (x - 0.2)(x - 0.5)(x - 0.8) = 0
- **Expected roots**: x = 0.2, 0.5, 0.8
- **Domain**: [0, 1]

---

## Step 1: Power Series Form

**File**: `step1_power_series.png`

### What it shows:
- The polynomial plotted in standard power basis form
- Roots marked as red circles at x = 0.2, 0.5, 0.8

### Power series coefficients:
```
p(x) = a0 + a1*x + a2*x^2 + a3*x^3

a0 = -0.080000  (constant term)
a1 = +0.660000  (linear coefficient)
a2 = -1.500000  (quadratic coefficient)
a3 = +1.000000  (cubic coefficient)
```

### Key insight:
The power basis is the standard polynomial representation, but it's not ideal for subdivision methods because:
- Coefficients don't directly bound the polynomial values
- Subdivision requires expensive polynomial evaluation

---

## Step 2: Bernstein Form

**File**: `step2_bernstein.png`

### What it shows:
- The same polynomial in Bernstein basis
- **Control points** (green circles) at positions (i/n, b_i)
- **Control polygon** (green dashed line) connecting control points
- The polynomial curve (blue) lies within the convex hull of control points

### Bernstein coefficients:
```
p(x) = Σ bi * B_i^3(x)  where B_i^3(x) = C(3,i) * x^i * (1-x)^(3-i)

b0 = -0.080000  at t = 0.0000
b1 = +0.140000  at t = 0.3333
b2 = -0.140000  at t = 0.6667
b3 = +0.080000  at t = 1.0000
```

### Control points:
```
P0 = (0.0000, -0.080000)
P1 = (0.3333, +0.140000)
P2 = (0.6667, -0.140000)
P3 = (1.0000, +0.080000)
```

### Key insight:
The Bernstein basis has the **convex hull property**:
- The polynomial curve lies entirely within the convex hull of control points
- This property enables efficient bounding and pruning

---

## Step 3: Convex Hull and X-Axis Intersection

**File**: `step3_convex_hull.png`

### What it shows:
- **Convex hull** (yellow shaded region) of the control points
- **Hull vertices** (red circles) forming the boundary
- **X-axis intersection** (green shaded band) showing where roots can exist
- Actual roots (red stars) at x = 0.2, 0.5, 0.8

### Convex hull vertices (counter-clockwise):
```
Vertex 0: (0.0000, -0.080000)  [bottom-left]
Vertex 1: (0.6667, -0.140000)  [bottom-right]
Vertex 2: (1.0000, +0.080000)  [top-right]
Vertex 3: (0.3333, +0.140000)  [top-left]
```

### X-axis intersection:
```
Intersection range: [0.121212, 0.878788]
Width: 0.757576
```

### Key insight:
The **PP method** uses convex hull intersection to find tight bounds:
- If convex hull doesn't intersect x-axis → **no roots** (prune this box)
- If convex hull intersects x-axis at [t_min, t_max] → **all roots** must be in [t_min, t_max]
- This gives much tighter bounds than just using [0, 1]

---

## Step 4: Subdivision Process

**Files**: 
- `step4_subdivision_depth0.png` - Initial box
- `step4_subdivision_depth1.png` - First subdivision (2 boxes)
- `step4_subdivision_depth2.png` - Second subdivision (4 boxes)
- `step4_subdivision_depth3.png` - Third subdivision (8 boxes, 4 pruned)

### Subdivision statistics:
```
Depth 0: 1 box  (0 pruned)
Depth 1: 2 boxes (0 pruned)
Depth 2: 4 boxes (0 pruned)
Depth 3: 8 boxes (4 pruned)
Total: 15 boxes processed
```

### What each depth shows:

#### Depth 0 (Initial Box)
- **Domain**: [0.0, 1.0]
- **Convex hull bounds**: [0.121, 0.879]
- Shows the full polynomial with all 3 roots

#### Depth 1 (First Subdivision)
- **Left box**: [0.0, 0.5]
  - Contains roots at x = 0.2
  - Convex hull gives tighter bounds
  
- **Right box**: [0.5, 1.0]
  - Contains roots at x = 0.5, 0.8
  - Convex hull gives tighter bounds

#### Depth 2 (Second Subdivision)
- 4 boxes, each containing 0-2 roots
- Convex hull bounds become progressively tighter
- Some boxes may be pruned if convex hull doesn't intersect x-axis

#### Depth 3 (Third Subdivision)
- 8 boxes total
- **4 boxes pruned** (convex hull doesn't intersect x-axis)
- **4 boxes kept** (contain roots)
- Each kept box is very small and close to a root

### Visualization features:

Each subplot shows:
- **Blue curve**: Polynomial in that box's domain
- **Green circles**: Control points
- **Green dashed line**: Control polygon
- **Yellow region**: Convex hull
- **Red boundary**: Convex hull edges
- **Green shaded band**: X-axis intersection (root bounds)
- **Red stars**: Actual roots (if in this box)
- **Title**: Box range and computed bounds (or "PRUNED")

### Key insight:
The subdivision process:
1. **Subdivides** each box into two halves
2. **Computes** convex hull for each half
3. **Prunes** boxes where convex hull doesn't intersect x-axis
4. **Refines** remaining boxes until they're smaller than tolerance
5. **Reports** center of each small box as a root

---

## Mathematical Foundation

### Bernstein Basis Functions

For degree n, the i-th Bernstein basis function is:
```
B_i^n(t) = C(n, i) * t^i * (1-t)^(n-i)
```

where C(n, i) is the binomial coefficient.

### Convex Hull Property

For a Bernstein polynomial:
```
p(t) = Σ b_i * B_i^n(t)
```

The curve p(t) lies within the convex hull of control points {(i/n, b_i)}.

### PP Method Algorithm

```
function PP_METHOD(coeffs, box):
    1. Get control points: P_i = (i/n, coeffs[i])
    2. Compute convex hull of control points
    3. Find intersection of convex hull with x-axis
    4. If no intersection:
         PRUNE (no roots in this box)
    5. If intersection [t_min, t_max]:
         If box_width < tolerance:
             REPORT root at box center
         Else:
             Subdivide box at midpoint
             Recursively process left and right sub-boxes
```

---

## Performance Benefits

### Pruning Efficiency

The PP method can prune large regions of the search space:
- **Depth 3**: 4 out of 8 boxes pruned (50% reduction)
- Without PP method: would need to subdivide all 8 boxes
- With PP method: only 4 boxes need further subdivision

### Tighter Bounds

Convex hull intersection gives much tighter bounds than naive subdivision:
- **Naive bounds**: [0.0, 1.0] → width = 1.0
- **PP bounds**: [0.121, 0.879] → width = 0.758 (24% reduction)
- After subdivision, bounds become even tighter

---

## Customization

### Change the polynomial

Edit `examples/visualize_pp_method.py`:
```python
# Use different roots
roots = [0.1, 0.3, 0.6, 0.9]  # 4 roots instead of 3

# Or use a different polynomial
roots = [0.25, 0.75]  # Quadratic with 2 roots
```

### Change subdivision depth

```python
# Show more subdivision levels
boxes = visualize_subdivision_tree(bern_coeffs, roots, max_depth=5)
```

### Change domain

```python
# Use domain [0, 10] instead of [0, 1]
# Need to normalize the polynomial first
```

---

## Interpreting the Visualizations

### Good signs:
- ✅ Convex hull tightly bounds the polynomial curve
- ✅ X-axis intersection contains all actual roots
- ✅ Many boxes are pruned at deeper levels
- ✅ Remaining boxes are small and centered on roots

### Warning signs:
- ⚠️ Convex hull is very loose (control points far from curve)
- ⚠️ No boxes are pruned (PP method not effective)
- ⚠️ X-axis intersection is almost the full domain (bounds not tight)

### For high-degree polynomials:
- Control polygon may oscillate wildly
- Convex hull may be very large initially
- More subdivision levels needed
- Numerical issues may arise (use higher precision)

---

## Comparison with Other Methods

### PP Method (Projected Polyhedron)
- ✅ Uses convex hull for tight bounds
- ✅ Can prune large regions
- ✅ Works well for sparse roots
- ⚠️ Convex hull computation has overhead

### LP Method (Linear Programming)
- ✅ Even tighter bounds than PP
- ✅ Can handle constraints
- ⚠️ More expensive per box
- ⚠️ Requires LP solver

### Hybrid Method
- ✅ Uses PP for initial pruning
- ✅ Uses LP for final refinement
- ✅ Best of both worlds
- ⚠️ More complex implementation

---

## References

- **Bernstein Polynomials**: Basis for computer-aided geometric design
- **Convex Hull Property**: Fundamental property enabling PP method
- **Graham Scan**: O(n log n) algorithm for 2D convex hull
- **De Casteljau Algorithm**: Stable subdivision of Bernstein polynomials

---

## Next Steps

1. **Run the visualization** to see the PP method in action
2. **Experiment with different polynomials** to understand behavior
3. **Try the 20-root test** to see performance on complex polynomials
4. **Compare with LP method** to see the trade-offs

```bash
# Run visualization
uv run python examples/visualize_pp_method.py

# Run 20-root test
uv run python examples/test_20_roots_improved.py
```

