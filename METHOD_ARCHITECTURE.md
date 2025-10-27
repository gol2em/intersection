# Bounding Method Architecture

## Overview

The subdivision solver is designed to be **method-agnostic**, supporting multiple bounding methods (PP, LP, Hybrid) with minimal code duplication.

**Key Principle**: PP, LP, and Hybrid methods differ **only** in how they compute bounding boxes. All other logic (subdivision, tightening, recursion) is identical.

## Architecture

### Method Abstraction

The solver uses a **pluggable bounding method** architecture:

```python
class SubdivisionSolver:
    def _find_bounding_box(self, coeffs_list):
        """Dispatch to the appropriate bounding method."""
        if self.config.method == BoundingMethod.PP:
            return self._find_bounding_box_pp(coeffs_list)
        elif self.config.method == BoundingMethod.LP:
            return self._find_bounding_box_lp(coeffs_list)
        elif self.config.method == BoundingMethod.HYBRID:
            return self._find_bounding_box_hybrid(coeffs_list)
```

### Core Algorithm (Method-Independent)

The core subdivision algorithm is **identical** for all methods:

```python
def _solve_recursive(self, coeffs_list, box_range, depth):
    # Step 1: Apply bounding method (ONLY DIFFERENCE BETWEEN METHODS)
    bounding_result = self._find_bounding_box(coeffs_list)
    
    if bounding_result is None:
        return  # Prune - no roots
    
    # Step 2: Check if small enough (SAME FOR ALL METHODS)
    if all_dimensions_small(bounding_result, box_range):
        record_solution(midpoint(bounding_result, box_range))
        return
    
    # Step 3: Check depth limit (SAME FOR ALL METHODS)
    if depth >= max_depth:
        record_solution(midpoint(bounding_result, box_range))
        return
    
    # Step 4: Check CRIT - tighten or subdivide? (SAME FOR ALL METHODS)
    dims_to_subdivide = [i for i in range(k) if bound_width[i] > CRIT]
    
    if not dims_to_subdivide:
        # TIGHTEN (SAME FOR ALL METHODS)
        tight_coeffs = extract_subbox(coeffs_list, bounding_result)
        solve_recursive(tight_coeffs, tight_box_range, depth)
    else:
        # SUBDIVIDE (SAME FOR ALL METHODS)
        left_coeffs = extract_subbox(coeffs_list, left_ranges)
        right_coeffs = extract_subbox(coeffs_list, right_ranges)
        solve_recursive(left_coeffs, left_box_range, depth + 1)
        solve_recursive(right_coeffs, right_box_range, depth + 1)
```

## Bounding Methods

### PP (Projected Polyhedron) Method

**Status**: ✅ Implemented

**Algorithm**:
1. For each dimension j, project all Bernstein control points onto 2D plane (t_j, f)
2. Compute 2D convex hull of projected points
3. Intersect convex hull with t-axis (f=0) to get bounds

**Advantages**:
- Fast (O(n log n) per dimension)
- No external dependencies
- Works well for most problems

**Disadvantages**:
- May give loose bounds for some problems
- Only considers one dimension at a time

**Implementation**: `src/intersection/convex_hull.py::find_root_box_pp_nd()`

### LP (Linear Programming) Method

**Status**: ⏳ To be implemented

**Algorithm**:
1. Formulate bounding box problem as linear program
2. For each dimension j, solve two LPs:
   - Minimize t_j subject to: all control points ≥ 0
   - Maximize t_j subject to: all control points ≥ 0
3. Use LP solver (e.g., scipy.optimize.linprog)

**Advantages**:
- Tighter bounds than PP method
- Considers all dimensions simultaneously
- Can handle complex constraints

**Disadvantages**:
- Slower than PP method (LP solving overhead)
- Requires external LP solver

**Implementation**: `src/intersection/subdivision_solver.py::_find_bounding_box_lp()` (TODO)

### Hybrid Method

**Status**: ⏳ To be implemented

**Algorithm**:
1. First apply PP method for quick bounds
2. If PP bounds are not tight enough (width > threshold), refine with LP method
3. Otherwise, use PP bounds directly

**Advantages**:
- Balances speed and accuracy
- Fast for easy problems (uses PP)
- Accurate for hard problems (uses LP)

**Disadvantages**:
- More complex logic
- Requires tuning threshold parameter

**Implementation**: `src/intersection/subdivision_solver.py::_find_bounding_box_hybrid()` (TODO)

## Usage

### Selecting a Method

```python
from src.intersection.subdivision_solver import SubdivisionSolver, SolverConfig, BoundingMethod

# PP method (default)
config_pp = SolverConfig(method=BoundingMethod.PP)
solver_pp = SubdivisionSolver(config_pp)

# LP method (to be implemented)
config_lp = SolverConfig(method=BoundingMethod.LP)
solver_lp = SubdivisionSolver(config_lp)

# Hybrid method (to be implemented)
config_hybrid = SolverConfig(method=BoundingMethod.HYBRID)
solver_hybrid = SubdivisionSolver(config_hybrid)
```

### Comparing Methods

See `examples/test_methods.py` for a complete example:

```python
# Test all three methods on the same problem
pp_stats = test_circle_ellipse_with_method(BoundingMethod.PP)
lp_stats = test_circle_ellipse_with_method(BoundingMethod.LP)
hybrid_stats = test_circle_ellipse_with_method(BoundingMethod.HYBRID)

# Compare performance
print(f"PP:     {pp_stats['boxes_processed']} boxes")
print(f"LP:     {lp_stats['boxes_processed']} boxes")
print(f"Hybrid: {hybrid_stats['boxes_processed']} boxes")
```

## Implementation Guidelines

### Adding a New Bounding Method

To add a new bounding method (e.g., LP or Hybrid):

1. **Add enum value** in `subdivision_solver.py`:
   ```python
   class BoundingMethod(Enum):
       PP = "pp"
       LP = "lp"
       HYBRID = "hybrid"
       NEW_METHOD = "new_method"  # Add here
   ```

2. **Implement bounding function**:
   ```python
   def _find_bounding_box_new_method(self, coeffs_list):
       """
       New method for finding bounding box.
       
       Parameters
       ----------
       coeffs_list : List[np.ndarray]
           Bernstein coefficients for each equation
           
       Returns
       -------
       Optional[List[Tuple[float, float]]]
           Bounding box in [0,1]^k space, or None if no roots exist
       """
       # Implement your bounding method here
       # Must return List[Tuple[float, float]] or None
       pass
   ```

3. **Add dispatch case** in `_find_bounding_box()`:
   ```python
   def _find_bounding_box(self, coeffs_list):
       if self.config.method == BoundingMethod.PP:
           return self._find_bounding_box_pp(coeffs_list)
       elif self.config.method == BoundingMethod.LP:
           return self._find_bounding_box_lp(coeffs_list)
       elif self.config.method == BoundingMethod.HYBRID:
           return self._find_bounding_box_hybrid(coeffs_list)
       elif self.config.method == BoundingMethod.NEW_METHOD:
           return self._find_bounding_box_new_method(coeffs_list)  # Add here
   ```

4. **Test the new method**:
   ```python
   config = SolverConfig(method=BoundingMethod.NEW_METHOD)
   solver = SubdivisionSolver(config)
   solutions = solver.solve(equation_coeffs, parameter_ranges)
   ```

### Bounding Function Contract

All bounding functions must follow this contract:

**Input**:
- `coeffs_list`: List of Bernstein coefficient arrays (one per equation)
- Each coefficient array is in [0,1]^k space

**Output**:
- `List[Tuple[float, float]]`: Bounding box in [0,1]^k space
  - Each tuple (t_min, t_max) represents bounds for one dimension
  - Must satisfy: 0 ≤ t_min ≤ t_max ≤ 1
- `None`: If no roots exist in the box (prune)

**Guarantees**:
- If the function returns a bounding box, it must **contain all roots**
- If the function returns None, there must be **no roots** in the box
- The bounding box should be as **tight as possible** (smaller = better)

## Performance Considerations

### Expected Performance Characteristics

| Method | Speed | Accuracy | Boxes Processed |
|--------|-------|----------|-----------------|
| PP     | Fast  | Good     | Baseline        |
| LP     | Slow  | Excellent| Fewer than PP   |
| Hybrid | Medium| Excellent| Fewer than PP   |

### When to Use Each Method

- **PP**: Default choice for most problems. Fast and reliable.
- **LP**: When you need maximum accuracy and can afford slower solving.
- **Hybrid**: When you want to balance speed and accuracy automatically.

## Testing

### Unit Tests

Each bounding method should have unit tests:

```python
def test_bounding_method_pp():
    """Test PP method on known problems."""
    # Test that PP method finds correct bounds
    pass

def test_bounding_method_lp():
    """Test LP method on known problems."""
    # Test that LP method finds tighter bounds than PP
    pass

def test_bounding_method_hybrid():
    """Test Hybrid method on known problems."""
    # Test that Hybrid method balances speed and accuracy
    pass
```

### Benchmark Tests

Use the benchmark framework to compare methods:

```python
from src.intersection.benchmark import ScientificBenchmark

benchmark = ScientificBenchmark()
results_pp = benchmark.run_benchmark(config_pp, ...)
results_lp = benchmark.run_benchmark(config_lp, ...)
results_hybrid = benchmark.run_benchmark(config_hybrid, ...)

benchmark.compare_methods([results_pp, results_lp, results_hybrid])
```

## Future Work

### LP Method Implementation

**TODO**:
1. Implement LP formulation for bounding box problem
2. Integrate scipy.optimize.linprog or other LP solver
3. Handle edge cases (unbounded, infeasible)
4. Optimize for performance (caching, warm starts)

### Hybrid Method Implementation

**TODO**:
1. Implement PP-first strategy
2. Add threshold parameter for when to use LP
3. Tune threshold based on problem characteristics
4. Add adaptive threshold adjustment

### Other Potential Methods

- **Interval Arithmetic**: Use interval arithmetic for bounding
- **Affine Arithmetic**: More accurate than interval arithmetic
- **Taylor Models**: Polynomial approximations with error bounds
- **SDP (Semidefinite Programming)**: Even tighter bounds than LP

## References

- PP Method: Based on convex hull properties of Bernstein polynomials
- LP Method: Standard linear programming techniques
- Hybrid Method: Combines PP and LP for optimal performance

