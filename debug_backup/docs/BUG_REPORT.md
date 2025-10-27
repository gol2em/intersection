# Bug Report: PP Method Implementation Issue

## Summary

The PP method implementation has a critical bug that causes:
1. **Excessive box creation**: 9 active boxes for a 3-root polynomial
2. **Missing roots**: Only finds 1 out of 3 roots before hitting the safety limit
3. **Poor pruning**: Only 1 box pruned out of 11 processed (9% pruning rate)
4. **Slow performance**: Takes >10 seconds for a simple degree-3 polynomial

## Test Case

**Polynomial**: `(x - 0.2)(x - 0.5)(x - 0.8) = 0`

**Expected**:
- 3 roots at x = 0.2, 0.5, 0.8
- Should process ~10-20 boxes total
- Should prune boxes that don't contain roots

**Actual**:
- Found only 1 root at x = 0.200000
- Processed 11 boxes before safety stop
- Only 1 box pruned
- 9 boxes still active (should be ≤ 3)

## Visualization Evidence

Generated 13 PNG files showing step-by-step execution:

### Key Observations from Visualizations

1. **Step 1** (`debug_step_0001_depth0_subdivide.png`):
   - Initial box: [0.0, 1.0]
   - PP bounds: [0.121212, 0.878788]
   - **Good**: PP method correctly identifies that roots are in ~[0.12, 0.88]
   - Subdivides at t=0.5

2. **Step 2** (`debug_step_0002_depth1_subdivide.png`):
   - Left box: [0.0, 0.5]
   - PP bounds: [0.242424, 1.000000] → mapped to [0.121212, 0.500000]
   - **Problem**: PP upper bound is 1.0 in normalized space, meaning it extends to the right edge
   - This suggests the box contains roots near 0.5, but the subdivision doesn't isolate them

3. **Steps 3-11**: Deep recursion on left side
   - Algorithm keeps subdividing the left portion
   - Converges to root at x = 0.2 after 11 subdivisions
   - **Problem**: The right portions from each subdivision are left unprocessed

4. **Step 12** (`debug_step_0012_depth11_solution.png`):
   - Found solution at x = 0.200000
   - Width: 2.77e-07 (well below tolerance)
   - ✓ This root is correct

5. **Step 13** (`debug_step_0013_depth11_pruned.png`):
   - Right sibling of step 12: [0.200001, 0.200002]
   - Correctly pruned (no roots in this tiny box)

## Root Cause Analysis

The bug appears to be in the **subdivision strategy**:

### Current (Buggy) Behavior:
```
subdivide_box(box):
    pp_bounds = find_pp_bounds(box)
    if pp_bounds.width < tolerance:
        return SOLUTION
    
    # Subdivide at MIDPOINT OF PP BOUNDS
    mid = (pp_bounds.min + pp_bounds.max) / 2
    left, right = subdivide_at(mid)
    
    # Process left FIRST (depth-first)
    subdivide_box(left)   # ← Goes deep before processing right
    subdivide_box(right)  # ← Never reached if left branch is long
```

### Expected Behavior:
```
subdivide_box(box):
    pp_bounds = find_pp_bounds(box)
    if pp_bounds.width < tolerance:
        return SOLUTION
    
    # Extract sub-box with PP bounds
    tight_box = extract_subbox(box, pp_bounds)
    
    # Subdivide the TIGHT box
    mid = (pp_bounds.min + pp_bounds.max) / 2
    left, right = subdivide_at(mid)
    
    # Add both to QUEUE (breadth-first or balanced)
    queue.add(left)
    queue.add(right)
```

## The "9 Active Boxes" Problem

After processing the left branch to depth 11:
- **1 solution found** (x = 0.2)
- **1 box pruned** (right sibling of solution)
- **9 boxes still waiting**: These are the RIGHT siblings from depths 0-8

The algorithm uses **depth-first recursion**, which means:
1. Subdivide at depth 0 → left + right
2. Process left immediately (depth-first)
3. Left subdivides → left-left + left-right
4. Process left-left immediately
5. ... continues until solution found
6. **Right siblings accumulate on the call stack**

For a 3-root polynomial, we should have at most 3 active regions:
- Region around x = 0.2
- Region around x = 0.5
- Region around x = 0.8

Having 9 active boxes means the subdivision is creating too many fragments.

## Tolerance Analysis

The test used:
- `tolerance = 1e-6` (solution precision in parameter space)
- `subdivision_tolerance = 1e-10` (zero detection in function value space)
- Coefficient magnitude: 1.4e-01

**Verdict**: Tolerances are correctly set. The issue is NOT tolerance-related.

## Comparison with Working Test

The simple test `x - 0.5 = 0` worked correctly:
- Processed: 1 box
- Pruned: 0 boxes
- Solutions: 1
- Time: <0.1 seconds

This suggests the bug manifests when:
- Multiple roots exist
- Subdivision depth > 1
- PP bounds don't tightly isolate individual roots

## Recommended Fix

The issue is likely in `src/intersection/subdivision_solver.py`:

### Option 1: Fix the subdivision logic
Check how `_subdivide_box()` creates sub-boxes. It should:
1. Use PP bounds to extract a tight sub-box
2. Subdivide the tight sub-box (not the original box)
3. Ensure both children are processed

### Option 2: Switch to breadth-first search
Instead of depth-first recursion, use a queue:
```python
queue = [initial_box]
while queue:
    box = queue.pop(0)  # FIFO
    result = process_box(box)
    if result == SUBDIVIDE:
        left, right = subdivide(box)
        queue.append(left)
        queue.append(right)
```

This would prevent the "9 active boxes" problem by processing all boxes at the same depth before going deeper.

## Files Generated

- `debug_step_0001_depth0_subdivide.png` through `debug_step_0013_depth11_pruned.png`
- Each shows:
  - Left panel: Polynomial curve with control points
  - Right panel: Convex hull and PP bounds
  - Status: SUBDIVIDE, SOLUTION, or PRUNED

## Next Steps

1. ✅ **Bug confirmed** with visual evidence
2. ✅ **Tolerance classification** completed (not the issue)
3. ⏳ **Fix subdivision logic** in `subdivision_solver.py`
4. ⏳ **Re-test** with 3-root polynomial
5. ⏳ **Test** with 20-root polynomial after fix

## Conclusion

The PP method implementation has a fundamental bug in how it manages the subdivision tree. The tolerances are correctly set - the issue is architectural, not numerical.

**The 20-root polynomial failure is a symptom of this bug, not a tolerance problem.**

