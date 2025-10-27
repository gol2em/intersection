# CRIT Logic Bug Fix

## Bug Report

### User's Observation
> "With Crit set to 0.8, pp reduction more than 20% should not be bisected, but it did. Check what is wrong."

**Absolutely correct!** The user identified a critical bug in the CRIT logic.

## The Bug

### What CRIT Should Do

`CRIT = 0.8` means:
- If PP reduces box by **≥ 20%**, DON'T subdivide (apply PP again to the tighter box)
- If PP reduces box by **< 20%**, DO subdivide (PP isn't helping much)

### Example

**Initial box**: [0, 1], width = 1.0  
**PP bounds**: [0.121, 0.879], width = 0.758  
**Reduction**: (1.0 - 0.758) / 1.0 = **24.2%**

Since 24.2% > 20%, should **NOT subdivide** - should extract sub-box [0.121, 0.879] and apply PP again.

### What the Code Was Doing (WRONG)

**File**: `src/intersection/subdivision_solver.py`  
**Lines**: 351-362 (before fix)

```python
# Determine which dimensions need subdivision
dims_to_subdivide = []
for i, (t_min, t_max) in enumerate(containing_ranges):
    size = t_max - t_min
    if size > self.config.crit:  # Check: 0.758 > 0.8 → FALSE
        dims_to_subdivide.append(i)

if not dims_to_subdivide:
    # All dimensions are small enough, but not below tolerance
    # Subdivide along the largest dimension  ← BUG!
    sizes = [t_max - t_min for t_min, t_max in containing_ranges]
    dims_to_subdivide = [np.argmax(sizes)]
```

**Problem**: When PP successfully reduced the box (size < CRIT), the code thought "all dimensions are small enough" and then **forced subdivision anyway**!

### Why This Happened

The logic was:
1. Check if `size > CRIT` → `0.758 > 0.8` → **FALSE**
2. `dims_to_subdivide` is empty
3. Code interprets this as "all dimensions small enough"
4. **Forces subdivision** along the largest dimension

**This is backwards!** When PP successfully reduces the box (all dimensions < CRIT), we should:
- **Extract the tighter sub-box**
- **Apply PP again** to the tighter box
- **NOT subdivide**

## The Fix

### New Logic (CORRECT)

**File**: `src/intersection/subdivision_solver.py`  
**Lines**: 351-382 (after fix)

```python
# Determine which dimensions need subdivision
# containing_ranges are in [0,1] space, so original size is 1.0
dims_to_subdivide = []
for i, (t_min, t_max) in enumerate(containing_ranges):
    size = t_max - t_min
    # If size > CRIT, PP didn't reduce enough (< (1-CRIT)% reduction)
    # Example: CRIT=0.8 means if size > 0.8, PP reduced < 20%, so subdivide
    if size > self.config.crit:
        dims_to_subdivide.append(i)

if not dims_to_subdivide:
    # PP successfully reduced all dimensions by ≥ (1-CRIT)
    # Extract the tighter sub-box and apply PP again (don't subdivide!)
    # This is the key to PP method efficiency
    sub_coeffs_list = []
    for eq_coeffs in sub_box.coeffs:
        sub_coeffs, new_box = extract_subbox_with_box(
            eq_coeffs,
            sub_box.box,
            containing_ranges,
            tolerance=self.config.subdivision_tolerance,
            verbose=False
        )
        sub_coeffs_list.append(sub_coeffs)
    
    # Return single sub-box with tighter bounds (same depth, not subdivided)
    return [SubdivisionBox(
        box=new_box,
        coeffs=sub_coeffs_list,
        depth=sub_box.depth  # Same depth - we're tightening, not subdividing
    )]
```

### Key Changes

1. **Removed forced subdivision** (lines 358-362 in old code)
2. **Added sub-box extraction** when PP is successful
3. **Return single tighter box** instead of subdividing
4. **Keep same depth** (we're tightening, not subdividing)

## Test Results

### Before Fix

**Test**: `(x - 0.2)(x - 0.5)(x - 0.8) = 0`

```
Processed: 64,400+ boxes
Found: 2 solutions (both at 0.5, missing 0.2 and 0.8)
Pruned: 6 boxes (0.009% pruning rate)
Time: >10 seconds
Status: ✗ FAILURE
```

### After Fix

**Test**: `(x - 0.2)(x - 0.5)(x - 0.8) = 0`

```
Processed: 12 boxes
Found: 3 solutions (0.2, 0.5, 0.8)
Pruned: 0 boxes
Time: 0.093 seconds
Status: ✓ SUCCESS
```

**Improvement**:
- **5,366× fewer boxes** (64,400 → 12)
- **107× faster** (>10s → 0.093s)
- **All roots found** (3/3 instead of 1/3)

## Why This Matters

### PP Method Efficiency

The PP method's power comes from **iterative tightening**:

1. Start with box [0, 1]
2. PP finds tighter bounds [0.121, 0.879] (24% reduction)
3. **Extract sub-box** [0.121, 0.879]
4. Apply PP again → even tighter bounds
5. Repeat until box is small enough

**Without the fix**: The algorithm subdivides immediately, creating exponential explosion.

**With the fix**: The algorithm tightens iteratively, converging quickly to roots.

### CRIT Parameter Meaning

`CRIT = 0.8` means:
- **Threshold for "PP is helping"**: If PP reduces box to < 80% of original, it's helping
- **Action when helping**: Extract tighter box and apply PP again
- **Action when not helping**: Subdivide to isolate roots better

**Example values**:
- `CRIT = 0.9`: Very aggressive (subdivide only if PP reduces < 10%)
- `CRIT = 0.8`: Balanced (subdivide if PP reduces < 20%)
- `CRIT = 0.5`: Conservative (subdivide if PP reduces < 50%)

## Impact on 20-Root Polynomial

### Before Fix

```
Polynomial: (x-0.05)(x-0.10)...(x-1.00) = 0
Processed: 420,000+ boxes in 180+ seconds
Pruned: 0 boxes (0% pruning)
Found: 0 solutions
Status: ✗ TIMEOUT
```

### After Fix (Expected)

With the fix, the PP method should:
1. Apply PP iteratively to tighten bounds
2. Subdivide only when PP doesn't help
3. Prune boxes that don't contain roots
4. Find all 20 roots efficiently

**Next step**: Test with 20-root polynomial to verify the fix works for high-degree cases.

## Files Modified

1. **`src/intersection/subdivision_solver.py`**
   - Lines 326-382: Fixed `_subdivide_box()` method
   - Removed forced subdivision when PP is successful
   - Added sub-box extraction and return

## Verification

### Test Files Created

1. **`examples/test_crit_logic.py`** - Demonstrates the bug with clear examples
2. **`examples/test_crit_with_debug.py`** - Tests with verbose output
3. **`examples/debug_pp_step_by_step.py`** - Step-by-step visualization

### Visualization Evidence

Generated 13 PNG files showing:
- Each subdivision step
- PP bounds at each level
- Which boxes are pruned vs. kept
- The bug in action (before fix)

## Conclusion

✅ **Bug identified**: CRIT logic was forcing subdivision when PP was successful  
✅ **Root cause**: Lines 358-362 in `_subdivide_box()` method  
✅ **Fix implemented**: Extract sub-box instead of subdividing when PP helps  
✅ **Test passed**: 3-root polynomial now works correctly (12 boxes vs 64,400+)  
⏳ **Next step**: Test with 20-root polynomial

**The user's observation was 100% correct - this was a critical bug that completely broke the PP method's efficiency!**

