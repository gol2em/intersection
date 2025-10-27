# Solver Refactoring Summary

## Overview

The `solve_polynomial_system()` function has been refactored to accept a `method` parameter, allowing you to choose between different solving algorithms.

## Changes Made

### 1. New Function Signature

```python
def solve_polynomial_system(
    system: Dict[str, Any],
    method: str = 'auto',           # NEW: Method selection
    tolerance: float = 1e-6,        # NEW: For subdivision methods
    max_depth: int = 20,            # NEW: For subdivision methods
    verbose: bool = False
) -> List[Dict[str, float]]:
```

### 2. Supported Methods

| Method | Description | Status |
|--------|-------------|--------|
| `'auto'` | Auto-select based on system (default) | ✅ Implemented |
| `'lp'` | Linear Programming (Sherbrooke & Patrikalakis 1993) | 🚧 Stub (falls back to numerical) |
| `'pp'` | Projected Polyhedron (Sherbrooke & Patrikalakis 1993) | 🚧 Stub (falls back to numerical) |
| `'numerical'` | Numerical methods (numpy roots, fsolve) | ✅ Implemented (existing) |
| `'subdivision'` | Simple Bernstein subdivision | 🚧 Stub (falls back to numerical) |

### 3. Architecture

The solver now uses a **dispatcher pattern**:

```
solve_polynomial_system()
    ├─> _auto_select_method()
    ├─> _solve_lp()           [TODO: Implement LP method]
    ├─> _solve_pp()           [TODO: Implement PP method]
    ├─> _solve_numerical()    [Existing implementation]
    └─> _solve_subdivision()  [TODO: Implement subdivision]
```

### 4. Backward Compatibility

✅ **Fully backward compatible!**

All existing code continues to work:

```python
# Old usage (still works)
solutions = solve_polynomial_system(system, verbose=True)

# New usage
solutions = solve_polynomial_system(system, method='lp', verbose=True)
solutions = solve_polynomial_system(system, method='pp', tolerance=1e-8)
```

### 5. Code Structure

```
src/intersection/solver.py
├─ solve_polynomial_system()        # Main entry point
├─ _auto_select_method()             # Method selection logic
├─ _solve_lp()                       # LP method (stub)
├─ _solve_pp()                       # PP method (stub)
├─ _solve_subdivision()              # Subdivision method (stub)
├─ _solve_numerical()                # Numerical dispatcher
│
├─ Numerical Solvers (Existing)
│  ├─ solve_2d_system()
│  ├─ solve_3d_system()
│  ├─ refine_root_newton()
│  ├─ remove_duplicate_roots()
│  └─ subdivide_and_solve()
```

## Usage Examples

### Example 1: Auto-select method (default)

```python
from src.intersection.geometry import Line, Hypersurface
from src.intersection.polynomial_system import create_intersection_system
from src.intersection.solver import solve_polynomial_system

# Create system
system = create_intersection_system(line, hypersurface)

# Solve with auto-selected method
solutions = solve_polynomial_system(system)
```

### Example 2: Specify LP method

```python
# Solve using Linear Programming method
solutions = solve_polynomial_system(
    system,
    method='lp',
    tolerance=1e-6,
    max_depth=20,
    verbose=True
)
```

### Example 3: Specify PP method

```python
# Solve using Projected Polyhedron method
solutions = solve_polynomial_system(
    system,
    method='pp',
    tolerance=1e-8,
    max_depth=30,
    verbose=True
)
```

### Example 4: Use numerical method explicitly

```python
# Explicitly use numerical methods
solutions = solve_polynomial_system(
    system,
    method='numerical',
    verbose=True
)
```

## Testing

All existing examples continue to work:

```bash
# Test 2D examples
uv run python examples/example_2d.py

# Test 3D examples
uv run python examples/example_3d.py

# Test Bernstein coefficient access
uv run python examples/example_get_bernstein_coeffs.py
```

✅ All tests pass!

## Next Steps

### To Implement LP Method:

1. **Implement de Casteljau subdivision** for Bernstein coefficients
   - `_de_casteljau_1d()` - 1D subdivision
   - `_de_casteljau_2d()` - 2D subdivision
   - `_de_casteljau_kd()` - k-D subdivision

2. **Implement LP bounding**
   - `_lp_bounding()` - Use scipy.optimize.linprog
   - `_build_lp_constraints()` - Formulate constraints from Bernstein coefficients

3. **Implement subdivision logic**
   - `_subdivide_box()` - Recursive subdivision
   - `_can_exclude()` - Exclusion test using Bernstein sign
   - `_has_converged()` - Convergence check

4. **Create Box class** for parameter domain representation

### To Implement PP Method:

1. **Implement PP bounding** (simpler than LP)
   - Use min/max of Bernstein coefficients directly
   - No optimization needed

2. **Reuse subdivision infrastructure** from LP method

### To Implement Subdivision Method:

1. **Extend existing `subdivide_and_solve()`** function
2. **Add multi-dimensional support**

## File Changes

### Modified Files:
- `src/intersection/solver.py` - Refactored with method parameter

### No Changes Required:
- `src/intersection/geometry.py`
- `src/intersection/polynomial_system.py`
- `src/intersection/bernstein.py`
- `src/intersection/interpolation.py`
- `examples/*.py`
- `tests/*.py`

## Benefits

1. ✅ **Extensible**: Easy to add new solving methods
2. ✅ **Backward Compatible**: Existing code works without changes
3. ✅ **Clean Interface**: Single entry point with method selection
4. ✅ **Documented**: Clear docstrings and examples
5. ✅ **Tested**: All existing examples pass
6. ✅ **Ready for Implementation**: Stubs in place for LP, PP, subdivision methods

## Summary

The solver has been successfully refactored to support multiple solving methods through a clean, extensible interface. The existing numerical solver continues to work, and stubs are in place for implementing the LP and PP methods from the research papers.

**Status**: ✅ Refactoring Complete - Ready for LP/PP Implementation

