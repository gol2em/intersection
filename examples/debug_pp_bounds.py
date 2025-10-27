"""
Debug script to see what PP bounds each equation gives individually.
"""

import numpy as np
from intersection.bernstein import polynomial_nd_to_bernstein
from intersection.convex_hull import find_root_box_pp_nd, _extract_dimension_range


# Define polynomials
eq1_power = np.zeros((3, 3))
eq1_power[0, 0] = -1
eq1_power[2, 0] = 1
eq1_power[0, 2] = 1

eq2_power = np.zeros((3, 3))
eq2_power[0, 0] = -1
eq2_power[2, 0] = 0.25
eq2_power[0, 2] = 4

# Convert to Bernstein
eq1_bern = polynomial_nd_to_bernstein(eq1_power, k=2)
eq2_bern = polynomial_nd_to_bernstein(eq2_power, k=2)

print("=" * 80)
print("INDIVIDUAL EQUATION PP BOUNDS")
print("=" * 80)

print("\nEquation 1: x² + y² - 1 = 0")
print("Bernstein coefficients:")
print(eq1_bern)

# Get PP bounds for eq1 alone
pp1 = find_root_box_pp_nd([eq1_bern], k=2)
print(f"\nPP bounds for eq1 alone: {pp1}")

# Get dimension ranges for eq1
dim0_range_eq1 = _extract_dimension_range(eq1_bern, dim=0, k=2)
dim1_range_eq1 = _extract_dimension_range(eq1_bern, dim=1, k=2)
print(f"  Dimension 0 (x) range: {dim0_range_eq1}")
print(f"  Dimension 1 (y) range: {dim1_range_eq1}")

print("\n" + "-" * 80)

print("\nEquation 2: x²/4 + 4y² - 1 = 0")
print("Bernstein coefficients:")
print(eq2_bern)

# Get PP bounds for eq2 alone
pp2 = find_root_box_pp_nd([eq2_bern], k=2)
print(f"\nPP bounds for eq2 alone: {pp2}")

# Get dimension ranges for eq2
dim0_range_eq2 = _extract_dimension_range(eq2_bern, dim=0, k=2)
dim1_range_eq2 = _extract_dimension_range(eq2_bern, dim=1, k=2)
print(f"  Dimension 0 (x) range: {dim0_range_eq2}")
print(f"  Dimension 1 (y) range: {dim1_range_eq2}")

print("\n" + "=" * 80)
print("COMBINED PP BOUNDS (INTERSECTION)")
print("=" * 80)

pp_combined = find_root_box_pp_nd([eq1_bern, eq2_bern], k=2)
print(f"\nPP bounds for both equations: {pp_combined}")

if pp1 and pp2:
    print("\nIntersection calculation:")
    print(f"  Dimension 0 (x):")
    print(f"    eq1: {pp1[0]}")
    print(f"    eq2: {pp2[0]}")
    x_min = max(pp1[0][0], pp2[0][0])
    x_max = min(pp1[0][1], pp2[0][1])
    print(f"    intersection: ({x_min}, {x_max})")
    
    print(f"  Dimension 1 (y):")
    print(f"    eq1: {pp1[1]}")
    print(f"    eq2: {pp2[1]}")
    y_min = max(pp1[1][0], pp2[1][0])
    y_max = min(pp1[1][1], pp2[1][1])
    print(f"    intersection: ({y_min}, {y_max})")

print("\n" + "=" * 80)
print("EXPECTED SOLUTION")
print("=" * 80)

x_exact = 2 / np.sqrt(5)
y_exact = 1 / np.sqrt(5)
print(f"\nExpected solution: x = {x_exact:.6f}, y = {y_exact:.6f}")

if pp_combined:
    x_in = pp_combined[0][0] <= x_exact <= pp_combined[0][1]
    y_in = pp_combined[1][0] <= y_exact <= pp_combined[1][1]
    print(f"\nIs solution in PP bounds?")
    print(f"  x: {x_in} ({pp_combined[0][0]:.6f} <= {x_exact:.6f} <= {pp_combined[0][1]:.6f})")
    print(f"  y: {y_in} ({pp_combined[1][0]:.6f} <= {y_exact:.6f} <= {pp_combined[1][1]:.6f})")
    print(f"  Both: {x_in and y_in}")

