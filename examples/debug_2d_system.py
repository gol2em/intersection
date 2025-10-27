"""
Debug the 2D system to see what's happening.
"""

import numpy as np
from intersection.bernstein import polynomial_nd_to_bernstein
from intersection.convex_hull import find_root_box_pp_nd


# Define polynomials in power form
# Equation 1: x² + y² - 1 = 0
eq1_power = np.zeros((3, 3))
eq1_power[0, 0] = -1  # constant term
eq1_power[2, 0] = 1   # x² term
eq1_power[0, 2] = 1   # y² term

# Equation 2: x²/4 + 4y² - 1 = 0
eq2_power = np.zeros((3, 3))
eq2_power[0, 0] = -1    # constant term
eq2_power[2, 0] = 0.25  # x²/4 term
eq2_power[0, 2] = 4     # 4y² term

print("Power form:")
print("Equation 1:")
print(eq1_power)
print("\nEquation 2:")
print(eq2_power)

# Convert to Bernstein
eq1_bern = polynomial_nd_to_bernstein(eq1_power, k=2)
eq2_bern = polynomial_nd_to_bernstein(eq2_power, k=2)

print("\n" + "=" * 80)
print("Bernstein form:")
print("=" * 80)
print("Equation 1:")
print(eq1_bern)
print("\nEquation 2:")
print(eq2_bern)

# Check if PP method finds the root box
print("\n" + "=" * 80)
print("Testing PP method on initial box [0,1]²")
print("=" * 80)

result = find_root_box_pp_nd([eq1_bern, eq2_bern], k=2, tolerance=1e-10)

if result is None:
    print("\n✗ PP method returned None - no roots found!")
    print("This means the convex hull doesn't intersect the zero level.")
else:
    print(f"\n✓ PP method found root box:")
    for i, (t_min, t_max) in enumerate(result):
        print(f"  Dimension {i}: [{t_min:.6f}, {t_max:.6f}]")

# Let's also check the Bernstein coefficients at the expected solution
print("\n" + "=" * 80)
print("Checking Bernstein coefficients")
print("=" * 80)

print("\nEquation 1 coefficients (should have sign changes):")
print(eq1_bern)
print(f"Min: {np.min(eq1_bern):.6f}, Max: {np.max(eq1_bern):.6f}")

print("\nEquation 2 coefficients (should have sign changes):")
print(eq2_bern)
print(f"Min: {np.min(eq2_bern):.6f}, Max: {np.max(eq2_bern):.6f}")

# Analytical solution
x_exact = 2 / np.sqrt(5)
y_exact = 1 / np.sqrt(5)
print(f"\nExpected solution: x = {x_exact:.6f}, y = {y_exact:.6f}")

# Evaluate polynomials at grid points to see the sign pattern
print("\n" + "=" * 80)
print("Evaluating at grid points")
print("=" * 80)

for x in [0.0, 0.5, 0.894427, 1.0]:
    for y in [0.0, 0.447214, 0.5, 1.0]:
        eq1_val = x**2 + y**2 - 1
        eq2_val = x**2/4 + 4*y**2 - 1
        print(f"({x:.3f}, {y:.3f}): eq1={eq1_val:+.3f}, eq2={eq2_val:+.3f}")

