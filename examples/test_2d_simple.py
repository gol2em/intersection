"""
Simpler 2D test to debug the issue.
"""

import numpy as np
from intersection.polynomial_solver import solve_polynomial_system, PolynomialSystem
from intersection.bernstein import polynomial_nd_to_bernstein
import time
# No timeout on Windows


# Define polynomials
eq1_power = np.zeros((3, 3))
eq1_power[0, 0] = -1
eq1_power[2, 0] = 1
eq1_power[0, 2] = 1

eq2_power = np.zeros((3, 3))
eq2_power[0, 0] = -1
eq2_power[2, 0] = 0.25
eq2_power[0, 2] = 4

eq1_bern = polynomial_nd_to_bernstein(eq1_power, k=2)
eq2_bern = polynomial_nd_to_bernstein(eq2_power, k=2)

system = PolynomialSystem(
    equation_coeffs=[eq1_bern, eq2_bern],
    param_ranges=[(0.0, 1.0), (0.0, 1.0)],
    k=2,
    degree=2,
    param_names=['x', 'y']
)

print("Testing 2D system with verbose output...")
print("Expected solution: x ≈ 0.894427, y ≈ 0.447214")
print()

solutions = solve_polynomial_system(
    system,
    method='pp',
    tolerance=1e-3,  # Larger tolerance to see if it helps
    crit=0.5,
    max_depth=20,
    subdivision_tolerance=1e-10,
    refine=False,
    verbose=True
)

print(f"\n✓ Found {len(solutions)} solutions")
for i, sol in enumerate(solutions):
    print(f"  Solution {i+1}: x={sol['x']:.6f}, y={sol['y']:.6f}")

