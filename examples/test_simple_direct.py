"""
Minimal test to check if solver works at all.
"""

import numpy as np
from intersection.bernstein import polynomial_nd_to_bernstein
from intersection.subdivision_solver import solve_with_subdivision

# Simple polynomial: (x - 0.5) = 0
# Power form: -0.5 + x
power_coeffs = np.array([-0.5, 1.0])

print("Testing simple polynomial: x - 0.5 = 0")
print(f"Power coefficients: {power_coeffs}")

# Convert to Bernstein
bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
print(f"Bernstein coefficients: {bern_coeffs}")

# Solve
print("\nSolving...")
solutions = solve_with_subdivision(
    [bern_coeffs],
    k=1,
    method='pp',
    tolerance=1e-6,
    crit=0.8,
    max_depth=30,
    subdivision_tolerance=1e-10,
    verbose=True
)

print(f"\nSolutions: {solutions}")
print(f"Expected: [0.5]")

