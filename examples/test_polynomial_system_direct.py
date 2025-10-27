"""
Test solve_polynomial_system directly.
"""

import numpy as np
from intersection.polynomial_solver import create_polynomial_system, solve_polynomial_system
from intersection.bernstein import polynomial_nd_to_bernstein

# Simple polynomial: (x - 0.5) = 0
power_coeffs = np.array([-0.5, 1.0])

print("Testing polynomial_solver with: x - 0.5 = 0")
print(f"Power coefficients: {power_coeffs}")

# Convert to Bernstein
bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
print(f"Bernstein coefficients: {bern_coeffs}")

# Create system
system = create_polynomial_system(
    equation_coeffs=[bern_coeffs],
    param_ranges=[(0.0, 1.0)]
)

print(f"\nSystem created:")
print(f"  k = {system.k}")
print(f"  param_names = {system.param_names}")
print(f"  param_ranges = {system.param_ranges}")

# Solve
print("\nSolving with solve_polynomial_system...")
solutions = solve_polynomial_system(
    system,
    method='pp',
    tolerance=1e-6,
    crit=0.8,
    max_depth=30,
    subdivision_tolerance=1e-10,
    refine=False,
    verbose=True
)

print(f"\nSolutions: {solutions}")
print(f"Expected: [{{'x': 0.5}}]")

