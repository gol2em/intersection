"""
Detailed debug of 2D system with manual subdivision tracking.
"""

import numpy as np
from intersection.bernstein import polynomial_nd_to_bernstein
from intersection.subdivision_solver import SubdivisionSolver, SubdivisionConfig, SubdivisionBox
from intersection.box import Box

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

print("Bernstein coefficients:")
print("Equation 1:")
print(eq1_bern)
print("\nEquation 2:")
print(eq2_bern)

# Create solver
config = SubdivisionConfig(
    method='pp',
    tolerance=1e-6,
    crit=0.5,
    max_depth=50,
    subdivision_tolerance=1e-10,
    verbose=True
)

solver = SubdivisionSolver(config)

# Create initial box
initial_box = Box(
    k=2,
    param_ranges=[(0.0, 1.0), (0.0, 1.0)]
)

initial_sub_box = SubdivisionBox(
    box=initial_box,
    coeffs=[eq1_bern, eq2_bern],
    depth=0
)

print("\n" + "=" * 80)
print("Starting subdivision")
print("=" * 80)

# Manually call solve to see what happens
solutions = solver.solve([initial_sub_box])

print("\n" + "=" * 80)
print("Results")
print("=" * 80)
print(f"Solutions: {len(solutions)}")
print(f"Stats: {solver.stats}")

for i, sol in enumerate(solutions):
    print(f"\nSolution {i+1}: {sol}")

