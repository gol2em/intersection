"""
Test 3-root polynomial directly.
"""

import numpy as np
import time
from intersection.polynomial_solver import create_polynomial_system, solve_polynomial_system
from intersection.bernstein import polynomial_nd_to_bernstein


def expand_polynomial_product(roots):
    """Expand (x - r1)(x - r2)...(x - rn) into power series."""
    coeffs = np.array([1.0])
    for root in roots:
        new_coeffs = np.zeros(len(coeffs) + 1)
        new_coeffs[1:] += coeffs
        new_coeffs[:-1] -= root * coeffs
        coeffs = new_coeffs
    return coeffs


# Test (x-0.2)(x-0.5)(x-0.8) = 0
roots = [0.2, 0.5, 0.8]

print(f"Testing polynomial with roots: {roots}")

# Expand
power_coeffs = expand_polynomial_product(roots)
print(f"Power coefficients: {power_coeffs}")

# Convert to Bernstein
bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
print(f"Bernstein coefficients: {bern_coeffs}")
print(f"Coefficient magnitude: {np.max(np.abs(bern_coeffs)):.6e}")

# Create system
system = create_polynomial_system(
    equation_coeffs=[bern_coeffs],
    param_ranges=[(0.0, 1.0)]
)

# Solve
print("\nSolving...")
start_time = time.time()

solutions = solve_polynomial_system(
    system,
    method='pp',
    tolerance=1e-6,
    crit=0.8,
    max_depth=30,
    subdivision_tolerance=1e-10,
    refine=False,
    verbose=True  # Enable verbose to see where it hangs
)

solve_time = time.time() - start_time

print(f"Solve time: {solve_time:.3f} seconds")
print(f"Solutions found: {len(solutions)}")

if len(solutions) > 0:
    param_name = list(solutions[0].keys())[0]
    found_roots = sorted([sol[param_name] for sol in solutions])
    print(f"Found roots: {found_roots}")
    print(f"Expected roots: {sorted(roots)}")
    
    if len(found_roots) == len(roots):
        errors = [abs(f - e) for f, e in zip(found_roots, sorted(roots))]
        print(f"Errors: {errors}")
        print(f"Max error: {max(errors):.6e}")

