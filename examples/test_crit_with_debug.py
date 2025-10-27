"""
Test CRIT logic with debug output to see what's happening.
"""

import numpy as np
from intersection.polynomial_solver import solve_polynomial_system, PolynomialSystem


def expand_polynomial_product(roots):
    """Expand (x - r1)(x - r2)...(x - rn) into power series."""
    coeffs = np.array([1.0])
    for root in roots:
        new_coeffs = np.zeros(len(coeffs) + 1)
        new_coeffs[1:] += coeffs
        new_coeffs[:-1] -= root * coeffs
        coeffs = new_coeffs
    return coeffs


if __name__ == "__main__":
    print("=" * 80)
    print("TEST: CRIT Logic with Debug Output")
    print("=" * 80)
    
    # Test polynomial: (x - 0.2)(x - 0.5)(x - 0.8)
    roots = [0.2, 0.5, 0.8]
    
    print(f"\nPolynomial: (x - 0.2)(x - 0.5)(x - 0.8) = 0")
    print(f"Expected roots: {roots}")
    
    # Expand to power form
    power_coeffs = expand_polynomial_product(roots)
    print(f"\nPower coefficients: {power_coeffs}")
    
    # Create polynomial system
    system = PolynomialSystem(
        equations=[power_coeffs],
        k=1,
        param_names=['t'],
        param_ranges=[(0.0, 1.0)]
    )
    
    # Solve with verbose output
    print("\n" + "=" * 80)
    print("SOLVING WITH VERBOSE OUTPUT")
    print("=" * 80)
    
    solutions = solve_polynomial_system(
        system,
        method='pp',
        tolerance=1e-6,
        crit=0.8,
        max_depth=30,
        refine=False,
        verbose=True
    )
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Solutions found: {len(solutions)}")
    
    found_roots = sorted([sol['t'] for sol in solutions])
    print(f"Found roots: {found_roots}")
    print(f"Expected roots: {roots}")
    
    # Check accuracy
    if len(found_roots) == len(roots):
        errors = [abs(f - e) for f, e in zip(found_roots, roots)]
        print(f"Errors: {errors}")
        print(f"Max error: {max(errors):.6e}")
        print("\n✓ SUCCESS: All roots found!")
    else:
        print(f"\n✗ FAILURE: Expected {len(roots)} roots, found {len(found_roots)}")

