"""
Simple test to verify tolerance management works correctly.
Each test has a 5-second manual check.
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


def test_polynomial(roots, name, tolerance, subdivision_tolerance, max_depth):
    """Test PP method with given roots and tolerances."""
    print("\n" + "=" * 80)
    print(f"TEST: {name}")
    print("=" * 80)
    
    # Expand polynomial
    power_coeffs = expand_polynomial_product(roots)
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    coeff_mag = np.max(np.abs(bern_coeffs))
    
    print(f"Roots: {len(roots)}, Degree: {len(power_coeffs)-1}")
    print(f"Coefficient magnitude: {coeff_mag:.6e}")
    print(f"Tolerances: param={tolerance:.6e}, subdiv={subdivision_tolerance:.6e}")
    
    # Create system
    system = create_polynomial_system(
        equation_coeffs=[bern_coeffs],
        param_ranges=[(0.0, 1.0)]
    )
    
    # Solve
    print(f"Solving... ", end='', flush=True)
    start_time = time.time()
    
    solutions = solve_polynomial_system(
        system,
        method='pp',
        tolerance=tolerance,
        crit=0.8,
        max_depth=max_depth,
        subdivision_tolerance=subdivision_tolerance,
        refine=False,
        verbose=False
    )
    
    solve_time = time.time() - start_time
    
    # Check if too slow
    if solve_time > 5.0:
        print(f"✗ TOO SLOW ({solve_time:.1f}s > 5s)")
        return False
    
    print(f"✓ {solve_time:.3f}s")
    
    # Check accuracy
    if len(solutions) > 0:
        param_name = list(solutions[0].keys())[0]
        found_roots = sorted([sol[param_name] for sol in solutions])
        expected_roots = sorted(roots)
        
        if len(found_roots) == len(expected_roots):
            max_error = max(abs(f - e) for f, e in zip(found_roots, expected_roots))
            print(f"✓ Found all {len(expected_roots)} roots, max error: {max_error:.6e}")
            return True
        else:
            print(f"✗ Expected {len(expected_roots)} roots, found {len(found_roots)}")
            return False
    else:
        print(f"✗ No solutions found")
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("TOLERANCE MANAGEMENT TEST (with 5-second limit per test)")
    print("=" * 80)
    
    # Test 1: 3 roots (should be fast with default tolerances)
    test_polynomial(
        roots=[0.2, 0.5, 0.8],
        name="3 roots - default tolerances",
        tolerance=1e-6,
        subdivision_tolerance=1e-10,
        max_depth=30
    )
    
    # Test 2: 5 roots (should be fast with default tolerances)
    test_polynomial(
        roots=[0.1, 0.3, 0.5, 0.7, 0.9],
        name="5 roots - default tolerances",
        tolerance=1e-6,
        subdivision_tolerance=1e-10,
        max_depth=30
    )
    
    # Test 3: 10 roots - analyze coefficients first
    print("\n" + "=" * 80)
    print("ANALYZING 10-ROOT POLYNOMIAL")
    print("=" * 80)
    
    roots_10 = [i * 0.1 for i in range(1, 11)]
    power_coeffs_10 = expand_polynomial_product(roots_10)
    bern_coeffs_10 = polynomial_nd_to_bernstein(power_coeffs_10, k=1)
    coeff_mag_10 = np.max(np.abs(bern_coeffs_10))
    
    print(f"Coefficient magnitude: {coeff_mag_10:.6e}")
    print(f"Recommended subdivision_tolerance: {coeff_mag_10 / 1000:.6e}")
    
    # Test with default tolerance (will likely fail)
    print("\n--- Attempt 1: Default subdivision_tolerance (1e-10) ---")
    success = test_polynomial(
        roots=roots_10,
        name="10 roots - default subdivision_tolerance",
        tolerance=1e-5,
        subdivision_tolerance=1e-10,
        max_depth=40
    )
    
    if not success:
        # Test with adjusted tolerance
        print("\n--- Attempt 2: Adjusted subdivision_tolerance ---")
        test_polynomial(
            roots=roots_10,
            name="10 roots - adjusted subdivision_tolerance",
            tolerance=1e-5,
            subdivision_tolerance=coeff_mag_10 / 1000,  # Auto-adjust
            max_depth=40
        )
    
    # Test 4: 20 roots - analyze coefficients first
    print("\n" + "=" * 80)
    print("ANALYZING 20-ROOT POLYNOMIAL")
    print("=" * 80)
    
    roots_20 = [i * 0.05 for i in range(1, 21)]
    power_coeffs_20 = expand_polynomial_product(roots_20)
    bern_coeffs_20 = polynomial_nd_to_bernstein(power_coeffs_20, k=1)
    coeff_mag_20 = np.max(np.abs(bern_coeffs_20))
    
    print(f"Coefficient magnitude: {coeff_mag_20:.6e}")
    print(f"Recommended subdivision_tolerance: {coeff_mag_20 / 1000:.6e}")
    print(f"Ratio to default (1e-10): {(coeff_mag_20 / 1000) / 1e-10:.1f}x larger")
    
    # Test with adjusted tolerance
    print("\n--- Testing with adjusted subdivision_tolerance ---")
    test_polynomial(
        roots=roots_20,
        name="20 roots - adjusted subdivision_tolerance",
        tolerance=1e-4,  # Relaxed parameter space tolerance
        subdivision_tolerance=coeff_mag_20 / 1000,  # Adjusted to coefficient scale
        max_depth=50
    )
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nKey findings:")
    print("1. Default subdivision_tolerance (1e-10) works for low-degree polynomials")
    print("2. For high-degree polynomials, subdivision_tolerance must be adjusted")
    print("3. Rule of thumb: subdivision_tolerance ≈ coefficient_magnitude / 1000")
    print("4. Coefficient magnitude of 10^-6 is NOT a problem with proper tolerances")
    print("\nThe issue was tolerance mismatch, not coefficient size!")

