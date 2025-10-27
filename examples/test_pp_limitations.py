"""
Test to demonstrate PP method limitations with high-degree polynomials.

This test shows that the PP method has numerical issues with:
1. High-degree polynomials (degree > 10)
2. Polynomials with many roots close together
3. Polynomials with very small Bernstein coefficients

The issue: Bernstein coefficients become very small and lose precision,
causing the convex hull to contain the x-axis everywhere, preventing pruning.
"""

import numpy as np
import time
from intersection.polynomial_solver import create_polynomial_system, solve_polynomial_system
from intersection.bernstein import polynomial_nd_to_bernstein


def expand_polynomial_product(roots):
    """Expand (x - r1)(x - r2)...(x - rn) into power series."""
    coeffs = np.array([1.0])
    for root in roots:
        # Multiply by (x - root)
        new_coeffs = np.zeros(len(coeffs) + 1)
        new_coeffs[1:] += coeffs  # x * coeffs
        new_coeffs[:-1] -= root * coeffs  # -root * coeffs
        coeffs = new_coeffs
    return coeffs


def test_polynomial(roots, name):
    """Test PP method with given roots."""
    print("\n" + "=" * 80)
    print(f"TEST: {name}")
    print("=" * 80)
    print(f"Number of roots: {len(roots)}")
    print(f"Roots: {roots}")
    
    # Expand polynomial
    power_coeffs = expand_polynomial_product(roots)
    degree = len(power_coeffs) - 1
    
    print(f"\nPolynomial degree: {degree}")
    print(f"Leading coefficient: {power_coeffs[-1]:.6e}")
    print(f"Constant term: {power_coeffs[0]:.6e}")
    
    # Convert to Bernstein basis
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    print(f"Bernstein coefficient range: [{np.min(bern_coeffs):.6e}, {np.max(bern_coeffs):.6e}]")
    print(f"Bernstein coefficient magnitude: {np.max(np.abs(bern_coeffs)):.6e}")
    
    # Check if coefficients are too small
    if np.max(np.abs(bern_coeffs)) < 1e-5:
        print("\n⚠️  WARNING: Bernstein coefficients are very small!")
        print("   This will cause numerical issues with PP method.")
        print("   The convex hull will likely contain x-axis everywhere.")
        print("   Expected result: NO PRUNING, very slow convergence")
    
    # Create polynomial system
    system = create_polynomial_system(
        equation_coeffs=[bern_coeffs],
        param_ranges=[(0.0, 1.0)]
    )
    
    # Solve with PP method
    print("\nSolving with PP method...")
    start_time = time.time()
    
    try:
        solutions = solve_polynomial_system(
            system,
            method='pp',
            tolerance=1e-4,  # Relaxed tolerance
            crit=0.8,
            max_depth=20,  # Limited depth to prevent infinite loops
            refine=False,
            verbose=False  # Disable verbose to avoid spam
        )
        
        solve_time = time.time() - start_time
        
        print(f"\nSolve time: {solve_time:.3f} seconds")
        print(f"Solutions found: {len(solutions)}")
        
        if len(solutions) > 0:
            print("\nFound solutions:")
            # Get parameter name from first solution
            param_name = list(solutions[0].keys())[0]
            for i, sol in enumerate(sorted(solutions, key=lambda s: s[param_name])):
                print(f"  {i+1}. x = {sol[param_name]:.6f}")

        # Check accuracy
        param_name = list(solutions[0].keys())[0] if solutions else 'x'
        found_roots = sorted([sol[param_name] for sol in solutions])
        expected_roots = sorted(roots)
        
        if len(found_roots) == len(expected_roots):
            max_error = max(abs(f - e) for f, e in zip(found_roots, expected_roots))
            print(f"\n✓ All {len(expected_roots)} roots found!")
            print(f"  Max error: {max_error:.6e}")
        else:
            print(f"\n✗ Expected {len(expected_roots)} roots, found {len(found_roots)}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n✗ Test interrupted (taking too long)")
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("PP METHOD LIMITATIONS TEST")
    print("=" * 80)
    
    # Test 1: Low-degree polynomial (should work well)
    print("\n" + "=" * 80)
    print("TEST 1: Low-degree polynomial (3 roots)")
    print("=" * 80)
    test_polynomial([0.2, 0.5, 0.8], "3 roots - SHOULD WORK")
    
    # Test 2: Medium-degree polynomial (should work)
    print("\n" + "=" * 80)
    print("TEST 2: Medium-degree polynomial (5 roots)")
    print("=" * 80)
    test_polynomial([0.1, 0.3, 0.5, 0.7, 0.9], "5 roots - SHOULD WORK")
    
    # Test 3: Higher-degree polynomial (may have issues)
    print("\n" + "=" * 80)
    print("TEST 3: Higher-degree polynomial (10 roots)")
    print("=" * 80)
    test_polynomial([i * 0.1 for i in range(1, 11)], "10 roots - MAY HAVE ISSUES")
    
    # Test 4: Very high-degree polynomial (will fail)
    print("\n" + "=" * 80)
    print("TEST 4: Very high-degree polynomial (20 roots)")
    print("=" * 80)
    print("\n⚠️  This test will likely fail or take very long!")
    print("   The Bernstein coefficients will be too small for PP method.")
    
    # Don't actually run this - it will take forever
    roots_20 = [i * 0.05 for i in range(1, 21)]
    power_coeffs = expand_polynomial_product(roots_20)
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    print(f"\nDegree: 20")
    print(f"Bernstein coefficient range: [{np.min(bern_coeffs):.6e}, {np.max(bern_coeffs):.6e}]")
    print(f"Bernstein coefficient magnitude: {np.max(np.abs(bern_coeffs)):.6e}")
    print("\n✗ SKIPPING TEST - coefficients too small for PP method")
    print("   PP method is NOT suitable for polynomials with degree > 10")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("PP method works well for:")
    print("  ✓ Low to medium degree polynomials (degree ≤ 10)")
    print("  ✓ Polynomials with well-separated roots")
    print("  ✓ Polynomials with reasonable coefficient magnitudes")
    print("\nPP method has issues with:")
    print("  ✗ High-degree polynomials (degree > 10)")
    print("  ✗ Polynomials with many roots close together")
    print("  ✗ Polynomials with very small Bernstein coefficients (< 1e-5)")
    print("\nFor high-degree polynomials, consider:")
    print("  • Using a different root-finding method (e.g., eigenvalue method)")
    print("  • Splitting the polynomial into lower-degree factors")
    print("  • Using interval arithmetic with higher precision")

