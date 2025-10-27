"""
Test: Solve polynomial with 20 roots using PP method with tolerance 1e-8

Polynomial: (x-1)(x-2)(x-3)...(x-20) = 0
Expected roots: x = 1, 2, 3, ..., 20
Domain: x ∈ [0, 25]
Tolerance: 1e-8
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from intersection.polynomial_solver import (
    create_polynomial_system,
    solve_polynomial_system
)
from intersection.bernstein import (
    polynomial_nd_to_bernstein,
    transform_polynomial_domain_1d
)
import time


def expand_polynomial_product(roots):
    """
    Expand (x - r1)(x - r2)...(x - rn) into power basis coefficients.
    
    Returns coefficients [a0, a1, a2, ..., an] for a0 + a1*x + a2*x^2 + ... + an*x^n
    """
    # Start with polynomial = 1
    poly = np.array([1.0])
    
    # Multiply by (x - root) for each root
    for root in roots:
        # (x - root) has coefficients [-root, 1]
        factor = np.array([-root, 1.0])
        # Multiply polynomials
        poly = np.convolve(poly, factor)
    
    return poly


def test_20_roots_high_precision():
    """Test polynomial with 20 roots using tolerance 1e-8."""
    print("\n" + "=" * 80)
    print("TEST: Polynomial with 20 Roots (Tolerance 1e-8)")
    print("=" * 80)
    
    # Define roots
    roots = list(range(1, 21))  # [1, 2, 3, ..., 20]
    print(f"\nExpected roots: {roots}")
    
    # Expand polynomial in original domain [0, 25]
    print("\nExpanding polynomial (x-1)(x-2)...(x-20)...")
    power_coeffs_original = expand_polynomial_product(roots)
    
    print(f"Polynomial degree: {len(power_coeffs_original) - 1}")
    print(f"Leading coefficient: {power_coeffs_original[-1]:.2e}")
    print(f"Constant term: {power_coeffs_original[0]:.2e}")
    
    # Transform to [0, 1] domain using helper function
    print("\nTransforming to [0, 1] domain using helper function...")
    start_time = time.time()
    power_coeffs_normalized = transform_polynomial_domain_1d(
        power_coeffs_original,
        from_range=(0.0, 25.0),
        to_range=(0.0, 1.0),
        verbose=True
    )
    transform_time = time.time() - start_time
    
    print(f"\nTransformation time: {transform_time:.3f} seconds")
    print(f"Normalized constant term: {power_coeffs_normalized[0]:.2e}")
    print(f"Normalized leading coefficient: {power_coeffs_normalized[-1]:.2e}")
    
    # Convert to Bernstein basis
    print("\nConverting to Bernstein basis...")
    start_time = time.time()
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs_normalized, k=1)
    conversion_time = time.time() - start_time
    
    print(f"Conversion time: {conversion_time:.3f} seconds")
    print(f"Bernstein coefficients shape: {bern_coeffs.shape}")
    print(f"Bernstein coefficient range: [{np.min(bern_coeffs):.2e}, {np.max(bern_coeffs):.2e}]")
    
    # Create system with ORIGINAL domain for denormalization
    print("\nCreating polynomial system...")
    system = create_polynomial_system(
        equation_coeffs=[bern_coeffs],
        param_ranges=[(0.0, 25.0)],  # Original domain
        param_names=['x']
    )
    
    # Solve with PP method and tolerance 1e-8
    print("\n" + "=" * 80)
    print("SOLVING WITH PP METHOD (tolerance = 1e-8)")
    print("=" * 80)
    
    start_time = time.time()
    solutions = solve_polynomial_system(
        system,
        method='pp',
        tolerance=1e-8,  # High precision!
        crit=0.8,
        max_depth=100,  # Increased max depth for high precision
        refine=True,
        verbose=True
    )
    solve_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print(f"\nTransformation time: {transform_time:.3f} seconds")
    print(f"Conversion time: {conversion_time:.3f} seconds")
    print(f"Solve time: {solve_time:.3f} seconds")
    print(f"Total time: {transform_time + conversion_time + solve_time:.3f} seconds")
    print(f"\nSolutions found: {len(solutions)}")
    print(f"Expected solutions: {len(roots)}")
    
    # Sort solutions by x value
    solutions_sorted = sorted(solutions, key=lambda s: s['x'])
    
    print("\nSolutions:")
    for i, sol in enumerate(solutions_sorted):
        x = sol['x']
        # Verify by evaluating polynomial
        residual = np.prod([x - r for r in roots])
        print(f"  {i+1:2d}. x = {x:8.5f}  (residual = {residual:12.2e})")
    
    # Check which expected roots were found
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    found_roots = []
    missing_roots = []
    tolerance = 0.1  # Consider a root found if within 0.1 of expected value
    
    for expected_root in roots:
        found = False
        best_match = None
        best_distance = float('inf')
        
        for sol in solutions:
            distance = abs(sol['x'] - expected_root)
            if distance < tolerance:
                found = True
                if distance < best_distance:
                    best_distance = distance
                    best_match = sol['x']
        
        if found:
            found_roots.append((expected_root, best_match, best_distance))
        else:
            missing_roots.append(expected_root)
    
    print(f"\nFound {len(found_roots)} out of {len(roots)} expected roots")
    
    if found_roots:
        print(f"\nFound roots (expected → actual, error):")
        for expected, actual, error in found_roots:
            print(f"  {expected:2d} → {actual:8.5f}  (error = {error:.2e})")
    
    if missing_roots:
        print(f"\n⚠️  Missing roots: {missing_roots}")
    else:
        print("\n✅ All expected roots found!")
    
    # Check for spurious roots
    spurious = []
    for sol in solutions:
        x = sol['x']
        is_spurious = True
        for expected_root in roots:
            if abs(x - expected_root) < tolerance:
                is_spurious = False
                break
        if is_spurious:
            spurious.append(x)
    
    if spurious:
        print(f"\n⚠️  Spurious roots found: {spurious}")
    else:
        print("✅ No spurious roots")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Expected roots: {len(roots)}")
    print(f"Found roots: {len(found_roots)}")
    print(f"Missing roots: {len(missing_roots)}")
    print(f"Spurious roots: {len(spurious)}")
    print(f"Success rate: {len(found_roots) / len(roots) * 100:.1f}%")
    print(f"Total time: {transform_time + conversion_time + solve_time:.3f} seconds")
    
    if len(found_roots) == len(roots) and len(spurious) == 0:
        print("\n🎉 TEST PASSED! All roots found with no spurious roots.")
    else:
        print("\n⚠️  TEST INCOMPLETE: Some roots missing or spurious roots found.")
    
    print("=" * 80)


if __name__ == "__main__":
    test_20_roots_high_precision()

