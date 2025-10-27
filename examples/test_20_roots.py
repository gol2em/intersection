"""
Test: Solve polynomial with 20 roots using PP method

Polynomial: (x-1)(x-2)(x-3)...(x-20) = 0
Expected roots: x = 1, 2, 3, ..., 20
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.intersection.polynomial_solver import (
    create_polynomial_system,
    solve_polynomial_system
)
from src.intersection.bernstein import polynomial_nd_to_bernstein
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


def test_20_roots():
    """Test polynomial with 20 roots."""
    print("\n" + "=" * 80)
    print("TEST: Polynomial with 20 Roots")
    print("=" * 80)
    
    # Define roots
    roots = list(range(1, 21))  # [1, 2, 3, ..., 20]
    print(f"\nExpected roots: {roots}")
    
    # Expand polynomial
    print("\nExpanding polynomial (x-1)(x-2)...(x-20)...")
    power_coeffs = expand_polynomial_product(roots)
    
    print(f"Polynomial degree: {len(power_coeffs) - 1}")
    print(f"Leading coefficient: {power_coeffs[-1]:.2e}")
    print(f"Constant term: {power_coeffs[0]:.2e}")
    
    # For domain [0, 25], we need to normalize to [0, 1]
    # x = 0 + 25*s where s in [0, 1]
    # We need to express the polynomial in terms of s
    # p(x) = p(25*s)
    
    # Substitute x = 25*s into the polynomial
    print("\nNormalizing to domain [0, 25]...")
    # Create polynomial in s: p(25*s)
    normalized_coeffs = np.zeros_like(power_coeffs)
    for i, coeff in enumerate(power_coeffs):
        # x^i = (25*s)^i = 25^i * s^i
        normalized_coeffs[i] = coeff * (25.0 ** i)
    
    print(f"Normalized constant term: {normalized_coeffs[0]:.2e}")
    print(f"Normalized leading coefficient: {normalized_coeffs[-1]:.2e}")
    
    # Convert to Bernstein basis
    print("\nConverting to Bernstein basis...")
    start_time = time.time()
    bern_coeffs = polynomial_nd_to_bernstein(normalized_coeffs, k=1)
    conversion_time = time.time() - start_time
    
    print(f"Conversion time: {conversion_time:.3f} seconds")
    print(f"Bernstein coefficients shape: {bern_coeffs.shape}")
    print(f"Bernstein coefficient range: [{np.min(bern_coeffs):.2e}, {np.max(bern_coeffs):.2e}]")
    
    # Create system
    print("\nCreating polynomial system...")
    system = create_polynomial_system(
        equation_coeffs=[bern_coeffs],
        param_ranges=[(0.0, 25.0)],
        param_names=['x']
    )
    
    # Solve with PP method
    print("\n" + "=" * 80)
    print("SOLVING WITH PP METHOD")
    print("=" * 80)
    
    start_time = time.time()
    solutions = solve_polynomial_system(
        system,
        method='pp',
        tolerance=1e-3,  # Relaxed tolerance for high-degree polynomial
        crit=0.8,
        max_depth=50,  # Increased max depth
        refine=True,
        verbose=True
    )
    solve_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print(f"\nSolve time: {solve_time:.3f} seconds")
    print(f"Solutions found: {len(solutions)}")
    print(f"Expected solutions: {len(roots)}")
    
    # Sort solutions by x value
    solutions_sorted = sorted(solutions, key=lambda s: s['x'])
    
    print("\nSolutions:")
    for i, sol in enumerate(solutions_sorted):
        x = sol['x']
        # Verify by evaluating polynomial
        residual = np.prod([x - r for r in roots])
        print(f"  {i+1:2d}. x = {x:6.3f}  (residual = {residual:10.2e})")
    
    # Check which expected roots were found
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    found_roots = []
    missing_roots = []
    tolerance = 0.5  # Consider a root found if within 0.5 of expected value
    
    for expected_root in roots:
        found = False
        for sol in solutions:
            if abs(sol['x'] - expected_root) < tolerance:
                found = True
                found_roots.append(expected_root)
                break
        if not found:
            missing_roots.append(expected_root)
    
    print(f"\nFound {len(found_roots)} out of {len(roots)} expected roots")
    
    if found_roots:
        print(f"\nFound roots: {found_roots}")
    
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
    print(f"Total time: {conversion_time + solve_time:.3f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    test_20_roots()

