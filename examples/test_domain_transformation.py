"""
Test domain transformation helper functions.

These functions help users transform polynomials from custom domains to [0,1]^k
before converting to Bernstein basis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from intersection.bernstein import (
    transform_polynomial_domain_1d,
    transform_polynomial_domain_2d,
    polynomial_nd_to_bernstein
)
from intersection.polynomial_solver import create_polynomial_system, solve_polynomial_system


def test_1d_domain_transformation():
    """
    Test 1D domain transformation.
    
    Original: f(x) = x - 5 on domain x ∈ [2, 8]
    Expected: f(s) = 6*s - 3 on domain s ∈ [0, 1]
    """
    print("\n" + "="*80)
    print("TEST 1: 1D Domain Transformation")
    print("="*80)
    
    # Original polynomial: f(x) = -5 + 1*x
    power_coeffs = np.array([-5.0, 1.0])
    print(f"\nOriginal polynomial: f(x) = {power_coeffs[0]} + {power_coeffs[1]}*x")
    print(f"Original domain: x ∈ [2, 8]")
    
    # Transform to [0, 1]
    transformed = transform_polynomial_domain_1d(
        power_coeffs, 
        from_range=(2, 8), 
        to_range=(0, 1),
        verbose=True
    )
    
    print(f"\nTransformed polynomial: f(s) = {transformed[0]} + {transformed[1]}*s")
    print(f"Transformed domain: s ∈ [0, 1]")
    
    # Verify
    expected = np.array([-3.0, 6.0])
    assert np.allclose(transformed, expected), f"Expected {expected}, got {transformed}"
    
    # Verify that f(x=5) = f(s=0.5) = 0
    # x = 2 + 6*s, so x=5 means s=0.5
    x_val = 5.0
    s_val = (x_val - 2) / 6
    
    f_x = power_coeffs[0] + power_coeffs[1] * x_val
    f_s = transformed[0] + transformed[1] * s_val
    
    print(f"\nVerification:")
    print(f"  f(x={x_val}) = {f_x}")
    print(f"  f(s={s_val}) = {f_s}")
    assert np.isclose(f_x, f_s), f"Values don't match: f(x)={f_x}, f(s)={f_s}"
    
    print("\n✅ Test PASSED!")
    return transformed


def test_1d_complete_workflow_with_helper():
    """
    Test complete workflow using the domain transformation helper.
    
    This is the RECOMMENDED way to use custom domains.
    """
    print("\n" + "="*80)
    print("TEST 2: 1D Complete Workflow with Helper Function")
    print("="*80)
    
    # Step 1: Define polynomial in power basis on custom domain
    print("\nStep 1: Define polynomial on custom domain")
    print("  Equation: f(x) = x - 5 = 0")
    print("  Domain: x ∈ [2, 8]")
    
    power_coeffs_original = np.array([-5.0, 1.0])
    original_domain = (2.0, 8.0)
    
    # Step 2: Transform to [0,1] domain
    print("\nStep 2: Transform to [0,1] domain using helper")
    power_coeffs_normalized = transform_polynomial_domain_1d(
        power_coeffs_original,
        from_range=original_domain,
        to_range=(0, 1),
        verbose=True
    )
    
    # Step 3: Convert to Bernstein basis
    print("\nStep 3: Convert to Bernstein basis")
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs_normalized, k=1)
    print(f"  Bernstein coefficients: {bern_coeffs}")
    
    # Step 4: Create system with ORIGINAL domain for denormalization
    print("\nStep 4: Create polynomial system")
    system = create_polynomial_system(
        equation_coeffs=[bern_coeffs],
        param_ranges=[original_domain],  # Use original domain!
        param_names=['x']
    )
    
    # Step 5: Solve
    print("\nStep 5: Solve")
    solutions = solve_polynomial_system(
        system,
        method='pp',
        tolerance=1e-6,
        refine=True,
        verbose=False
    )
    
    # Verify
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    print(f"Expected solution: x = 5.0")
    print(f"Found {len(solutions)} solutions:")
    for i, sol in enumerate(solutions):
        x = sol['x']
        residual = x - 5.0
        print(f"  Solution {i+1}: x = {x:.6f}, residual = {residual:.2e}")
    
    assert len(solutions) == 1, f"Expected 1 solution, got {len(solutions)}"
    assert np.isclose(solutions[0]['x'], 5.0), f"Expected x=5.0, got {solutions[0]['x']}"
    
    print("\n✅ Test PASSED!")
    return solutions


def test_2d_domain_transformation():
    """
    Test 2D domain transformation.
    """
    print("\n" + "="*80)
    print("TEST 3: 2D Domain Transformation")
    print("="*80)
    
    # Original polynomial: f(x,y) = x + y
    # Coefficients: f(x,y) = 0 + 1*y + 1*x + 0*x*y
    power_coeffs = np.array([
        [0.0, 1.0],  # 0 + 1*y
        [1.0, 0.0]   # 1*x + 0*x*y
    ])
    
    print(f"\nOriginal polynomial: f(x,y) = x + y")
    print(f"Original domain: x ∈ [2, 8], y ∈ [3, 7]")
    print(f"Power coefficients:\n{power_coeffs}")
    
    # Transform to [0, 1] × [0, 1]
    transformed = transform_polynomial_domain_2d(
        power_coeffs,
        from_ranges=[(2, 8), (3, 7)],
        to_ranges=[(0, 1), (0, 1)],
        verbose=True
    )
    
    print(f"\nTransformed polynomial coefficients:\n{transformed}")
    
    # Verify at a point
    # x = 2 + 6*s, y = 3 + 4*t
    # At (x, y) = (5, 5): s = 0.5, t = 0.5
    x_val, y_val = 5.0, 5.0
    s_val = (x_val - 2) / 6
    t_val = (y_val - 3) / 4
    
    # Evaluate original polynomial
    f_xy = power_coeffs[0, 0] + power_coeffs[0, 1] * y_val + \
           power_coeffs[1, 0] * x_val + power_coeffs[1, 1] * x_val * y_val
    
    # Evaluate transformed polynomial
    f_st = transformed[0, 0] + transformed[0, 1] * t_val + \
           transformed[1, 0] * s_val + transformed[1, 1] * s_val * t_val
    
    print(f"\nVerification at (x, y) = ({x_val}, {y_val}):")
    print(f"  Corresponding (s, t) = ({s_val}, {t_val})")
    print(f"  f(x, y) = {f_xy}")
    print(f"  f(s, t) = {f_st}")
    
    assert np.isclose(f_xy, f_st), f"Values don't match: f(x,y)={f_xy}, f(s,t)={f_st}"
    
    print("\n✅ Test PASSED!")
    return transformed


def main():
    """Run all domain transformation tests."""
    print("="*80)
    print("DOMAIN TRANSFORMATION TESTS")
    print("="*80)
    print("\nTesting helper functions for transforming polynomials")
    print("from custom domains to [0,1]^k before Bernstein conversion.")
    
    # Run tests
    test_1d_domain_transformation()
    test_1d_complete_workflow_with_helper()
    test_2d_domain_transformation()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED! ✅")
    print("="*80)
    print("\nThe domain transformation helper functions work correctly!")
    print("\nRECOMMENDED WORKFLOW for custom domains:")
    print("  1. Define polynomial in power basis on custom domain")
    print("  2. Use transform_polynomial_domain_*d() to transform to [0,1]^k")
    print("  3. Convert transformed polynomial to Bernstein basis")
    print("  4. Create system with ORIGINAL param_ranges (for denormalization)")
    print("  5. Solve and get solutions in original domain")


if __name__ == "__main__":
    main()

