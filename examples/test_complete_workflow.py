"""
Test the complete polynomial system solver workflow.

Workflow:
1. Given a system in power basis and domain
2. Convert to Bernstein basis and normalize
3. Solve with given tolerance
4. (Optional, default True) Newton refine
5. Denormalize all solutions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from intersection.bernstein import (
    polynomial_nd_to_bernstein,
    transform_polynomial_domain_1d
)
from intersection.polynomial_solver import create_polynomial_system, solve_polynomial_system


def test_1d_workflow():
    """
    Test 1D workflow: f(t) = (t - 0.3)(t - 0.7) = 0
    Domain: t ∈ [0, 1]
    Expected solutions: t = 0.3, 0.7
    """
    print("\n" + "="*80)
    print("TEST 1: 1D Quadratic - Complete Workflow")
    print("="*80)
    
    # Step 1: Define system in power basis
    print("\nStep 1: Define system in power basis")
    print("  Equation: f(t) = (t - 0.3)(t - 0.7) = t^2 - t + 0.21")
    print("  Domain: t ∈ [0, 1]")
    
    # Power basis: f(t) = 0.21 - 1.0*t + 1.0*t^2
    power_coeffs = np.array([0.21, -1.0, 1.0])
    print(f"  Power coefficients: {power_coeffs}")
    
    # Step 2: Convert to Bernstein basis
    print("\nStep 2: Convert to Bernstein basis")
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    print(f"  Bernstein coefficients: {bern_coeffs}")
    
    # Step 3: Create polynomial system
    print("\nStep 3: Create polynomial system")
    system = create_polynomial_system(
        equation_coeffs=[bern_coeffs],
        param_ranges=[(0.0, 1.0)],
        param_names=['t']
    )
    print(f"  System created: {system.k}D, degree {system.degree}")
    
    # Step 4: Solve (includes normalization, solving, refinement, denormalization)
    print("\nStep 4: Solve with complete workflow")
    solutions = solve_polynomial_system(
        system,
        method='pp',
        tolerance=1e-6,
        refine=True,
        verbose=True
    )
    
    # Verify
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    print(f"Expected solutions: t = 0.3, 0.7")
    print(f"Found {len(solutions)} solutions:")
    for i, sol in enumerate(solutions):
        t = sol['t']
        residual = t**2 - t + 0.21
        print(f"  Solution {i+1}: t = {t:.6f}, residual = {residual:.2e}")
    
    # Check correctness
    assert len(solutions) == 2, f"Expected 2 solutions, got {len(solutions)}"
    t_values = sorted([sol['t'] for sol in solutions])
    assert abs(t_values[0] - 0.3) < 1e-6, f"Expected t=0.3, got {t_values[0]}"
    assert abs(t_values[1] - 0.7) < 1e-6, f"Expected t=0.7, got {t_values[1]}"
    print("\n✅ Test PASSED!")
    
    return solutions


def test_2d_workflow():
    """
    Test 2D workflow: Circle-Ellipse intersection
    System:
        x^2 + y^2 - 1 = 0  (circle)
        x^2/4 + 4y^2 - 1 = 0  (ellipse)
    Domain: x ∈ [0, 1], y ∈ [0, 1]
    Expected solution: (x, y) ≈ (0.894427, 0.447214)
    """
    print("\n" + "="*80)
    print("TEST 2: 2D Circle-Ellipse - Complete Workflow")
    print("="*80)
    
    # Step 1: Define system in power basis
    print("\nStep 1: Define system in power basis")
    print("  Equation 1: x^2 + y^2 - 1 = 0")
    print("  Equation 2: x^2/4 + 4y^2 - 1 = 0")
    print("  Domain: x ∈ [0, 1], y ∈ [0, 1]")
    
    # Circle: x^2 + y^2 - 1 = 0
    # Power basis: coeffs[i][j] for x^i * y^j
    circle_power = np.array([
        [-1.0, 0.0, 1.0],   # -1 + y^2
        [ 0.0, 0.0, 0.0],   # 0
        [ 1.0, 0.0, 0.0]    # x^2
    ])
    
    # Ellipse: x^2/4 + 4y^2 - 1 = 0
    ellipse_power = np.array([
        [-1.0, 0.0, 4.0],   # -1 + 4y^2
        [ 0.0, 0.0, 0.0],   # 0
        [ 0.25, 0.0, 0.0]   # x^2/4
    ])
    
    print(f"  Circle power coefficients:\n{circle_power}")
    print(f"  Ellipse power coefficients:\n{ellipse_power}")
    
    # Step 2: Convert to Bernstein basis
    print("\nStep 2: Convert to Bernstein basis")
    circle_bern = polynomial_nd_to_bernstein(circle_power, k=2)
    ellipse_bern = polynomial_nd_to_bernstein(ellipse_power, k=2)
    print(f"  Circle Bernstein coefficients:\n{circle_bern}")
    print(f"  Ellipse Bernstein coefficients:\n{ellipse_bern}")
    
    # Step 3: Create polynomial system
    print("\nStep 3: Create polynomial system")
    system = create_polynomial_system(
        equation_coeffs=[circle_bern, ellipse_bern],
        param_ranges=[(0.0, 1.0), (0.0, 1.0)],
        param_names=['x', 'y']
    )
    print(f"  System created: {system.k}D, degree {system.degree}")
    
    # Step 4: Solve (includes normalization, solving, refinement, denormalization)
    print("\nStep 4: Solve with complete workflow")
    solutions = solve_polynomial_system(
        system,
        method='pp',
        tolerance=1e-6,
        refine=True,
        verbose=True
    )
    
    # Verify
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    print(f"Expected solution: x ≈ 0.894427, y ≈ 0.447214")
    print(f"Found {len(solutions)} solutions:")
    for i, sol in enumerate(solutions):
        x, y = sol['x'], sol['y']
        circle_residual = x**2 + y**2 - 1
        ellipse_residual = x**2/4 + 4*y**2 - 1
        print(f"  Solution {i+1}: x = {x:.6f}, y = {y:.6f}")
        print(f"    Circle residual: {circle_residual:.2e}")
        print(f"    Ellipse residual: {ellipse_residual:.2e}")
    
    # Check correctness
    assert len(solutions) == 1, f"Expected 1 solution, got {len(solutions)}"
    x, y = solutions[0]['x'], solutions[0]['y']
    assert abs(x - 0.894427) < 1e-3, f"Expected x≈0.894427, got {x}"
    assert abs(y - 0.447214) < 1e-3, f"Expected y≈0.447214, got {y}"
    print("\n✅ Test PASSED!")
    
    return solutions


def test_custom_domain():
    """
    Test with custom domain (not [0,1]).
    System: f(x) = x - 5 = 0
    Domain: x ∈ [2, 8]
    Expected solution: x = 5

    This test demonstrates the RECOMMENDED workflow using the helper function.
    """
    print("\n" + "="*80)
    print("TEST 3: Custom Domain - Complete Workflow (with Helper)")
    print("="*80)

    # Step 1: Define system in power basis on ORIGINAL domain
    print("\nStep 1: Define system in power basis on original domain")
    print("  Equation: f(x) = x - 5 = 0")
    print("  Domain: x ∈ [2, 8]")

    # Power basis: f(x) = -5 + 1*x
    power_coeffs_original = np.array([-5.0, 1.0])
    original_domain = (2.0, 8.0)
    print(f"  Power coefficients: {power_coeffs_original}")

    # Step 1b: Transform to [0,1] domain using helper function
    print("\nStep 1b: Transform to [0,1] domain using helper")
    power_coeffs = transform_polynomial_domain_1d(
        power_coeffs_original,
        from_range=original_domain,
        to_range=(0, 1),
        verbose=True
    )

    # Step 2: Convert to Bernstein basis
    print("\nStep 2: Convert to Bernstein basis")
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    print(f"  Bernstein coefficients: {bern_coeffs}")
    
    # Step 3: Create polynomial system with ORIGINAL domain
    print("\nStep 3: Create polynomial system")
    system = create_polynomial_system(
        equation_coeffs=[bern_coeffs],
        param_ranges=[original_domain],  # Use ORIGINAL domain for denormalization!
        param_names=['x']
    )
    print(f"  System created: {system.k}D, degree {system.degree}")
    print(f"  Domain: x ∈ {system.param_ranges[0]}")
    
    # Step 4: Solve (includes normalization, solving, refinement, denormalization)
    print("\nStep 4: Solve with complete workflow")
    solutions = solve_polynomial_system(
        system,
        method='pp',
        tolerance=1e-6,
        refine=True,
        verbose=True
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
    
    # Check correctness
    assert len(solutions) == 1, f"Expected 1 solution, got {len(solutions)}"
    x = solutions[0]['x']
    assert abs(x - 5.0) < 1e-6, f"Expected x=5.0, got {x}"
    print("\n✅ Test PASSED!")
    
    return solutions


def main():
    """Run all workflow tests."""
    print("="*80)
    print("COMPLETE WORKFLOW TESTS")
    print("="*80)
    print("\nTesting the complete polynomial system solver workflow:")
    print("1. Given a system in power basis and domain")
    print("2. Convert to Bernstein basis and normalize")
    print("3. Solve with given tolerance")
    print("4. (Optional, default True) Newton refine")
    print("5. Denormalize all solutions")
    
    # Run tests
    test_1d_workflow()
    test_2d_workflow()
    test_custom_domain()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED! ✅")
    print("="*80)
    print("\nThe complete workflow is working correctly:")
    print("  ✅ Power basis → Bernstein basis conversion")
    print("  ✅ Normalization to [0,1]^k")
    print("  ✅ Subdivision solving with PP method")
    print("  ✅ Newton refinement")
    print("  ✅ Denormalization to original domain")
    print("  ✅ Works with custom domains (not just [0,1])")


if __name__ == "__main__":
    main()

