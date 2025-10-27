"""
Test standalone polynomial system solver.
"""

import numpy as np
from src.intersection.polynomial_solver import (
    create_polynomial_system,
    solve_polynomial_system,
    PolynomialSystem
)
from src.intersection.bernstein import polynomial_nd_to_bernstein


def test_1d_simple():
    """Test 1D system: f(t) = t - 0.5 = 0"""
    print("\n" + "=" * 80)
    print("TEST 1: 1D System - Linear Equation")
    print("=" * 80)
    
    # f(t) = t - 0.5
    # In power basis: [-0.5, 1.0]
    # Convert to Bernstein basis
    power_coeffs = np.array([-0.5, 1.0])
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    print(f"\nEquation: f(t) = t - 0.5 = 0")
    print(f"Power coefficients: {power_coeffs}")
    print(f"Bernstein coefficients: {bern_coeffs}")
    print(f"Expected solution: t = 0.5")
    
    # Create system
    system = create_polynomial_system(
        equation_coeffs=[bern_coeffs],
        param_ranges=[(0.0, 1.0)]
    )
    
    # Solve
    solutions = solve_polynomial_system(system, verbose=True)
    
    print(f"\n--- Results ---")
    print(f"Number of solutions: {len(solutions)}")
    for i, sol in enumerate(solutions):
        print(f"  Solution {i+1}: t = {sol['t']:.8f}")
    
    # Verify
    assert len(solutions) == 1, f"Expected 1 solution, got {len(solutions)}"
    assert abs(solutions[0]['t'] - 0.5) < 1e-4, "Solution not accurate"
    
    print("\n✅ Test 1 passed!")


def test_1d_quadratic():
    """Test 1D system: f(t) = (t - 0.3)(t - 0.7) = 0"""
    print("\n" + "=" * 80)
    print("TEST 2: 1D System - Quadratic with Two Roots")
    print("=" * 80)
    
    # f(t) = (t - 0.3)(t - 0.7) = t^2 - t + 0.21
    power_coeffs = np.array([0.21, -1.0, 1.0])
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    print(f"\nEquation: f(t) = (t - 0.3)(t - 0.7) = 0")
    print(f"Bernstein coefficients: {bern_coeffs}")
    print(f"Expected solutions: t = 0.3, 0.7")
    
    # Create system
    system = create_polynomial_system(
        equation_coeffs=[bern_coeffs],
        param_ranges=[(0.0, 1.0)]
    )
    
    # Solve
    solutions = solve_polynomial_system(system, tolerance=1e-6, verbose=True)
    
    print(f"\n--- Results ---")
    print(f"Number of solutions: {len(solutions)}")
    for i, sol in enumerate(solutions):
        print(f"  Solution {i+1}: t = {sol['t']:.8f}")
    
    # Verify
    assert len(solutions) == 2, f"Expected 2 solutions, got {len(solutions)}"
    
    t_values = sorted([sol['t'] for sol in solutions])
    assert abs(t_values[0] - 0.3) < 1e-3, f"First root not accurate: {t_values[0]}"
    assert abs(t_values[1] - 0.7) < 1e-3, f"Second root not accurate: {t_values[1]}"
    
    print("\n✅ Test 2 passed!")


def test_1d_custom_domain():
    """Test 1D system with custom parameter domain."""
    print("\n" + "=" * 80)
    print("TEST 3: 1D System - Custom Parameter Domain")
    print("=" * 80)

    # For domain [1, 2], we need to express the polynomial in terms of
    # the normalized parameter s in [0, 1] where t = 1 + s
    # f(t) = t - 1.5 = 0 becomes f(s) = (1 + s) - 1.5 = s - 0.5 = 0
    # Expected solution: s = 0.5, which maps to t = 1.5

    # In power basis: f(s) = s - 0.5
    power_coeffs = np.array([-0.5, 1.0])
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)

    print(f"\nEquation: f(t) = t - 1.5 = 0 for t in [1, 2]")
    print(f"Normalized: f(s) = s - 0.5 = 0 for s in [0, 1]")
    print(f"Bernstein coefficients: {bern_coeffs}")
    print(f"Expected solution: t = 1.5 (s = 0.5)")

    # Create system with custom domain
    # The Bernstein coefficients are for the normalized domain [0, 1]
    # The param_ranges specify how to map back to original domain
    system = create_polynomial_system(
        equation_coeffs=[bern_coeffs],
        param_ranges=[(1.0, 2.0)]
    )

    # Solve
    solutions = solve_polynomial_system(system, verbose=True)

    print(f"\n--- Results ---")
    print(f"Number of solutions: {len(solutions)}")
    for i, sol in enumerate(solutions):
        print(f"  Solution {i+1}: t = {sol['t']:.8f}")

    # Verify
    assert len(solutions) == 1, f"Expected 1 solution, got {len(solutions)}"
    assert abs(solutions[0]['t'] - 1.5) < 1e-4, "Solution not accurate"

    print("\n✅ Test 3 passed!")


def test_2d_simple():
    """Test 2D system: f1(u,v) = u - 0.5 = 0, f2(u,v) = v - 0.5 = 0"""
    print("\n" + "=" * 80)
    print("TEST 4: 2D System - Two Linear Equations")
    print("=" * 80)
    
    # f1(u,v) = u - 0.5
    # In power basis: [[-0.5, 0], [1.0, 0]]
    power_coeffs_1 = np.array([[-0.5, 0.0], [1.0, 0.0]])
    bern_coeffs_1 = polynomial_nd_to_bernstein(power_coeffs_1, k=2)
    
    # f2(u,v) = v - 0.5
    # In power basis: [[-0.5, 1.0], [0, 0]]
    power_coeffs_2 = np.array([[-0.5, 1.0], [0.0, 0.0]])
    bern_coeffs_2 = polynomial_nd_to_bernstein(power_coeffs_2, k=2)
    
    print(f"\nEquation 1: f1(u,v) = u - 0.5 = 0")
    print(f"  Bernstein coefficients shape: {bern_coeffs_1.shape}")
    
    print(f"\nEquation 2: f2(u,v) = v - 0.5 = 0")
    print(f"  Bernstein coefficients shape: {bern_coeffs_2.shape}")
    
    print(f"\nExpected solution: u = 0.5, v = 0.5")
    
    # Create system
    system = create_polynomial_system(
        equation_coeffs=[bern_coeffs_1, bern_coeffs_2],
        param_ranges=[(0.0, 1.0), (0.0, 1.0)],
        param_names=['u', 'v']
    )
    
    # Solve
    solutions = solve_polynomial_system(system, tolerance=1e-6, verbose=True)
    
    print(f"\n--- Results ---")
    print(f"Number of solutions: {len(solutions)}")
    for i, sol in enumerate(solutions):
        print(f"  Solution {i+1}: u = {sol['u']:.8f}, v = {sol['v']:.8f}")
    
    # Verify
    assert len(solutions) == 1, f"Expected 1 solution, got {len(solutions)}"
    assert abs(solutions[0]['u'] - 0.5) < 1e-3, "u not accurate"
    assert abs(solutions[0]['v'] - 0.5) < 1e-3, "v not accurate"
    
    print("\n✅ Test 4 passed!")


def test_no_solution():
    """Test system with no solutions."""
    print("\n" + "=" * 80)
    print("TEST 5: System with No Solutions")
    print("=" * 80)
    
    # f(t) = t^2 + 1 = 0 (no real roots)
    power_coeffs = np.array([1.0, 0.0, 1.0])
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    print(f"\nEquation: f(t) = t^2 + 1 = 0")
    print(f"Bernstein coefficients: {bern_coeffs}")
    print(f"Expected: No solutions")
    
    # Create system
    system = create_polynomial_system(
        equation_coeffs=[bern_coeffs],
        param_ranges=[(0.0, 1.0)]
    )
    
    # Solve
    solutions = solve_polynomial_system(system, verbose=True)
    
    print(f"\n--- Results ---")
    print(f"Number of solutions: {len(solutions)}")
    
    # Verify
    assert len(solutions) == 0, f"Expected 0 solutions, got {len(solutions)}"
    
    print("\n✅ Test 5 passed!")


def test_polynomial_system_class():
    """Test PolynomialSystem class."""
    print("\n" + "=" * 80)
    print("TEST 6: PolynomialSystem Class")
    print("=" * 80)
    
    # Create system using class directly
    bern_coeffs = np.array([-0.5, 0.5])
    
    system = PolynomialSystem(
        equation_coeffs=[bern_coeffs],
        param_ranges=[(0.0, 1.0)],
        k=1,
        degree=1,
        param_names=['t'],
        metadata={'description': 'Test system'}
    )
    
    print(f"\nSystem created:")
    print(f"  k = {system.k}")
    print(f"  degree = {system.degree}")
    print(f"  param_names = {system.param_names}")
    print(f"  param_ranges = {system.param_ranges}")
    print(f"  metadata = {system.metadata}")
    
    # Solve
    solutions = solve_polynomial_system(system, verbose=False)
    
    print(f"\nSolutions: {solutions}")
    
    assert len(solutions) == 1
    assert abs(solutions[0]['t'] - 0.5) < 1e-4
    
    print("\n✅ Test 6 passed!")


if __name__ == "__main__":
    test_1d_simple()
    test_1d_quadratic()
    test_1d_custom_domain()
    test_2d_simple()
    test_no_solution()
    test_polynomial_system_class()
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✅")
    print("=" * 80)
    print("\nStandalone polynomial solver is working correctly!")
    print("=" * 80)

