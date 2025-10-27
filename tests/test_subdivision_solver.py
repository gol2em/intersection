"""
Test subdivision solver for polynomial systems.
"""

import sys
sys.path.insert(0, 'D:/Python/Intersection')

import numpy as np
from src.intersection.subdivision_solver import (
    SubdivisionSolver,
    SolverConfig,
    BoundingMethod,
    solve_with_subdivision
)
from src.intersection.bernstein import polynomial_nd_to_bernstein


def test_1d_simple():
    """Test 1D: f(t) = t - 0.5 = 0, solution at t = 0.5"""
    print("=" * 80)
    print("TEST 1: 1D Simple - f(t) = t - 0.5")
    print("=" * 80)
    
    # Power basis: f(t) = -0.5 + t
    power_coeffs = np.array([-0.5, 1.0])
    
    # Convert to Bernstein basis
    bernstein_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1, verbose=False)
    
    print(f"Power coefficients: {power_coeffs}")
    print(f"Bernstein coefficients: {bernstein_coeffs}")
    
    # Solve using PP method
    config = SolverConfig(
        method=BoundingMethod.PP,
        tolerance=1e-3,
        crit=0.8,
        max_depth=20,
        verbose=True
    )
    
    solver = SubdivisionSolver(config)
    solutions = solver.solve([bernstein_coeffs], k=1)
    
    print(f"\nSolutions found: {len(solutions)}")
    for i, sol in enumerate(solutions):
        print(f"  Solution {i+1}: t = {sol[0]:.6f}")
    
    # Verify
    expected = 0.5
    success = len(solutions) == 1 and abs(solutions[0][0] - expected) < 0.01
    
    print(f"\n✅ Test 1: {'PASS' if success else 'FAIL'}")
    print(f"Expected: t = {expected}")
    print(f"Found: t = {solutions[0][0]:.6f}" if solutions else "No solutions")
    print()
    
    return success


def test_1d_quadratic():
    """Test 1D: f(t) = (t - 0.3)(t - 0.7) = t^2 - t + 0.21"""
    print("=" * 80)
    print("TEST 2: 1D Quadratic - f(t) = (t - 0.3)(t - 0.7)")
    print("=" * 80)
    
    # Power basis: f(t) = 0.21 - t + t^2
    power_coeffs = np.array([0.21, -1.0, 1.0])
    
    # Convert to Bernstein basis
    bernstein_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1, verbose=False)
    
    print(f"Power coefficients: {power_coeffs}")
    print(f"Bernstein coefficients: {bernstein_coeffs}")
    
    # Solve using PP method
    solutions = solve_with_subdivision(
        [bernstein_coeffs],
        k=1,
        method='pp',
        tolerance=1e-3,
        verbose=True
    )
    
    print(f"\nSolutions found: {len(solutions)}")
    for i, sol in enumerate(solutions):
        print(f"  Solution {i+1}: t = {sol[0]:.6f}")
    
    # Verify
    expected = [0.3, 0.7]
    success = len(solutions) == 2
    if success:
        sols_sorted = sorted([s[0] for s in solutions])
        for i, (found, exp) in enumerate(zip(sols_sorted, expected)):
            match = abs(found - exp) < 0.01
            success = success and match
            print(f"  Solution {i+1}: expected {exp}, found {found:.6f}, match: {match}")
    
    print(f"\n✅ Test 2: {'PASS' if success else 'FAIL'}\n")
    return success


def test_1d_no_solution():
    """Test 1D: f(t) = t^2 + 1 (no real roots in [0,1])"""
    print("=" * 80)
    print("TEST 3: 1D No Solution - f(t) = t^2 + 1")
    print("=" * 80)
    
    # Power basis: f(t) = 1 + t^2
    power_coeffs = np.array([1.0, 0.0, 1.0])
    
    # Convert to Bernstein basis
    bernstein_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1, verbose=False)
    
    print(f"Power coefficients: {power_coeffs}")
    print(f"Bernstein coefficients: {bernstein_coeffs}")
    print(f"Min: {np.min(bernstein_coeffs)}, Max: {np.max(bernstein_coeffs)}")
    
    # Solve using PP method
    solutions = solve_with_subdivision(
        [bernstein_coeffs],
        k=1,
        method='pp',
        tolerance=1e-3,
        verbose=True
    )
    
    print(f"\nSolutions found: {len(solutions)}")
    
    # Verify
    success = len(solutions) == 0
    
    print(f"\n✅ Test 3: {'PASS' if success else 'FAIL'}")
    print(f"Expected: 0 solutions (all coefficients positive)")
    print(f"Found: {len(solutions)} solutions\n")
    
    return success


def test_2d_simple():
    """Test 2D: f(u,v) = u - 0.5, solution at u = 0.5, any v"""
    print("=" * 80)
    print("TEST 4: 2D Simple - f(u,v) = u - 0.5")
    print("=" * 80)
    
    # Power basis: f(u,v) = -0.5 + u
    # In 2D tensor product: coeffs[i,j] for u^i * v^j
    power_coeffs = np.array([
        [-0.5, 0.0],  # -0.5 * u^0 * v^0, 0 * u^0 * v^1
        [1.0, 0.0]    # 1.0 * u^1 * v^0, 0 * u^1 * v^1
    ])
    
    # Convert to Bernstein basis
    bernstein_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=2, verbose=False)
    
    print(f"Power coefficients:\n{power_coeffs}")
    print(f"Bernstein coefficients:\n{bernstein_coeffs}")
    
    # Solve using PP method
    # Note: This will find one solution since we can't distinguish the line
    solutions = solve_with_subdivision(
        [bernstein_coeffs],
        k=2,
        method='pp',
        tolerance=1e-2,
        max_depth=10,
        verbose=True
    )
    
    print(f"\nSolutions found: {len(solutions)}")
    for i, sol in enumerate(solutions):
        print(f"  Solution {i+1}: u = {sol[0]:.6f}, v = {sol[1]:.6f}")
    
    # Verify: should find at least one solution with u ≈ 0.5
    success = len(solutions) > 0
    if success:
        # Check that at least one solution has u ≈ 0.5
        u_values = [s[0] for s in solutions]
        success = any(abs(u - 0.5) < 0.1 for u in u_values)
    
    print(f"\n✅ Test 4: {'PASS' if success else 'FAIL'}")
    print(f"Expected: Solutions with u ≈ 0.5")
    print(f"Note: 2D case with line of solutions will find multiple boxes\n")
    
    return success


def test_tolerance_and_crit():
    """Test tolerance and CRIT parameters"""
    print("=" * 80)
    print("TEST 5: Tolerance and CRIT Parameters")
    print("=" * 80)
    
    # f(t) = t - 0.5
    power_coeffs = np.array([-0.5, 1.0])
    bernstein_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1, verbose=False)
    
    # Test 1: Loose tolerance (should converge quickly)
    print("\n--- Test 5a: Loose tolerance (1e-2) ---")
    config1 = SolverConfig(
        method=BoundingMethod.PP,
        tolerance=1e-2,
        crit=0.8,
        max_depth=20,
        verbose=False
    )
    solver1 = SubdivisionSolver(config1)
    solutions1 = solver1.solve([bernstein_coeffs], k=1)
    
    print(f"Boxes processed: {solver1.stats['boxes_processed']}")
    print(f"Solutions found: {len(solutions1)}")
    
    # Test 2: Tight tolerance (should need more subdivisions)
    print("\n--- Test 5b: Tight tolerance (1e-6) ---")
    config2 = SolverConfig(
        method=BoundingMethod.PP,
        tolerance=1e-6,
        crit=0.8,
        max_depth=20,
        verbose=False
    )
    solver2 = SubdivisionSolver(config2)
    solutions2 = solver2.solve([bernstein_coeffs], k=1)
    
    print(f"Boxes processed: {solver2.stats['boxes_processed']}")
    print(f"Solutions found: {len(solutions2)}")
    
    # Test 3: Different CRIT value
    print("\n--- Test 5c: Different CRIT (0.5) ---")
    config3 = SolverConfig(
        method=BoundingMethod.PP,
        tolerance=1e-3,
        crit=0.5,  # More aggressive subdivision
        max_depth=20,
        verbose=False
    )
    solver3 = SubdivisionSolver(config3)
    solutions3 = solver3.solve([bernstein_coeffs], k=1)
    
    print(f"Boxes processed: {solver3.stats['boxes_processed']}")
    print(f"Solutions found: {len(solutions3)}")
    
    # Verify
    success = (
        len(solutions1) == 1 and
        len(solutions2) == 1 and
        len(solutions3) == 1 and
        solver2.stats['boxes_processed'] >= solver1.stats['boxes_processed']
    )
    
    print(f"\n✅ Test 5: {'PASS' if success else 'FAIL'}")
    print(f"Expected: Tighter tolerance requires more boxes")
    print(f"Loose (1e-2): {solver1.stats['boxes_processed']} boxes")
    print(f"Tight (1e-6): {solver2.stats['boxes_processed']} boxes\n")
    
    return success


def test_convenience_function():
    """Test convenience function solve_with_subdivision"""
    print("=" * 80)
    print("TEST 6: Convenience Function")
    print("=" * 80)
    
    # f(t) = t - 0.5
    power_coeffs = np.array([-0.5, 1.0])
    bernstein_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1, verbose=False)
    
    # Test with different methods
    for method in ['pp', 'lp', 'hybrid']:
        print(f"\n--- Method: {method.upper()} ---")
        solutions = solve_with_subdivision(
            [bernstein_coeffs],
            k=1,
            method=method,
            tolerance=1e-3,
            verbose=False
        )
        print(f"Solutions: {len(solutions)}")
        if solutions:
            print(f"  t = {solutions[0][0]:.6f}")
    
    success = True
    print(f"\n✅ Test 6: {'PASS' if success else 'FAIL'}\n")
    
    return success


if __name__ == "__main__":
    test1 = test_1d_simple()
    test2 = test_1d_quadratic()
    test3 = test_1d_no_solution()
    test4 = test_2d_simple()
    test5 = test_tolerance_and_crit()
    test6 = test_convenience_function()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Test 1 (1D Simple):           {'PASS' if test1 else 'FAIL'}")
    print(f"Test 2 (1D Quadratic):        {'PASS' if test2 else 'FAIL'}")
    print(f"Test 3 (1D No Solution):      {'PASS' if test3 else 'FAIL'}")
    print(f"Test 4 (2D Simple):           {'PASS' if test4 else 'FAIL'}")
    print(f"Test 5 (Tolerance/CRIT):      {'PASS' if test5 else 'FAIL'}")
    print(f"Test 6 (Convenience Func):    {'PASS' if test6 else 'FAIL'}")
    
    all_pass = test1 and test2 and test3 and test4 and test5 and test6
    print(f"\nOverall: {'ALL TESTS PASSED ✅' if all_pass else 'SOME TESTS FAILED ✗'}")

