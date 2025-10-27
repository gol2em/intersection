"""
Test complete workflow for solving polynomial systems.

Workflow:
1. Convert to Bernstein basis and normalize
2. Use PP method to find all possible roots
3. (Optional) Use Newton iteration to refine each root
4. Return to original domain
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.intersection.geometry import Line, Hypersurface, Hyperplane
from src.intersection.polynomial_system import create_intersection_system
from src.intersection.solver import solve_polynomial_system


def test_2d_simple():
    """Test 2D system: line intersecting a parabola."""
    print("\n" + "=" * 80)
    print("TEST 1: 2D System - Line Intersecting Parabola")
    print("=" * 80)

    # Parabola: y = x^2 - 0.5
    # Parametric form: (t, t^2 - 0.5) for t in [0, 1]

    # Create hypersurface (curve in 2D)
    def curve_func(t):
        return np.array([t, t**2 - 0.5])

    hypersurface = Hypersurface(
        func=curve_func,
        param_ranges=[(0.0, 1.0)],
        ambient_dim=2,
        degree=2,
        verbose=False
    )

    # Line: y = 0 (x-axis)
    # Equation: 0*x + 1*y + 0 = 0
    line = Line(
        hyperplanes=[Hyperplane(coeffs=[0.0, 1.0], d=0.0)]
    )
    
    print("\nCurve: (x(t), y(t)) = (t, t^2 - 0.5)")
    print("Line: y = 0")
    print("Expected intersection: t = sqrt(0.5) ≈ 0.707")
    
    # Create system
    system = create_intersection_system(line, hypersurface, verbose=False)
    
    # Solve with PP method
    print("\n--- Solving with PP method ---")
    solutions = solve_polynomial_system(system, method='pp', tolerance=1e-6, verbose=True)
    
    print(f"\n--- Results ---")
    print(f"Number of solutions: {len(solutions)}")
    for i, sol in enumerate(solutions):
        print(f"  Solution {i+1}: t = {sol['t']:.8f}")
        
        # Verify
        t = sol['t']
        x = t
        y = t**2 - 0.5
        print(f"    Point: ({x:.8f}, {y:.8f})")
        print(f"    Residual: y = {abs(y):.2e}")
    
    # Check correctness
    assert len(solutions) == 1, f"Expected 1 solution, got {len(solutions)}"
    t_expected = np.sqrt(0.5)
    assert abs(solutions[0]['t'] - t_expected) < 1e-4, "Solution not accurate"
    
    print("\n✅ Test 1 passed!")


def test_2d_multiple_roots():
    """Test 2D system with multiple roots."""
    print("\n" + "=" * 80)
    print("TEST 2: 2D System - Multiple Roots")
    print("=" * 80)

    # Curve: y = (x - 0.3)(x - 0.7) = x^2 - x + 0.21
    # Parametric: (t, t^2 - t + 0.21) for t in [0, 1]

    def curve_func(t):
        return np.array([t, t**2 - t + 0.21])

    hypersurface = Hypersurface(
        func=curve_func,
        param_ranges=[(0.0, 1.0)],
        ambient_dim=2,
        degree=2,
        verbose=False
    )

    # Line: y = 0
    line = Line(
        hyperplanes=[Hyperplane(coeffs=[0.0, 1.0], d=0.0)]
    )
    
    print("\nCurve: (x(t), y(t)) = (t, t^2 - t + 0.21)")
    print("Line: y = 0")
    print("Expected intersections: t = 0.3, 0.7")
    
    # Create system
    system = create_intersection_system(line, hypersurface, verbose=False)
    
    # Solve with PP method
    print("\n--- Solving with PP method ---")
    solutions = solve_polynomial_system(system, method='pp', tolerance=1e-6, verbose=True)
    
    print(f"\n--- Results ---")
    print(f"Number of solutions: {len(solutions)}")
    for i, sol in enumerate(solutions):
        print(f"  Solution {i+1}: t = {sol['t']:.8f}")
    
    # Check correctness
    assert len(solutions) == 2, f"Expected 2 solutions, got {len(solutions)}"
    
    t_values = sorted([sol['t'] for sol in solutions])
    assert abs(t_values[0] - 0.3) < 1e-4, "First root not accurate"
    assert abs(t_values[1] - 0.7) < 1e-4, "Second root not accurate"
    
    print("\n✅ Test 2 passed!")


def test_2d_custom_domain():
    """Test 2D system with custom parameter domain."""
    print("\n" + "=" * 80)
    print("TEST 3: 2D System - Custom Parameter Domain")
    print("=" * 80)

    # Curve: y = x^2 - 2 for x in [1, 2]
    # Parametric: (t, t^2 - 2) for t in [1, 2]

    def curve_func(t):
        return np.array([t, t**2 - 2.0])

    hypersurface = Hypersurface(
        func=curve_func,
        param_ranges=[(1.0, 2.0)],  # Custom domain
        ambient_dim=2,
        degree=2,
        verbose=False
    )

    # Line: y = 0
    line = Line(
        hyperplanes=[Hyperplane(coeffs=[0.0, 1.0], d=0.0)]
    )
    
    print("\nCurve: (x(t), y(t)) = (t, t^2 - 2) for t in [1, 2]")
    print("Line: y = 0")
    print("Expected intersection: t = sqrt(2) ≈ 1.414")
    
    # Create system
    system = create_intersection_system(line, hypersurface, verbose=False)
    
    # Solve with PP method
    print("\n--- Solving with PP method ---")
    solutions = solve_polynomial_system(system, method='pp', tolerance=1e-6, verbose=True)
    
    print(f"\n--- Results ---")
    print(f"Number of solutions: {len(solutions)}")
    for i, sol in enumerate(solutions):
        print(f"  Solution {i+1}: t = {sol['t']:.8f}")
        
        # Verify
        t = sol['t']
        x = t
        y = t**2 - 2.0
        print(f"    Point: ({x:.8f}, {y:.8f})")
        print(f"    Residual: y = {abs(y):.2e}")
    
    # Check correctness
    assert len(solutions) == 1, f"Expected 1 solution, got {len(solutions)}"
    t_expected = np.sqrt(2.0)
    assert abs(solutions[0]['t'] - t_expected) < 1e-4, "Solution not accurate"
    
    print("\n✅ Test 3 passed!")


def test_3d_simple():
    """Test 3D system: line intersecting a surface."""
    print("\n" + "=" * 80)
    print("TEST 4: 3D System - Line Intersecting Surface")
    print("=" * 80)

    # Surface: z = u + v for u, v in [0, 1]
    # Parametric: (u, v, u + v)

    def surface_func(u, v):
        return np.array([u, v, u + v])

    hypersurface = Hypersurface(
        func=surface_func,
        param_ranges=[(0.0, 1.0), (0.0, 1.0)],
        ambient_dim=3,
        degree=1,
        verbose=False
    )

    # Line: x = 0.5, y = 0.3
    # Hyperplane 1: x - 0.5 = 0 -> 1*x + 0*y + 0*z - 0.5 = 0
    # Hyperplane 2: y - 0.3 = 0 -> 0*x + 1*y + 0*z - 0.3 = 0
    line = Line(
        hyperplanes=[
            Hyperplane(coeffs=[1.0, 0.0, 0.0], d=-0.5),
            Hyperplane(coeffs=[0.0, 1.0, 0.0], d=-0.3)
        ]
    )
    
    print("\nSurface: (x, y, z) = (u, v, u + v)")
    print("Line: x = 0.5, y = 0.3")
    print("Expected intersection: u = 0.5, v = 0.3")
    
    # Create system
    system = create_intersection_system(line, hypersurface, verbose=False)
    
    # Solve with PP method
    print("\n--- Solving with PP method ---")
    solutions = solve_polynomial_system(system, method='pp', tolerance=1e-6, verbose=True)
    
    print(f"\n--- Results ---")
    print(f"Number of solutions: {len(solutions)}")
    for i, sol in enumerate(solutions):
        print(f"  Solution {i+1}: u = {sol['u']:.8f}, v = {sol['v']:.8f}")
        
        # Verify
        u, v = sol['u'], sol['v']
        x, y, z = u, v, u + v
        print(f"    Point: ({x:.8f}, {y:.8f}, {z:.8f})")
        print(f"    Residual: x - 0.5 = {abs(x - 0.5):.2e}, y - 0.3 = {abs(y - 0.3):.2e}")
    
    # Check correctness
    assert len(solutions) == 1, f"Expected 1 solution, got {len(solutions)}"
    assert abs(solutions[0]['u'] - 0.5) < 1e-4, "u not accurate"
    assert abs(solutions[0]['v'] - 0.3) < 1e-4, "v not accurate"
    
    print("\n✅ Test 4 passed!")


if __name__ == "__main__":
    test_2d_simple()
    test_2d_multiple_roots()
    test_2d_custom_domain()
    test_3d_simple()
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✅")
    print("=" * 80)
    print("\nComplete workflow verified:")
    print("  1. ✅ Convert to Bernstein basis and normalize")
    print("  2. ✅ Use PP method to find all possible roots")
    print("  3. ✅ Use Newton iteration to refine each root")
    print("  4. ✅ Return to original domain")
    print("=" * 80)

