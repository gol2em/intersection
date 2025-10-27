"""
Test different bounding methods (PP, LP, Hybrid) on the same problem.

This demonstrates how the solver is designed to be method-agnostic:
- PP, LP, and Hybrid methods differ only in how they compute bounding boxes
- All other logic (subdivision, tightening, recursion) is identical
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from intersection.subdivision_solver import SubdivisionSolver, SolverConfig, BoundingMethod
from intersection.bernstein import polynomial_nd_to_bernstein


def test_circle_ellipse_with_method(method: BoundingMethod):
    """
    Test circle-ellipse intersection with a specific bounding method.

    System:
        x^2 + y^2 - 1 = 0  (circle)
        x^2/4 + 4y^2 - 1 = 0  (ellipse)

    Expected solution: (x, y) ≈ (0.894427, 0.447214)
    """
    print(f"\n{'='*80}")
    print(f"Testing {method.value.upper()} Method")
    print(f"{'='*80}")

    # Define equations in standard form
    # Circle: x^2 + y^2 - 1 = 0
    # Coefficients for x^i * y^j: coeff[i][j]
    circle_coeffs = np.array([
        [-1.0, 0.0, 1.0],   # -1 + y^2
        [ 0.0, 0.0, 0.0],   # 0
        [ 1.0, 0.0, 0.0]    # x^2
    ])

    # Ellipse: x^2/4 + 4y^2 - 1 = 0
    ellipse_coeffs = np.array([
        [-1.0, 0.0, 4.0],   # -1 + 4y^2
        [ 0.0, 0.0, 0.0],   # 0
        [ 0.25, 0.0, 0.0]   # x^2/4
    ])

    # Convert to Bernstein form
    circle_bern = polynomial_nd_to_bernstein(circle_coeffs, k=2)
    ellipse_bern = polynomial_nd_to_bernstein(ellipse_coeffs, k=2)
    
    # Configure solver with the specified method
    config = SolverConfig(
        method=method,
        tolerance=1e-6,
        crit=0.8,
        max_depth=50,
        verbose=True
    )
    
    # Solve
    solver = SubdivisionSolver(config)
    solutions = solver.solve(
        equation_coeffs=[circle_bern, ellipse_bern],
        k=2
    )
    
    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS ({method.value.upper()} Method)")
    print(f"{'='*80}")
    print(f"Solutions found: {len(solutions)}")
    print(f"Boxes processed: {solver.stats['boxes_processed']}")
    print(f"Boxes pruned: {solver.stats['boxes_pruned']}")
    print(f"Subdivisions: {solver.stats['subdivisions']}")
    
    if solutions:
        for i, sol in enumerate(solutions):
            print(f"\nSolution {i+1}: x={sol[0]:.6f}, y={sol[1]:.6f}")
            
            # Verify solution
            circle_residual = sol[0]**2 + sol[1]**2 - 1
            ellipse_residual = sol[0]**2/4 + 4*sol[1]**2 - 1
            print(f"  Circle residual: {circle_residual:.2e}")
            print(f"  Ellipse residual: {ellipse_residual:.2e}")
    
    return solver.stats


def main():
    """Test all three methods on the same problem."""
    print("="*80)
    print("TESTING DIFFERENT BOUNDING METHODS")
    print("="*80)
    print("\nProblem: Circle-Ellipse Intersection")
    print("  Circle: x^2 + y^2 - 1 = 0")
    print("  Ellipse: x^2/4 + 4y^2 - 1 = 0")
    print("  Expected solution: (0.894427, 0.447214)")
    
    # Test PP method
    pp_stats = test_circle_ellipse_with_method(BoundingMethod.PP)
    
    # Test LP method (currently falls back to PP)
    lp_stats = test_circle_ellipse_with_method(BoundingMethod.LP)
    
    # Test Hybrid method (currently falls back to PP)
    hybrid_stats = test_circle_ellipse_with_method(BoundingMethod.HYBRID)
    
    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    print(f"{'Method':<10} {'Boxes':<10} {'Pruned':<10} {'Subdivisions':<15}")
    print(f"{'-'*80}")
    print(f"{'PP':<10} {pp_stats['boxes_processed']:<10} {pp_stats['boxes_pruned']:<10} {pp_stats['subdivisions']:<15}")
    print(f"{'LP':<10} {lp_stats['boxes_processed']:<10} {lp_stats['boxes_pruned']:<10} {lp_stats['subdivisions']:<15}")
    print(f"{'Hybrid':<10} {hybrid_stats['boxes_processed']:<10} {hybrid_stats['boxes_pruned']:<10} {hybrid_stats['subdivisions']:<15}")
    
    print(f"\n{'='*80}")
    print("NOTE: LP and Hybrid methods currently fall back to PP method.")
    print("Once LP and Hybrid are implemented, they should show different statistics.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

