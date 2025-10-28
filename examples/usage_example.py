"""
Usage Examples for Polynomial System Solver

This file demonstrates how to use the polynomial system solver with different methods
for both 1D and multi-dimensional problems.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from intersection.polynomial_solver import (
    create_polynomial_system,
    solve_polynomial_system
)
from intersection.bernstein import polynomial_nd_to_bernstein


def example_1d_simple():
    """
    Example 1: Simple 1D polynomial with PP method
    Solve: x - 0.5 = 0 on domain [0, 1]
    """
    print("=" * 80)
    print("Example 1: Simple 1D Polynomial (PP Method)")
    print("=" * 80)
    print("Equation: x - 0.5 = 0")
    print("Domain: [0, 1]")
    print()
    
    # Define polynomial coefficients in power basis: -0.5 + x
    poly = np.array([-0.5, 1.0])
    
    # Convert to Bernstein basis
    bernstein_coeffs = polynomial_nd_to_bernstein(poly, k=1)
    
    # Create polynomial system
    system = create_polynomial_system(
        equation_coeffs=[bernstein_coeffs],
        param_ranges=[(0.0, 1.0)]
    )
    
    # Solve using PP method
    solutions = solve_polynomial_system(
        system,
        method='pp',
        tolerance=1e-6,
        verbose=True
    )
    
    print(f"\nFound {len(solutions)} solution(s):")
    for i, sol in enumerate(solutions, 1):
        print(f"  Solution {i}: t = {sol['t']:.10f}")
    
    print()


def example_1d_multiple_roots():
    """
    Example 2: 1D polynomial with multiple roots using LP method
    Solve: (x-0.2)(x-0.5)(x-0.8) = 0 on domain [0, 1]
    """
    print("=" * 80)
    print("Example 2: 1D Polynomial with Multiple Roots (LP Method)")
    print("=" * 80)
    print("Equation: (x-0.2)(x-0.5)(x-0.8) = 0")
    print("Domain: [0, 1]")
    print()
    
    # Define polynomial by expanding (x-0.2)(x-0.5)(x-0.8)
    roots = [0.2, 0.5, 0.8]
    poly = np.array([1.0])
    for root in roots:
        factor = np.array([-root, 1.0])
        poly = np.convolve(poly, factor)
    
    # Convert to Bernstein basis
    bernstein_coeffs = polynomial_nd_to_bernstein(poly, k=1)
    
    # Create polynomial system
    system = create_polynomial_system(
        equation_coeffs=[bernstein_coeffs],
        param_ranges=[(0.0, 1.0)]
    )
    
    # Solve using LP method
    solutions = solve_polynomial_system(
        system,
        method='lp',
        tolerance=1e-6,
        verbose=True
    )
    
    print(f"\nFound {len(solutions)} solution(s):")
    for i, sol in enumerate(solutions, 1):
        print(f"  Solution {i}: t = {sol['t']:.10f}")
    
    print(f"\nExpected roots: {roots}")
    print()


def example_2d_circle_ellipse():
    """
    Example 3: 2D system - Circle and Ellipse intersection
    Solve: x^2 + y^2 - 1 = 0 and x^2/4 + 4*y^2 - 1 = 0 on [0,1]×[0,1]
    """
    print("=" * 80)
    print("Example 3: 2D Circle-Ellipse Intersection (PP Method)")
    print("=" * 80)
    print("Equation 1: x^2 + y^2 - 1 = 0")
    print("Equation 2: x^2/4 + 4*y^2 - 1 = 0")
    print("Domain: [0, 1] × [0, 1]")
    print()
    
    # Circle: x^2 + y^2 - 1 = 0
    # Power basis representation as 2D array
    circle_power = np.array([
        [-1.0, 0.0, 1.0],   # Constant, y, y^2 terms for x^0
        [0.0, 0.0, 0.0],    # Constant, y, y^2 terms for x^1
        [1.0, 0.0, 0.0]     # Constant, y, y^2 terms for x^2
    ])
    
    # Ellipse: x^2/4 + 4*y^2 - 1 = 0
    ellipse_power = np.array([
        [-1.0, 0.0, 4.0],   # Constant, y, y^2 terms for x^0
        [0.0, 0.0, 0.0],    # Constant, y, y^2 terms for x^1
        [0.25, 0.0, 0.0]    # Constant, y, y^2 terms for x^2
    ])
    
    # Convert to Bernstein basis
    circle_bernstein = polynomial_nd_to_bernstein(circle_power, k=2)
    ellipse_bernstein = polynomial_nd_to_bernstein(ellipse_power, k=2)
    
    # Create polynomial system
    system = create_polynomial_system(
        equation_coeffs=[circle_bernstein, ellipse_bernstein],
        param_ranges=[(0.0, 1.0), (0.0, 1.0)]
    )
    
    # Solve using PP method
    solutions = solve_polynomial_system(
        system,
        method='pp',
        tolerance=1e-6,
        verbose=True
    )
    
    print(f"\nFound {len(solutions)} solution(s):")
    for i, sol in enumerate(solutions, 1):
        x, y = sol['t'], sol['u']
        print(f"  Solution {i}: (x, y) = ({x:.10f}, {y:.10f})")
        # Verify
        circle_val = x**2 + y**2 - 1
        ellipse_val = x**2/4 + 4*y**2 - 1
        print(f"    Circle residual: {abs(circle_val):.2e}")
        print(f"    Ellipse residual: {abs(ellipse_val):.2e}")
    
    print()


def example_2d_lp_method():
    """
    Example 4: Same 2D system using LP method for comparison
    """
    print("=" * 80)
    print("Example 4: 2D Circle-Ellipse Intersection (LP Method)")
    print("=" * 80)
    print("Equation 1: x^2 + y^2 - 1 = 0")
    print("Equation 2: x^2/4 + 4*y^2 - 1 = 0")
    print("Domain: [0, 1] × [0, 1]")
    print()
    
    # Same setup as Example 3
    circle_power = np.array([
        [-1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    
    ellipse_power = np.array([
        [-1.0, 0.0, 4.0],
        [0.0, 0.0, 0.0],
        [0.25, 0.0, 0.0]
    ])
    
    circle_bernstein = polynomial_nd_to_bernstein(circle_power, k=2)
    ellipse_bernstein = polynomial_nd_to_bernstein(ellipse_power, k=2)
    
    system = create_polynomial_system(
        equation_coeffs=[circle_bernstein, ellipse_bernstein],
        param_ranges=[(0.0, 1.0), (0.0, 1.0)]
    )
    
    # Solve using LP method
    solutions = solve_polynomial_system(
        system,
        method='lp',
        tolerance=1e-6,
        verbose=True
    )
    
    print(f"\nFound {len(solutions)} solution(s):")
    for i, sol in enumerate(solutions, 1):
        x, y = sol['t'], sol['u']
        print(f"  Solution {i}: (x, y) = ({x:.10f}, {y:.10f})")
    
    print()


def example_method_comparison():
    """
    Example 5: Compare PP and LP methods on the same problem
    """
    print("=" * 80)
    print("Example 5: Method Comparison (PP vs LP)")
    print("=" * 80)
    print("Problem: 2D Circle-Ellipse Intersection")
    print()
    
    # Setup system
    circle_power = np.array([[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ellipse_power = np.array([[-1.0, 0.0, 4.0], [0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    
    circle_bernstein = polynomial_nd_to_bernstein(circle_power, k=2)
    ellipse_bernstein = polynomial_nd_to_bernstein(ellipse_power, k=2)
    
    system = create_polynomial_system(
        equation_coeffs=[circle_bernstein, ellipse_bernstein],
        param_ranges=[(0.0, 1.0), (0.0, 1.0)]
    )
    
    # Test both methods
    for method in ['pp', 'lp']:
        print(f"\n{method.upper()} Method:")
        print("-" * 40)
        
        solutions, stats = solve_polynomial_system(
            system,
            method=method,
            tolerance=1e-6,
            verbose=False,
            return_stats=True
        )
        
        print(f"  Solutions found: {len(solutions)}")
        print(f"  Boxes processed: {stats['boxes_processed']}")
        print(f"  Max depth: {stats['max_depth_used']}")
        print(f"  Subdivisions: {stats['subdivisions']}")
    
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "POLYNOMIAL SYSTEM SOLVER USAGE EXAMPLES" + " " * 19 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Run examples
    example_1d_simple()
    example_1d_multiple_roots()
    example_2d_circle_ellipse()
    example_2d_lp_method()
    example_method_comparison()
    
    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

