"""
Example: Parameter Domain Normalization

Demonstrates how to normalize parameter domains to [0,1]^k for LP/PP methods.

This example uses a unit circle parameterized as (cos(u), sin(u)) where u ∈ [-π, π],
and shows how to normalize it to u ∈ [0, 1].
"""

import sys
sys.path.insert(0, 'D:/Python/Intersection')

import numpy as np
from src.intersection.geometry import Hyperplane, Line, Hypersurface
from src.intersection.polynomial_system import create_intersection_system
from src.intersection.normalization import (
    normalize_hypersurface,
    denormalize_solutions,
    verify_normalization
)


def example_circle_normalization():
    """Example: Normalize unit circle from [-π, π] to [0, 1]."""
    
    print("=" * 80)
    print("EXAMPLE: Unit Circle Normalization")
    print("=" * 80)
    
    # Create unit circle with parameter in [-π, π]
    print("\n--- Step 1: Create Original Circle ---")
    print("Parameterization: (cos(u), sin(u))")
    print("Parameter range: u ∈ [-π, π]")
    
    circle_original = Hypersurface(
        func=lambda u: np.array([np.cos(u), np.sin(u)]),
        param_ranges=[(-np.pi, np.pi)],
        ambient_dim=2,
        degree=8,
        verbose=False
    )
    
    print(f"✓ Original circle created")
    print(f"  Parameter range: {circle_original.param_ranges}")
    
    # Normalize the circle
    print("\n--- Step 2: Normalize Parameter Domain ---")
    
    circle_normalized, transform_info = normalize_hypersurface(circle_original, verbose=True)
    
    print(f"✓ Normalized circle created")
    print(f"  Parameter range: {circle_normalized.param_ranges}")
    
    # Verify normalization
    print("\n--- Step 3: Verify Normalization ---")
    
    passed = verify_normalization(
        circle_original,
        circle_normalized,
        transform_info,
        n_test_points=10,
        verbose=False
    )
    
    print(f"✓ Verification: {'PASSED' if passed else 'FAILED'}")
    
    # Test specific points
    print("\n--- Step 4: Test Specific Points ---")
    
    test_points = [
        (0.0, "Start (maps to -π)"),
        (0.25, "Quarter (maps to -π/2)"),
        (0.5, "Middle (maps to 0)"),
        (0.75, "Three-quarters (maps to π/2)"),
        (1.0, "End (maps to π)"),
    ]
    
    for u_norm, description in test_points:
        u_orig = transform_info['forward'](u_norm)[0]
        point_norm = circle_normalized.evaluate(u_norm)
        point_orig = circle_original.evaluate(u_orig)
        
        print(f"\n  u = {u_norm:.2f} ({description})")
        print(f"    Original parameter: {u_orig:+.4f}")
        print(f"    Point: ({point_norm[0]:+.4f}, {point_norm[1]:+.4f})")
        print(f"    Match: {np.allclose(point_norm, point_orig)}")
    
    # Create intersection system with normalized circle
    print("\n--- Step 5: Create Intersection System ---")
    
    # Create a diagonal line y = x
    line = Line([Hyperplane(coeffs=[1, -1], d=0)])
    print(f"Line: y = x")
    
    system_normalized = create_intersection_system(line, circle_normalized, verbose=False)
    
    print(f"✓ Intersection system created")
    print(f"  Equation Bernstein coefficients shape: {system_normalized['equation_bernstein_coeffs'][0].shape}")
    print(f"  Parameter range: [0, 1] (ready for LP/PP methods)")
    
    print("\n--- Step 6: Simulate Finding Solutions ---")
    
    # Simulate solutions in normalized space
    # (In practice, these would come from LP/PP solver)
    solutions_normalized = [
        {'t': 0.125},  # First intersection
        {'t': 0.625},  # Second intersection
    ]
    
    print(f"Found {len(solutions_normalized)} solutions in normalized space:")
    for i, sol in enumerate(solutions_normalized):
        print(f"  Solution {i+1}: t = {sol['t']:.4f}")
    
    # Denormalize solutions
    print("\n--- Step 7: Denormalize Solutions ---")
    
    solutions_original = denormalize_solutions(solutions_normalized, transform_info, verbose=False)
    
    print(f"Solutions in original parameter space:")
    for i, (sol_norm, sol_orig) in enumerate(zip(solutions_normalized, solutions_original)):
        u_norm = sol_norm['t']
        u_orig = sol_orig['t']
        point = circle_original.evaluate(u_orig)
        
        print(f"\n  Solution {i+1}:")
        print(f"    Normalized: t = {u_norm:.4f}")
        print(f"    Original:   t = {u_orig:+.4f}")
        print(f"    Point: ({point[0]:+.6f}, {point[1]:+.6f})")
        
        # Verify it's on the line y = x
        residual = point[0] - point[1]
        print(f"    On line y=x: {abs(residual) < 1e-4} (residual = {residual:.2e})")
    
    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. ✓ Normalization preserves geometry")
    print("2. ✓ Normalized domain is [0,1]^k (required for LP/PP)")
    print("3. ✓ Solutions can be denormalized back to original space")
    print("4. ✓ Ready to use with LP/PP solvers")


def example_surface_normalization():
    """Example: Normalize a 3D surface with 2 parameters."""
    
    print("\n\n" + "=" * 80)
    print("EXAMPLE: 3D Surface Normalization")
    print("=" * 80)
    
    # Create a paraboloid with non-standard parameter ranges
    print("\n--- Step 1: Create Original Surface ---")
    print("Parameterization: (u, v, u² + v²)")
    print("Parameter ranges: u ∈ [-2, 2], v ∈ [-1, 1]")
    
    surface_original = Hypersurface(
        func=lambda u, v: np.array([u, v, u**2 + v**2]),
        param_ranges=[(-2, 2), (-1, 1)],
        ambient_dim=3,
        degree=4,
        verbose=False
    )
    
    print(f"✓ Original surface created")
    print(f"  Parameter ranges: {surface_original.param_ranges}")
    
    # Normalize the surface
    print("\n--- Step 2: Normalize Parameter Domain ---")
    
    surface_normalized, transform_info = normalize_hypersurface(surface_original, verbose=True)
    
    print(f"✓ Normalized surface created")
    print(f"  Parameter ranges: {surface_normalized.param_ranges}")
    
    # Verify normalization
    print("\n--- Step 3: Verify Normalization ---")
    
    passed = verify_normalization(
        surface_original,
        surface_normalized,
        transform_info,
        n_test_points=10,
        verbose=False
    )
    
    print(f"✓ Verification: {'PASSED' if passed else 'FAILED'}")
    
    # Test specific points
    print("\n--- Step 4: Test Specific Points ---")
    
    test_points = [
        ((0.0, 0.0), "Corner (maps to (-2, -1))"),
        ((0.5, 0.5), "Center (maps to (0, 0))"),
        ((1.0, 1.0), "Corner (maps to (2, 1))"),
    ]
    
    for (u_norm, v_norm), description in test_points:
        u_orig, v_orig = transform_info['forward'](u_norm, v_norm)
        point_norm = surface_normalized.evaluate(u_norm, v_norm)
        point_orig = surface_original.evaluate(u_orig, v_orig)
        
        print(f"\n  (u,v) = ({u_norm:.2f}, {v_norm:.2f}) ({description})")
        print(f"    Original parameters: ({u_orig:+.2f}, {v_orig:+.2f})")
        print(f"    Point: ({point_norm[0]:+.2f}, {point_norm[1]:+.2f}, {point_norm[2]:+.2f})")
        print(f"    Match: {np.allclose(point_norm, point_orig)}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. ✓ Works for multi-parameter hypersurfaces")
    print("2. ✓ Each parameter normalized independently")
    print("3. ✓ Transformation is affine (linear + offset)")
    print("4. ✓ Ready for LP/PP methods in 2D parameter space")


if __name__ == '__main__':
    # Run examples
    example_circle_normalization()
    example_surface_normalization()
    
    print("\n\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nNext Steps:")
    print("1. Use normalized hypersurfaces with LP/PP solvers")
    print("2. Denormalize solutions back to original parameter space")
    print("3. Verify solutions using original parameterization")

