"""
Test: Solve polynomial with 20 roots using PP method (Improved version)

Polynomial: (x-0.05)(x-0.10)(x-0.15)...(x-1.00) = 0
Expected roots: x = 0.05, 0.10, 0.15, ..., 1.00
Domain: [0, 1] to reduce numerical issues

WITH DETAILED VISUALIZATION AND DEBUGGING
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.intersection.polynomial_solver import (
    create_polynomial_system,
    solve_polynomial_system
)
from src.intersection.bernstein import polynomial_nd_to_bernstein, evaluate_bernstein_kd
from src.intersection.convex_hull import convex_hull_2d, intersect_convex_hull_with_x_axis
from debug_subdivision_visualizer import SubdivisionVisualizer
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


def visualize_step_1_power_to_bernstein(power_coeffs, bern_coeffs, roots):
    """Visualize Step 1: Power series to Bernstein conversion."""
    print("\n" + "=" * 80)
    print("STEP 1: POWER SERIES → BERNSTEIN CONVERSION")
    print("=" * 80)

    degree = len(power_coeffs) - 1

    print(f"\nPolynomial degree: {degree}")
    print(f"Number of roots: {len(roots)}")
    print(f"\nPower coefficients range: [{np.min(power_coeffs):.6e}, {np.max(power_coeffs):.6e}]")
    print(f"Bernstein coefficients range: [{np.min(bern_coeffs):.6e}, {np.max(bern_coeffs):.6e}]")

    # Plot both forms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x_vals = np.linspace(0, 1, 1000)

    # Power basis
    y_power = np.polyval(power_coeffs[::-1], x_vals)
    ax1.plot(x_vals, y_power, 'b-', linewidth=2, label='p(x) in power basis')
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    for root in roots:
        ax1.plot(root, 0, 'ro', markersize=6)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('p(x)', fontsize=12)
    ax1.set_title('Power Basis Form', fontsize=14, fontweight='bold')
    ax1.legend()

    # Bernstein basis with control points
    y_bern = np.array([evaluate_bernstein_kd(bern_coeffs, x) for x in x_vals])
    control_points = np.array([[i / degree, bern_coeffs[i]] for i in range(degree + 1)])

    ax2.plot(x_vals, y_bern, 'b-', linewidth=2, label='p(x) in Bernstein basis')
    ax2.plot(control_points[:, 0], control_points[:, 1], 'go', markersize=4, label='Control points', alpha=0.5)
    ax2.plot(control_points[:, 0], control_points[:, 1], 'g--', linewidth=1, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    for root in roots:
        ax2.plot(root, 0, 'ro', markersize=6)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('p(x)', fontsize=12)
    ax2.set_title('Bernstein Basis Form with Control Points', fontsize=14, fontweight='bold')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('debug_step1_conversion.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: debug_step1_conversion.png")
    plt.close()


def visualize_step_2_initial_convex_hull(bern_coeffs, roots):
    """Visualize Step 2: Initial convex hull and bounds."""
    print("\n" + "=" * 80)
    print("STEP 2: INITIAL CONVEX HULL ANALYSIS")
    print("=" * 80)

    degree = len(bern_coeffs) - 1
    control_points = np.array([[i / degree, bern_coeffs[i]] for i in range(degree + 1)])

    # Compute convex hull
    hull_points = convex_hull_2d(control_points)

    print(f"\nControl points: {len(control_points)}")
    print(f"Convex hull vertices: {len(hull_points)}")

    # Find intersection
    result = intersect_convex_hull_with_x_axis(control_points)

    if result is None:
        print("\n✗ No intersection with x-axis!")
        return None

    t_min, t_max = result
    print(f"\nInitial bounds from convex hull: [{t_min:.6f}, {t_max:.6f}]")
    print(f"Width: {t_max - t_min:.6f}")
    print(f"Reduction from [0, 1]: {(1.0 - (t_max - t_min)) * 100:.1f}%")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    x_vals = np.linspace(0, 1, 1000)
    y_vals = np.array([evaluate_bernstein_kd(bern_coeffs, x) for x in x_vals])

    ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='p(x)', zorder=3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)

    # Control points
    ax.plot(control_points[:, 0], control_points[:, 1], 'go', markersize=3, label='Control points', alpha=0.3, zorder=4)

    # Convex hull
    hull_points_closed = np.vstack([hull_points, hull_points[0]])
    ax.fill(hull_points_closed[:, 0], hull_points_closed[:, 1], 'yellow', alpha=0.3, label='Convex hull')
    ax.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], 'r-', linewidth=2, zorder=5)
    ax.plot(hull_points[:, 0], hull_points[:, 1], 'ro', markersize=6, zorder=6)

    # Intersection
    ax.axvspan(t_min, t_max, alpha=0.2, color='green', label=f'Root bounds: [{t_min:.3f}, {t_max:.3f}]', zorder=2)
    ax.plot([t_min, t_max], [0, 0], 'g-', linewidth=4, zorder=7)

    # Actual roots
    for root in roots:
        ax.plot(root, 0, 'r*', markersize=10, zorder=8)

    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('p(x)', fontsize=12)
    ax.set_title(f'Initial Convex Hull (degree {degree}, {len(roots)} roots)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('debug_step2_initial_hull.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: debug_step2_initial_hull.png")
    plt.close()

    return result


def test_20_roots_improved():
    """Test polynomial with 20 roots in [0, 1] domain."""
    print("\n" + "=" * 80)
    print("TEST: Polynomial with 20 Roots (Improved - Domain [0, 1])")
    print("=" * 80)

    # Define roots in [0, 1] domain
    # Test with 20 roots as requested
    roots = [i * 0.05 for i in range(1, 21)]  # [0.05, 0.10, ..., 1.00] - 20 roots
    print(f"\nExpected roots:")
    for i in range(0, len(roots), 5):
        print(f"  {roots[i:i+5]}")
    
    # Expand polynomial
    print("\nExpanding polynomial (x-0.05)(x-0.10)...(x-1.00)...")
    power_coeffs = expand_polynomial_product(roots)
    
    print(f"Polynomial degree: {len(power_coeffs) - 1}")
    print(f"Leading coefficient: {power_coeffs[-1]:.6e}")
    print(f"Constant term: {power_coeffs[0]:.6e}")
    
    # Convert to Bernstein basis (already in [0, 1] domain)
    print("\nConverting to Bernstein basis...")
    start_time = time.time()
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    conversion_time = time.time() - start_time

    print(f"Conversion time: {conversion_time:.3f} seconds")
    print(f"Bernstein coefficients shape: {bern_coeffs.shape}")
    print(f"Bernstein coefficient range: [{np.min(bern_coeffs):.6e}, {np.max(bern_coeffs):.6e}]")

    # VISUALIZATION: Step 1 - Power to Bernstein conversion (disabled for speed)
    # visualize_step_1_power_to_bernstein(power_coeffs, bern_coeffs, roots)

    # VISUALIZATION: Step 2 - Initial convex hull (disabled for speed)
    # initial_bounds = visualize_step_2_initial_convex_hull(bern_coeffs, roots)
    
    # Create system
    print("\nCreating polynomial system...")
    system = create_polynomial_system(
        equation_coeffs=[bern_coeffs],
        param_ranges=[(0.0, 1.0)],
        param_names=['x']
    )
    
    # Solve with PP method using standard solver
    print("\n" + "=" * 80)
    print("SOLVING WITH PP METHOD")
    print("=" * 80)

    start_time = time.time()

    # Use standard solver
    solutions = solve_polynomial_system(
        system,
        method='pp',
        tolerance=1e-6,
        crit=0.8,
        max_depth=50,
        refine=False,
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
        # Find closest expected root
        closest_root = min(roots, key=lambda r: abs(x - r))
        error = x - closest_root
        print(f"  {i+1:2d}. x = {x:.6f}  (expected: {closest_root:.2f}, error: {error:+.2e}, residual: {residual:.2e})")
    
    # Check which expected roots were found
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    found_roots = []
    missing_roots = []
    tolerance = 0.02  # Consider a root found if within 0.02 of expected value
    
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
        print(f"\nFound roots:")
        for i in range(0, len(found_roots), 5):
            print(f"  {found_roots[i:i+5]}")
    
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
    
    # Accuracy analysis
    print("\n" + "=" * 80)
    print("ACCURACY ANALYSIS")
    print("=" * 80)
    
    errors = []
    for sol in solutions:
        x = sol['x']
        closest_root = min(roots, key=lambda r: abs(x - r))
        error = abs(x - closest_root)
        errors.append(error)
    
    if errors:
        print(f"Mean error: {np.mean(errors):.6e}")
        print(f"Max error: {np.max(errors):.6e}")
        print(f"Min error: {np.min(errors):.6e}")
        print(f"Std dev: {np.std(errors):.6e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Expected roots: {len(roots)}")
    print(f"Found roots: {len(found_roots)}")
    print(f"Missing roots: {len(missing_roots)}")
    print(f"Spurious roots: {len(spurious)}")
    print(f"Success rate: {len(found_roots) / len(roots) * 100:.1f}%")
    if errors:
        print(f"Mean accuracy: {np.mean(errors):.6e}")
    print(f"Total time: {conversion_time + solve_time:.3f} seconds")
    print("=" * 80)
    
    # Final verdict
    if len(found_roots) == len(roots) and len(spurious) == 0:
        print("\n🎉 SUCCESS! PP method found all 20 roots with no spurious roots!")
    elif len(found_roots) == len(roots):
        print(f"\n⚠️  PARTIAL SUCCESS: Found all roots but with {len(spurious)} spurious roots")
    else:
        print(f"\n❌ INCOMPLETE: Found {len(found_roots)}/{len(roots)} roots")


if __name__ == "__main__":
    test_20_roots_improved()

