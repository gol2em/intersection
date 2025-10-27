"""
Example demonstrating convex hull intersection for PP method.

This shows how the convex hull of Bernstein control points can give
tighter bounds than simple min/max for finding roots.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from src.intersection.bernstein import polynomial_nd_to_bernstein
from src.intersection.convex_hull import (
    convex_hull_2d,
    intersect_convex_hull_with_x_axis,
    find_root_box_pp_1d
)


def visualize_convex_hull_method(power_coeffs, title="Convex Hull Method for PP"):
    """
    Visualize how convex hull method finds tighter bounds.
    
    Args:
        power_coeffs: Polynomial coefficients in power basis
        title: Plot title
    """
    # Convert to Bernstein basis
    bernstein_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    n = len(bernstein_coeffs)
    
    # Create control points
    t_values = np.linspace(0, 1, n)
    control_points = np.column_stack([t_values, bernstein_coeffs])
    
    # Compute convex hull
    hull = convex_hull_2d(control_points)
    
    # Find intersection with x-axis
    intersection = intersect_convex_hull_with_x_axis(control_points)
    
    # Evaluate actual polynomial
    t_plot = np.linspace(0, 1, 200)
    p_values = np.polyval(power_coeffs[::-1], t_plot)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot actual polynomial
    ax.plot(t_plot, p_values, 'b-', linewidth=2, label='Actual polynomial', zorder=3)
    
    # Plot control points
    ax.plot(control_points[:, 0], control_points[:, 1], 'ro', 
            markersize=10, label='Bernstein control points', zorder=4)
    
    # Plot convex hull
    hull_closed = np.vstack([hull, hull[0]])
    ax.plot(hull_closed[:, 0], hull_closed[:, 1], 'g--', 
            linewidth=2, label='Convex hull', zorder=2)
    ax.fill(hull_closed[:, 0], hull_closed[:, 1], 
            color='green', alpha=0.1, zorder=1)
    
    # Plot x-axis
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.3)
    
    # Plot simple bounds (min/max)
    min_coeff = np.min(bernstein_coeffs)
    max_coeff = np.max(bernstein_coeffs)
    if min_coeff <= 0 <= max_coeff:
        ax.axvspan(0, 1, alpha=0.1, color='red', 
                   label=f'Simple bounds: [0, 1]', zorder=0)
    
    # Plot convex hull intersection
    if intersection:
        t_min, t_max = intersection
        ax.axvspan(t_min, t_max, alpha=0.2, color='blue', 
                   label=f'Convex hull bounds: [{t_min:.3f}, {t_max:.3f}]', zorder=0)
        
        # Mark intersection points
        ax.plot([t_min, t_max], [0, 0], 'mo', markersize=12, 
                label='Hull-axis intersections', zorder=5)
        
        # Add vertical lines
        ax.axvline(x=t_min, color='m', linestyle=':', linewidth=2, alpha=0.5)
        ax.axvline(x=t_max, color='m', linestyle=':', linewidth=2, alpha=0.5)
    
    # Find actual roots
    roots = np.roots(power_coeffs[::-1])
    real_roots = roots[np.isreal(roots)].real
    real_roots = real_roots[(real_roots >= 0) & (real_roots <= 1)]
    
    if len(real_roots) > 0:
        ax.plot(real_roots, np.zeros_like(real_roots), 'k*', 
                markersize=20, label='Actual roots', zorder=6)
    
    ax.set_xlabel('t', fontsize=14)
    ax.set_ylabel('f(t)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    # Add text box with statistics
    stats_text = f"Polynomial degree: {len(power_coeffs) - 1}\n"
    stats_text += f"Bernstein coeffs: {n}\n"
    stats_text += f"Min coeff: {min_coeff:.4f}\n"
    stats_text += f"Max coeff: {max_coeff:.4f}\n"
    
    if intersection:
        width = t_max - t_min
        improvement = (1.0 - width) / 1.0 * 100
        stats_text += f"\nSimple bounds: [0.000, 1.000]\n"
        stats_text += f"CH bounds: [{t_min:.3f}, {t_max:.3f}]\n"
        stats_text += f"Improvement: {improvement:.1f}% tighter"
    else:
        stats_text += f"\nNo roots (pruned by CH)"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig


def main():
    """Run examples."""
    print("=" * 80)
    print("Convex Hull Method for PP - Visual Examples")
    print("=" * 80)
    
    # Example 1: Quadratic with two roots
    print("\nExample 1: Quadratic with two roots")
    print("f(t) = (t - 0.3)(t - 0.7) = t^2 - t + 0.21")
    
    power_coeffs = np.array([0.21, -1.0, 1.0])
    fig1 = visualize_convex_hull_method(
        power_coeffs, 
        "Example 1: Quadratic f(t) = (t - 0.3)(t - 0.7)"
    )
    
    # Example 2: Cubic with one root
    print("\nExample 2: Cubic with one root")
    print("f(t) = (t - 0.5)^3")
    
    # (t - 0.5)^3 = t^3 - 1.5t^2 + 0.75t - 0.125
    power_coeffs = np.array([-0.125, 0.75, -1.5, 1.0])
    fig2 = visualize_convex_hull_method(
        power_coeffs,
        "Example 2: Cubic f(t) = (t - 0.5)³"
    )
    
    # Example 3: Polynomial with no roots
    print("\nExample 3: Polynomial with no roots")
    print("f(t) = t^2 + 0.5")
    
    power_coeffs = np.array([0.5, 0.0, 1.0])
    fig3 = visualize_convex_hull_method(
        power_coeffs,
        "Example 3: Quadratic f(t) = t² + 0.5 (no roots)"
    )
    
    # Example 4: High degree polynomial
    print("\nExample 4: High degree polynomial")
    print("f(t) = (t - 0.4)(t - 0.6)(t - 0.8)")
    
    # Expand (t - 0.4)(t - 0.6)(t - 0.8)
    p1 = np.poly1d([1, -0.4])
    p2 = np.poly1d([1, -0.6])
    p3 = np.poly1d([1, -0.8])
    p = p1 * p2 * p3
    power_coeffs = p.coefficients
    
    fig4 = visualize_convex_hull_method(
        power_coeffs,
        "Example 4: Cubic f(t) = (t - 0.4)(t - 0.6)(t - 0.8)"
    )
    
    plt.show()
    
    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)
    
    # Numerical comparison
    print("\n" + "=" * 80)
    print("Numerical Comparison: Simple vs Convex Hull Method")
    print("=" * 80)
    
    test_cases = [
        ("f(t) = t - 0.5", np.array([-0.5, 1.0])),
        ("f(t) = (t - 0.3)(t - 0.7)", np.array([0.21, -1.0, 1.0])),
        ("f(t) = t^2 + 0.5", np.array([0.5, 0.0, 1.0])),
        ("f(t) = (t - 0.5)^3", np.array([-0.125, 0.75, -1.5, 1.0])),
    ]
    
    for name, power_coeffs in test_cases:
        print(f"\n{name}")
        print("-" * 60)
        
        # Convert to Bernstein
        bernstein_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
        
        # Simple method
        min_coeff = np.min(bernstein_coeffs)
        max_coeff = np.max(bernstein_coeffs)
        simple_can_prune = (min_coeff > 0) or (max_coeff < 0)
        simple_box = None if simple_can_prune else (0.0, 1.0)
        
        # Convex hull method
        ch_box = find_root_box_pp_1d(bernstein_coeffs)
        
        print(f"  Bernstein coeffs: {bernstein_coeffs}")
        print(f"  Simple method: {simple_box}")
        print(f"  Convex hull:   {ch_box}")
        
        if simple_box and ch_box:
            width = ch_box[1] - ch_box[0]
            improvement = (1.0 - width) / 1.0 * 100
            print(f"  Improvement: {improvement:.1f}% tighter bounds")
        elif simple_box and not ch_box:
            print(f"  Improvement: Pruned by convex hull (simple method failed)")
        elif not simple_box and not ch_box:
            print(f"  Both methods correctly pruned")


if __name__ == "__main__":
    main()

