"""
Visualize how PP method tightens bounds before bisection.

This shows:
1. Original box [a, b]
2. PP method finds tighter bounds [t_min, t_max] via convex hull
3. If [t_min, t_max] is still large, bisect it
4. Repeat

The key insight: PP method gives tighter bounds FIRST, then we bisect those tighter bounds.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from src.intersection.bernstein import polynomial_nd_to_bernstein, evaluate_bernstein_kd
from src.intersection.convex_hull import convex_hull_2d, intersect_convex_hull_with_x_axis
from src.intersection.de_casteljau import de_casteljau_subdivide_1d, extract_subbox_1d

def visualize_pp_tightening_step(coeffs, box_range, step_num):
    """Visualize one step of PP tightening."""
    degree = len(coeffs) - 1
    
    # Get control points
    control_points = np.array([[i / degree, coeffs[i]] for i in range(degree + 1)])
    
    # Compute convex hull intersection
    result = intersect_convex_hull_with_x_axis(control_points)
    
    if result is None:
        print(f"Step {step_num}: Box [{box_range[0]:.6f}, {box_range[1]:.6f}] → PRUNED")
        return None, None
    
    t_min, t_max = result
    
    # Map to original box range
    x_min_tight = box_range[0] + t_min * (box_range[1] - box_range[0])
    x_max_tight = box_range[0] + t_max * (box_range[1] - box_range[0])
    
    box_width = box_range[1] - box_range[0]
    tight_width = x_max_tight - x_min_tight
    reduction = (1 - tight_width / box_width) * 100
    
    print(f"Step {step_num}:")
    print(f"  Original box: [{box_range[0]:.6f}, {box_range[1]:.6f}] (width = {box_width:.6f})")
    print(f"  PP bounds:    [{x_min_tight:.6f}, {x_max_tight:.6f}] (width = {tight_width:.6f})")
    print(f"  Reduction: {reduction:.1f}%")
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Plot polynomial in original box range
    x_vals = np.linspace(box_range[0], box_range[1], 500)
    # Map x_vals to [0, 1] for evaluation
    t_vals = (x_vals - box_range[0]) / (box_range[1] - box_range[0])
    y_vals = np.array([evaluate_bernstein_kd(coeffs, t) for t in t_vals])
    
    ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='Polynomial', zorder=3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, zorder=2)
    
    # Plot control points (mapped to original box range)
    cp_x = box_range[0] + control_points[:, 0] * (box_range[1] - box_range[0])
    cp_y = control_points[:, 1]
    ax.plot(cp_x, cp_y, 'go', markersize=6, label='Control points', zorder=4)
    ax.plot(cp_x, cp_y, 'g--', linewidth=1, alpha=0.3, zorder=3)
    
    # Plot convex hull
    hull_points = convex_hull_2d(control_points)
    hull_x = box_range[0] + hull_points[:, 0] * (box_range[1] - box_range[0])
    hull_y = hull_points[:, 1]
    hull_x_closed = np.append(hull_x, hull_x[0])
    hull_y_closed = np.append(hull_y, hull_y[0])
    ax.fill(hull_x_closed, hull_y_closed, 'yellow', alpha=0.3, label='Convex hull', zorder=2)
    ax.plot(hull_x_closed, hull_y_closed, 'r-', linewidth=2, zorder=5)
    
    # Show original box
    y_min, y_max = ax.get_ylim()
    box_height = y_max - y_min
    rect_orig = FancyBboxPatch(
        (box_range[0], y_min), box_width, box_height,
        boxstyle="round,pad=0.01", 
        edgecolor='blue', facecolor='blue', alpha=0.1,
        linewidth=3, linestyle='--',
        label=f'Original box [{box_range[0]:.3f}, {box_range[1]:.3f}]',
        zorder=1
    )
    ax.add_patch(rect_orig)
    
    # Show PP tightened bounds
    rect_tight = FancyBboxPatch(
        (x_min_tight, y_min), tight_width, box_height,
        boxstyle="round,pad=0.01",
        edgecolor='green', facecolor='green', alpha=0.2,
        linewidth=3,
        label=f'PP bounds [{x_min_tight:.3f}, {x_max_tight:.3f}] ({reduction:.1f}% reduction)',
        zorder=1
    )
    ax.add_patch(rect_tight)
    
    # Show intersection on x-axis
    ax.plot([x_min_tight, x_max_tight], [0, 0], 'g-', linewidth=6, zorder=6)
    ax.plot([x_min_tight, x_min_tight], [y_min, 0], 'g--', linewidth=2, alpha=0.5, zorder=5)
    ax.plot([x_max_tight, x_max_tight], [y_min, 0], 'g--', linewidth=2, alpha=0.5, zorder=5)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('p(x)', fontsize=14)
    ax.set_title(f'Step {step_num}: PP Method Tightens Bounds by {reduction:.1f}%', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.set_xlim(box_range[0] - 0.05 * box_width, box_range[1] + 0.05 * box_width)
    
    plt.tight_layout()
    plt.savefig(f'pp_tightening_step{step_num}.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: pp_tightening_step{step_num}.png\n")
    plt.close()
    
    return (x_min_tight, x_max_tight), (t_min, t_max)

def main():
    print("\n" + "=" * 80)
    print("PP METHOD: TIGHTENING BEFORE BISECTION")
    print("=" * 80)
    
    # Simple polynomial: (x - 0.3)(x - 0.7)
    power_coeffs = np.array([0.21, -1.0, 1.0])
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    print("\nPolynomial: (x - 0.3)(x - 0.7)")
    print(f"Bernstein coefficients: {bern_coeffs}")
    print(f"Expected roots: x = 0.3, 0.7\n")
    
    # Step 1: Initial box [0, 1]
    print("=" * 80)
    coeffs = bern_coeffs
    box_range = (0.0, 1.0)
    tight_range, tight_t = visualize_pp_tightening_step(coeffs, box_range, step_num=1)
    
    if tight_range is None:
        print("Box pruned!")
        return
    
    # Step 2: Extract sub-box with PP bounds and visualize
    print("=" * 80)
    print("Now we extract the tighter sub-box and continue...")
    print("=" * 80)
    
    # Extract coefficients for the tighter box
    sub_coeffs = extract_subbox_1d(coeffs, tight_t[0], tight_t[1])
    
    # Visualize the tighter box
    tight_range2, tight_t2 = visualize_pp_tightening_step(sub_coeffs, tight_range, step_num=2)
    
    if tight_range2 is None:
        print("Box pruned!")
        return
    
    # Step 3: If still too large, bisect the PP bounds
    print("=" * 80)
    print("PP bounds are still large, so we BISECT the PP bounds (not the original box)...")
    print("=" * 80)
    
    mid = (tight_range[0] + tight_range[1]) / 2
    print(f"\nBisecting PP bounds [{tight_range[0]:.6f}, {tight_range[1]:.6f}] at x = {mid:.6f}")
    print(f"  Left half:  [{tight_range[0]:.6f}, {mid:.6f}]")
    print(f"  Right half: [{mid:.6f}, {tight_range[1]:.6f}]\n")
    
    # Subdivide the coefficients
    left_coeffs, right_coeffs = de_casteljau_subdivide_1d(sub_coeffs, t=0.5)
    
    # Visualize left half
    print("=" * 80)
    print("LEFT HALF:")
    print("=" * 80)
    left_range = (tight_range[0], mid)
    visualize_pp_tightening_step(left_coeffs, left_range, step_num=3)
    
    # Visualize right half
    print("=" * 80)
    print("RIGHT HALF:")
    print("=" * 80)
    right_range = (mid, tight_range[1])
    visualize_pp_tightening_step(right_coeffs, right_range, step_num=4)
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nThe PP method workflow:")
    print("1. Start with box [0, 1]")
    print("2. PP method finds tighter bounds via convex hull → reduces box")
    print("3. Extract sub-box with tighter bounds")
    print("4. Apply PP method again → further reduction")
    print("5. If still too large, BISECT the PP bounds (not original box)")
    print("6. Apply PP method to each half → even tighter bounds")
    print("\nKey insight: PP gives tighter bounds FIRST, then we bisect those tighter bounds.")
    print("\nGenerated files:")
    print("  - pp_tightening_step1.png (Initial box → PP bounds)")
    print("  - pp_tightening_step2.png (PP bounds → tighter PP bounds)")
    print("  - pp_tightening_step3.png (Left half after bisection)")
    print("  - pp_tightening_step4.png (Right half after bisection)")
    print("=" * 80)

if __name__ == "__main__":
    main()

