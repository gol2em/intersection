"""
Visualize PP Method Step-by-Step

Shows:
1. Convert to power series
2. Convert to Bernstein form
3. Normalized system with graph
4. For each subdivision step: control points and bounding box
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.intersection.bernstein import polynomial_nd_to_bernstein, evaluate_bernstein_kd
from src.intersection.convex_hull import convex_hull_2d, intersect_convex_hull_with_x_axis
from src.intersection.subdivision_solver import Box
import time


def expand_polynomial_product(roots):
    """Expand (x - r1)(x - r2)...(x - rn) into power basis coefficients."""
    poly = np.array([1.0])
    for root in roots:
        factor = np.array([-root, 1.0])
        poly = np.convolve(poly, factor)
    return poly


def visualize_step_1_power_series(roots):
    """Step 1: Show the polynomial in power series form."""
    print("\n" + "=" * 80)
    print("STEP 1: CONVERT TO POWER SERIES")
    print("=" * 80)
    
    # Expand polynomial
    power_coeffs = expand_polynomial_product(roots)
    
    print(f"\nPolynomial: (x - {roots[0]})(x - {roots[1]})...(x - {roots[-1]})")
    print(f"Degree: {len(power_coeffs) - 1}")
    print(f"\nPower series coefficients [a0, a1, a2, ...]:")
    print(f"  (where p(x) = a0 + a1*x + a2*x^2 + ... + an*x^n)")
    
    for i, coeff in enumerate(power_coeffs):
        if abs(coeff) > 1e-10:
            print(f"  a{i} = {coeff:+.6e}")
    
    # Plot the polynomial
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x_vals = np.linspace(0, 1, 1000)
    y_vals = np.polyval(power_coeffs[::-1], x_vals)
    
    ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='p(x) in power basis')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Mark roots
    for root in roots:
        ax.plot(root, 0, 'ro', markersize=8)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('p(x)', fontsize=12)
    ax.set_title('Step 1: Polynomial in Power Series Form', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('step1_power_series.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: step1_power_series.png")
    plt.close()
    
    return power_coeffs


def visualize_step_2_bernstein(power_coeffs, roots):
    """Step 2: Convert to Bernstein form and show control points."""
    print("\n" + "=" * 80)
    print("STEP 2: CONVERT TO BERNSTEIN FORM")
    print("=" * 80)
    
    # Convert to Bernstein
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    degree = len(bern_coeffs) - 1
    
    print(f"\nBernstein coefficients [b0, b1, b2, ...]:")
    print(f"  (where p(x) = Σ bi * B_i^{degree}(x))")
    
    for i, coeff in enumerate(bern_coeffs):
        print(f"  b{i} = {coeff:+.6e}")
    
    # Control points
    control_points = np.array([[i / degree, bern_coeffs[i]] for i in range(degree + 1)])
    
    print(f"\nControl points (t, b_i):")
    for i, (t, b) in enumerate(control_points):
        print(f"  P{i} = ({t:.4f}, {b:+.6e})")
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Evaluate polynomial
    x_vals = np.linspace(0, 1, 1000)
    y_vals = np.array([evaluate_bernstein_kd(bern_coeffs, x) for x in x_vals])
    
    ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='p(x) in Bernstein basis')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    
    # Plot control points
    ax.plot(control_points[:, 0], control_points[:, 1], 'go', markersize=8, label='Control points', zorder=5)
    
    # Plot control polygon
    ax.plot(control_points[:, 0], control_points[:, 1], 'g--', linewidth=1, alpha=0.5, label='Control polygon')
    
    # Mark roots
    for root in roots:
        ax.plot(root, 0, 'ro', markersize=8)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('p(x)', fontsize=12)
    ax.set_title('Step 2: Polynomial in Bernstein Form with Control Points', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('step2_bernstein.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: step2_bernstein.png")
    plt.close()
    
    return bern_coeffs


def visualize_step_3_convex_hull(bern_coeffs, roots):
    """Step 3: Show convex hull and intersection with x-axis."""
    print("\n" + "=" * 80)
    print("STEP 3: CONVEX HULL AND X-AXIS INTERSECTION")
    print("=" * 80)
    
    degree = len(bern_coeffs) - 1
    control_points = np.array([[i / degree, bern_coeffs[i]] for i in range(degree + 1)])
    
    # Compute convex hull
    hull_points = convex_hull_2d(control_points)

    print(f"\nConvex hull vertices (in order):")
    for i, (t, b) in enumerate(hull_points):
        print(f"  {i}: ({t:.4f}, {b:+.6e})")
    
    # Find intersection with x-axis
    result = intersect_convex_hull_with_x_axis(control_points)
    
    if result is None:
        print("\n✗ No intersection with x-axis (polynomial has no roots in [0, 1])")
        t_min, t_max = None, None
    else:
        t_min, t_max = result
        print(f"\nIntersection with x-axis: [{t_min:.6f}, {t_max:.6f}]")
        print(f"  Width: {t_max - t_min:.6f}")
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Evaluate polynomial
    x_vals = np.linspace(0, 1, 1000)
    y_vals = np.array([evaluate_bernstein_kd(bern_coeffs, x) for x in x_vals])
    
    ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='p(x)', zorder=3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    
    # Plot control points
    ax.plot(control_points[:, 0], control_points[:, 1], 'go', markersize=6, label='Control points', zorder=4, alpha=0.5)
    
    # Plot convex hull
    hull_points_closed = np.vstack([hull_points, hull_points[0]])
    ax.fill(hull_points_closed[:, 0], hull_points_closed[:, 1], 'yellow', alpha=0.3, label='Convex hull')
    ax.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], 'r-', linewidth=2, zorder=5)
    
    # Highlight hull vertices
    ax.plot(hull_points[:, 0], hull_points[:, 1], 'ro', markersize=8, zorder=6)
    
    # Show intersection
    if result is not None:
        ax.axvspan(t_min, t_max, alpha=0.2, color='green', label=f'Root bounds: [{t_min:.3f}, {t_max:.3f}]')
        ax.plot([t_min, t_max], [0, 0], 'g-', linewidth=4, zorder=7)
    
    # Mark actual roots
    for root in roots:
        ax.plot(root, 0, 'r*', markersize=12, zorder=8)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('p(x)', fontsize=12)
    ax.set_title('Step 3: Convex Hull and X-Axis Intersection', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.savefig('step3_convex_hull.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: step3_convex_hull.png")
    plt.close()
    
    return result


def visualize_subdivision_tree(bern_coeffs, roots, max_depth=5):
    """Step 4: Visualize subdivision process."""
    print("\n" + "=" * 80)
    print("STEP 4: SUBDIVISION PROCESS")
    print("=" * 80)
    
    degree = len(bern_coeffs) - 1
    
    # We'll manually implement a simple subdivision to visualize
    from src.intersection.de_casteljau import de_casteljau_subdivide_1d
    
    # Track boxes to process
    boxes_to_visualize = []
    
    def process_box(coeffs, box_range, depth):
        """Process a box and its subdivisions."""
        if depth > max_depth:
            return
        
        # Get control points
        control_points = np.array([[i / degree, coeffs[i]] for i in range(degree + 1)])
        
        # Compute convex hull intersection
        result = intersect_convex_hull_with_x_axis(control_points)
        
        # Store box info
        box_info = {
            'depth': depth,
            'range': box_range,
            'coeffs': coeffs.copy(),
            'control_points': control_points.copy(),
            'intersection': result,
            'pruned': result is None
        }
        boxes_to_visualize.append(box_info)
        
        if result is None:
            # Pruned
            return
        
        t_min, t_max = result
        box_width = box_range[1] - box_range[0]
        
        # Check if small enough
        if box_width < 1e-3 or depth >= max_depth:
            return
        
        # Subdivide
        mid = (box_range[0] + box_range[1]) / 2
        left_coeffs, right_coeffs = de_casteljau_subdivide_1d(coeffs, t=0.5)
        
        # Process left and right
        process_box(left_coeffs, (box_range[0], mid), depth + 1)
        process_box(right_coeffs, (mid, box_range[1]), depth + 1)
    
    # Start subdivision
    process_box(bern_coeffs, (0.0, 1.0), 0)
    
    print(f"\nTotal boxes processed: {len(boxes_to_visualize)}")
    print(f"Boxes by depth:")
    for d in range(max_depth + 1):
        count = sum(1 for b in boxes_to_visualize if b['depth'] == d)
        pruned = sum(1 for b in boxes_to_visualize if b['depth'] == d and b['pruned'])
        print(f"  Depth {d}: {count} boxes ({pruned} pruned)")
    
    return boxes_to_visualize


def plot_subdivision_steps(boxes_to_visualize, roots, bern_coeffs):
    """Plot subdivision steps."""
    degree = len(bern_coeffs) - 1
    
    # Group by depth
    max_depth = max(b['depth'] for b in boxes_to_visualize)
    
    for depth in range(min(4, max_depth + 1)):  # Show first 4 depths
        boxes_at_depth = [b for b in boxes_to_visualize if b['depth'] == depth]
        
        if not boxes_at_depth:
            continue
        
        fig, axes = plt.subplots(1, min(4, len(boxes_at_depth)), figsize=(16, 4))
        if len(boxes_at_depth) == 1:
            axes = [axes]
        
        fig.suptitle(f'Depth {depth}: Subdivision Boxes', fontsize=14, fontweight='bold')
        
        for idx, (ax, box) in enumerate(zip(axes, boxes_at_depth[:4])):
            box_range = box['range']
            coeffs = box['coeffs']
            control_points = box['control_points']
            
            # Evaluate polynomial in this range
            x_vals = np.linspace(box_range[0], box_range[1], 200)
            # Map to [0, 1] for evaluation
            t_vals = (x_vals - box_range[0]) / (box_range[1] - box_range[0])
            y_vals = np.array([evaluate_bernstein_kd(coeffs, t) for t in t_vals])
            
            ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='p(x)')
            ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
            
            # Plot control points (mapped to box range)
            cp_x = box_range[0] + control_points[:, 0] * (box_range[1] - box_range[0])
            cp_y = control_points[:, 1]
            ax.plot(cp_x, cp_y, 'go', markersize=6, label='Control points', zorder=4)
            ax.plot(cp_x, cp_y, 'g--', linewidth=1, alpha=0.5)
            
            # Plot convex hull
            hull_points = convex_hull_2d(control_points)
            hull_x = box_range[0] + hull_points[:, 0] * (box_range[1] - box_range[0])
            hull_y = hull_points[:, 1]
            hull_x_closed = np.append(hull_x, hull_x[0])
            hull_y_closed = np.append(hull_y, hull_y[0])
            ax.fill(hull_x_closed, hull_y_closed, 'yellow', alpha=0.3)
            ax.plot(hull_x_closed, hull_y_closed, 'r-', linewidth=2)
            
            # Show intersection
            if box['intersection'] is not None:
                t_min, t_max = box['intersection']
                x_min = box_range[0] + t_min * (box_range[1] - box_range[0])
                x_max = box_range[0] + t_max * (box_range[1] - box_range[0])
                ax.axvspan(x_min, x_max, alpha=0.2, color='green')
                ax.set_title(f'Box [{box_range[0]:.4f}, {box_range[1]:.4f}]\nBounds: [{x_min:.4f}, {x_max:.4f}]', fontsize=10)
            else:
                ax.set_title(f'Box [{box_range[0]:.4f}, {box_range[1]:.4f}]\nPRUNED', fontsize=10, color='red')
            
            # Mark roots in this range
            for root in roots:
                if box_range[0] <= root <= box_range[1]:
                    ax.plot(root, 0, 'r*', markersize=12, zorder=8)
            
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x', fontsize=10)
            ax.set_ylabel('p(x)', fontsize=10)
            ax.legend(fontsize=8)
        
        # Hide unused axes
        for ax in axes[len(boxes_at_depth):]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'step4_subdivision_depth{depth}.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: step4_subdivision_depth{depth}.png")
        plt.close()


def main():
    """Main visualization function."""
    print("\n" + "=" * 80)
    print("PP METHOD VISUALIZATION")
    print("=" * 80)
    
    # Use a simpler polynomial for better visualization
    roots = [0.2, 0.5, 0.8]
    print(f"\nPolynomial roots: {roots}")
    print(f"Number of roots: {len(roots)}")
    
    # Step 1: Power series
    power_coeffs = visualize_step_1_power_series(roots)
    
    # Step 2: Bernstein form
    bern_coeffs = visualize_step_2_bernstein(power_coeffs, roots)
    
    # Step 3: Convex hull
    visualize_step_3_convex_hull(bern_coeffs, roots)
    
    # Step 4: Subdivision
    boxes = visualize_subdivision_tree(bern_coeffs, roots, max_depth=3)
    plot_subdivision_steps(boxes, roots, bern_coeffs)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - step1_power_series.png")
    print("  - step2_bernstein.png")
    print("  - step3_convex_hull.png")
    print("  - step4_subdivision_depth0.png")
    print("  - step4_subdivision_depth1.png")
    print("  - step4_subdivision_depth2.png")
    print("  - step4_subdivision_depth3.png")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

