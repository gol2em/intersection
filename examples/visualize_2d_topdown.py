"""
Top-down 2D view of the circle-ellipse intersection showing curves and bounding boxes.
"""

import numpy as np
import matplotlib.pyplot as plt
from intersection.bernstein import polynomial_nd_to_bernstein
from intersection.convex_hull import find_root_box_pp_nd
from intersection.de_casteljau import extract_subbox_2d
from intersection.box import Box


def evaluate_polynomial(coeffs_power, x, y):
    """Evaluate polynomial in power form at (x, y)."""
    result = 0.0
    for i in range(coeffs_power.shape[0]):
        for j in range(coeffs_power.shape[1]):
            result += coeffs_power[i, j] * (x ** i) * (y ** j)
    return result


def plot_topdown_view(box_range, eq1_power, eq2_power, pp_result, title, filename):
    """Create top-down 2D view showing curves and bounding boxes."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    x_min, x_max = box_range[0]
    y_min, y_max = box_range[1]
    
    # Create fine grid for plotting curves
    x = np.linspace(x_min, x_max, 200)
    y = np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate both polynomials
    Z1 = np.zeros_like(X)
    Z2 = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z1[i,j] = evaluate_polynomial(eq1_power, X[i,j], Y[i,j])
            Z2[i,j] = evaluate_polynomial(eq2_power, X[i,j], Y[i,j])
    
    # Plot zero contours (the curves)
    ax.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=3, label='Circle: x² + y² = 1')
    ax.contour(X, Y, Z2, levels=[0], colors='red', linewidths=3, label='Ellipse: x²/4 + 4y² = 1')
    
    # Draw current box boundary
    box_corners_x = [x_min, x_max, x_max, x_min, x_min]
    box_corners_y = [y_min, y_min, y_max, y_max, y_min]
    ax.plot(box_corners_x, box_corners_y, 'k-', linewidth=2, alpha=0.7, label='Current box')
    
    # Plot PP bounds if available
    if pp_result is not None:
        (pp_x_min, pp_x_max), (pp_y_min, pp_y_max) = pp_result
        
        # Map to box coordinates
        pp_x_min_box = x_min + pp_x_min * (x_max - x_min)
        pp_x_max_box = x_min + pp_x_max * (x_max - x_min)
        pp_y_min_box = y_min + pp_y_min * (y_max - y_min)
        pp_y_max_box = y_min + pp_y_max * (y_max - y_min)
        
        # Draw PP bounding box
        pp_corners_x = [pp_x_min_box, pp_x_max_box, pp_x_max_box, pp_x_min_box, pp_x_min_box]
        pp_corners_y = [pp_y_min_box, pp_y_min_box, pp_y_max_box, pp_y_max_box, pp_y_min_box]
        ax.plot(pp_corners_x, pp_corners_y, 'g-', linewidth=3, alpha=0.9, label='PP bounds')
        
        # Fill PP box with light green
        ax.fill(pp_corners_x, pp_corners_y, color='green', alpha=0.1)
    
    # Plot expected solution
    x_exact = 2 / np.sqrt(5)
    y_exact = 1 / np.sqrt(5)
    if x_min <= x_exact <= x_max and y_min <= y_exact <= y_max:
        ax.scatter([x_exact], [y_exact], color='gold', s=300, marker='*', 
                  edgecolors='black', linewidths=2, label='Expected solution', zorder=10)
    
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def main():
    """Generate top-down views for key steps."""
    
    # Define equation 1: x² + y² - 1 = 0
    eq1_power = np.zeros((3, 3))
    eq1_power[0, 0] = -1
    eq1_power[2, 0] = 1
    eq1_power[0, 2] = 1
    
    # Define equation 2: x²/4 + 4y² - 1 = 0
    eq2_power = np.zeros((3, 3))
    eq2_power[0, 0] = -1
    eq2_power[2, 0] = 0.25
    eq2_power[0, 2] = 4.0
    
    print("=" * 80)
    print("TOP-DOWN 2D VIEWS")
    print("=" * 80)
    
    # Step 1: Initial box
    box_range = [(0.0, 1.0), (0.0, 1.0)]
    eq1_bern = polynomial_nd_to_bernstein(eq1_power, k=2)
    eq2_bern = polynomial_nd_to_bernstein(eq2_power, k=2)
    pp_result = find_root_box_pp_nd([eq1_bern, eq2_bern], k=2)
    
    plot_topdown_view(box_range, eq1_power, eq2_power, pp_result,
                     "Step 1: Initial Box [0,1]²",
                     "topdown_step1_initial.png")
    
    # Step 2: After first subdivision - right half
    box_range = [(0.5, 1.0), (0.1875, 0.625)]
    eq1_bern = extract_subbox_2d(polynomial_nd_to_bernstein(eq1_power, k=2), box_range)
    eq2_bern = extract_subbox_2d(polynomial_nd_to_bernstein(eq2_power, k=2), box_range)
    pp_result = find_root_box_pp_nd([eq1_bern, eq2_bern], k=2)

    plot_topdown_view(box_range, eq1_power, eq2_power, pp_result,
                     "Step 2: Right Half After Subdivision",
                     "topdown_step2_right_half.png")

    # Step 3: After first tightening
    box_range = [(0.7395833333333333, 0.982421875), (0.375, 0.5)]
    eq1_bern = extract_subbox_2d(polynomial_nd_to_bernstein(eq1_power, k=2), box_range)
    eq2_bern = extract_subbox_2d(polynomial_nd_to_bernstein(eq2_power, k=2), box_range)
    pp_result = find_root_box_pp_nd([eq1_bern, eq2_bern], k=2)

    plot_topdown_view(box_range, eq1_power, eq2_power, pp_result,
                     "Step 3: After First Tightening",
                     "topdown_step3_tightened.png")

    # Step 4: Near solution
    box_range = [(0.8842123434004928, 0.902694432246521), (0.4425390005185237, 0.451927495115675)]
    eq1_bern = extract_subbox_2d(polynomial_nd_to_bernstein(eq1_power, k=2), box_range)
    eq2_bern = extract_subbox_2d(polynomial_nd_to_bernstein(eq2_power, k=2), box_range)
    pp_result = find_root_box_pp_nd([eq1_bern, eq2_bern], k=2)
    
    plot_topdown_view(box_range, eq1_power, eq2_power, pp_result,
                     "Step 4: Near Solution (Highly Tightened)",
                     "topdown_step4_near_solution.png")
    
    print("\n" + "=" * 80)
    print("Top-down views generated!")
    print("=" * 80)


if __name__ == "__main__":
    main()

