"""
Demonstration of control point projections on background walls.
Shows how the PP method projects control points onto 2D planes.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from intersection.bernstein import polynomial_nd_to_bernstein
from intersection.convex_hull import convex_hull_2d, intersect_convex_hull_with_x_axis


def evaluate_polynomial(coeffs_power, x, y):
    """Evaluate polynomial in power form at (x, y)."""
    result = 0.0
    for i in range(coeffs_power.shape[0]):
        for j in range(coeffs_power.shape[1]):
            result += coeffs_power[i, j] * (x ** i) * (y ** j)
    return result


def draw_projections_on_walls(ax, ctrl_points_3d, color, x_min, x_max, y_min, y_max):
    """
    Draw projections of control points and their convex hulls on background walls.
    
    For a 2D system with parameters (x, y) and function value f:
    - Project onto (x, f) plane → draw on back wall (y = y_max)
    - Project onto (y, f) plane → draw on side wall (x = x_min)
    """
    # Projection 1: (x, f) plane - draw on back wall at y = y_max
    proj_x = ctrl_points_3d[:, [0, 2]]  # (x, f) coordinates
    
    # Draw projected control points on back wall
    for pt in proj_x:
        ax.scatter([pt[0]], [y_max], [pt[1]], color=color, s=50, alpha=0.6, marker='o', edgecolors='black')
    
    # Compute and draw convex hull on back wall
    if len(proj_x) >= 3:
        try:
            hull_x = convex_hull_2d(proj_x)
            # Draw convex hull edges on back wall
            for i in range(len(hull_x)):
                p1 = hull_x[i]
                p2 = hull_x[(i + 1) % len(hull_x)]
                ax.plot([p1[0], p2[0]], [y_max, y_max], [p1[1], p2[1]], 
                       color=color, linewidth=3, alpha=0.8, linestyle='--')
            
            # Find and mark intersection with f=0 axis
            intersection = intersect_convex_hull_with_x_axis(proj_x)
            if intersection is not None:
                x_int_min, x_int_max = intersection
                # Draw the intersection range on back wall at f=0
                ax.plot([x_int_min, x_int_max], [y_max, y_max], [0, 0],
                       color=color, linewidth=5, alpha=1.0, label=f'X-projection bounds')
        except:
            pass
    
    # Projection 2: (y, f) plane - draw on side wall at x = x_min
    proj_y = ctrl_points_3d[:, [1, 2]]  # (y, f) coordinates
    
    # Draw projected control points on side wall
    for pt in proj_y:
        ax.scatter([x_min], [pt[0]], [pt[1]], color=color, s=50, alpha=0.6, marker='o', edgecolors='black')
    
    # Compute and draw convex hull on side wall
    if len(proj_y) >= 3:
        try:
            hull_y = convex_hull_2d(proj_y)
            # Draw convex hull edges on side wall
            for i in range(len(hull_y)):
                p1 = hull_y[i]
                p2 = hull_y[(i + 1) % len(hull_y)]
                ax.plot([x_min, x_min], [p1[0], p2[0]], [p1[1], p2[1]], 
                       color=color, linewidth=3, alpha=0.8, linestyle='--')
            
            # Find and mark intersection with f=0 axis
            intersection = intersect_convex_hull_with_x_axis(proj_y)
            if intersection is not None:
                y_int_min, y_int_max = intersection
                # Draw the intersection range on side wall at f=0
                ax.plot([x_min, x_min], [y_int_min, y_int_max], [0, 0],
                       color=color, linewidth=5, alpha=1.0, label=f'Y-projection bounds')
        except:
            pass


def demo_projection(equation_name, eq_power, color, filename):
    """Create a demonstration plot for one equation."""
    
    # Convert to Bernstein
    eq_bern = polynomial_nd_to_bernstein(eq_power, k=2)
    
    # Box range
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create grid for surface
    x = np.linspace(x_min, x_max, 40)
    y = np.linspace(y_min, y_max, 40)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate polynomial
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = evaluate_polynomial(eq_power, X[i,j], Y[i,j])
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, alpha=0.4, cmap=cm.coolwarm if color=='blue' else cm.Reds)
    
    # Plot zero plane
    ax.plot_surface(X, Y, np.zeros_like(Z), alpha=0.15, color='green')
    
    # Plot zero contour
    ax.contour(X, Y, Z, levels=[0], colors=color, linewidths=4, zdir='z', offset=0)
    
    # Plot Bernstein control points in 3D
    n, m = eq_bern.shape
    ctrl_points_3d = []
    for i in range(n):
        for j in range(m):
            x_ctrl = x_min + (i / (n-1)) * (x_max - x_min) if n > 1 else (x_min + x_max) / 2
            y_ctrl = y_min + (j / (m-1)) * (y_max - y_min) if m > 1 else (y_min + y_max) / 2
            z_ctrl = eq_bern[i, j]
            ax.scatter([x_ctrl], [y_ctrl], [z_ctrl], color='black', s=80, alpha=1.0, 
                      edgecolors='white', linewidths=2, zorder=100)
            ctrl_points_3d.append([x_ctrl, y_ctrl, z_ctrl])
    
    ctrl_points_3d = np.array(ctrl_points_3d)
    
    # Draw projections on walls
    draw_projections_on_walls(ax, ctrl_points_3d, color, x_min, x_max, y_min, y_max)
    
    # Draw background walls for clarity
    # Back wall (y = y_max)
    xx = np.array([[x_min, x_max], [x_min, x_max]])
    yy = np.array([[y_max, y_max], [y_max, y_max]])
    zz = np.array([[-2, -2], [4, 4]])
    ax.plot_surface(xx, yy, zz, alpha=0.05, color='gray')
    
    # Side wall (x = x_min)
    xx = np.array([[x_min, x_min], [x_min, x_min]])
    yy = np.array([[y_min, y_max], [y_min, y_max]])
    zz = np.array([[-2, -2], [4, 4]])
    ax.plot_surface(xx, yy, zz, alpha=0.05, color='gray')
    
    ax.set_xlabel('x', fontsize=14, fontweight='bold')
    ax.set_ylabel('y', fontsize=14, fontweight='bold')
    ax.set_zlabel('f(x,y)', fontsize=14, fontweight='bold')
    ax.set_title(f'{equation_name}\nControl Points Projected onto Background Walls', 
                fontsize=16, fontweight='bold')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(-2, 4)
    ax.legend(loc='upper left', fontsize=12)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def main():
    """Generate demonstration plots."""
    
    print("=" * 80)
    print("CONTROL POINT PROJECTION DEMONSTRATION")
    print("=" * 80)
    
    # Equation 1: x² + y² - 1 = 0 (circle)
    eq1_power = np.zeros((3, 3))
    eq1_power[0, 0] = -1
    eq1_power[2, 0] = 1
    eq1_power[0, 2] = 1
    
    demo_projection("Circle: x² + y² - 1 = 0", eq1_power, 'blue', 
                   'projection_demo_circle.png')
    
    # Equation 2: x²/4 + 4y² - 1 = 0 (ellipse)
    eq2_power = np.zeros((3, 3))
    eq2_power[0, 0] = -1
    eq2_power[2, 0] = 0.25
    eq2_power[0, 2] = 4.0
    
    demo_projection("Ellipse: x²/4 + 4y² - 1 = 0", eq2_power, 'red', 
                   'projection_demo_ellipse.png')
    
    print("\n" + "=" * 80)
    print("EXPLANATION")
    print("=" * 80)
    print("\nThe PP method projects control points onto 2D planes:")
    print("  • Back wall (y=1): Shows (x, f) projection")
    print("    - Black dots: Projected control points")
    print("    - Dashed lines: Convex hull of projected points")
    print("    - Thick solid line at f=0: PP bounds for x-dimension")
    print("\n  • Side wall (x=0): Shows (y, f) projection")
    print("    - Black dots: Projected control points")
    print("    - Dashed lines: Convex hull of projected points")
    print("    - Thick solid line at f=0: PP bounds for y-dimension")
    print("\nThe intersection of the convex hull with f=0 gives the PP bounds!")
    print("=" * 80)


if __name__ == "__main__":
    main()

