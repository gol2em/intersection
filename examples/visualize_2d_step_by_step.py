"""
Step-by-step 3D visualization of 2D polynomial system solving.
Similar to the 1D debug visualization but with 3D plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from intersection.bernstein import polynomial_nd_to_bernstein
from intersection.convex_hull import find_root_box_pp_nd
from intersection.de_casteljau import extract_subbox_2d
from intersection.box import Box


class Visualizer2D:
    """Visualize 2D polynomial system solving step by step."""
    
    def __init__(self, eq1_power, eq2_power, tolerance=1e-3, subdivision_tolerance=1e-10):
        self.eq1_power = eq1_power
        self.eq2_power = eq2_power
        self.tolerance = tolerance
        self.subdivision_tolerance = subdivision_tolerance
        
        # Convert to Bernstein
        self.eq1_bern_initial = polynomial_nd_to_bernstein(eq1_power, k=2)
        self.eq2_bern_initial = polynomial_nd_to_bernstein(eq2_power, k=2)
        
        # Statistics
        self.step_count = 0
        self.boxes_processed = 0
        self.boxes_pruned = 0
        self.boxes_subdivided = 0
        self.solutions = []
        
    def evaluate_polynomial(self, coeffs_power, x, y):
        """Evaluate polynomial in power form at (x, y)."""
        result = 0.0
        for i in range(coeffs_power.shape[0]):
            for j in range(coeffs_power.shape[1]):
                result += coeffs_power[i, j] * (x ** i) * (y ** j)
        return result

    def draw_projections_on_walls(self, ax, ctrl_points_3d, color, x_min, x_max, y_min, y_max):
        """
        Draw projections of control points and their convex hulls on background walls.

        For a 2D system with parameters (x, y) and function value f:
        - Project onto (x, f) plane → draw on back wall (y = y_max)
        - Project onto (y, f) plane → draw on side wall (x = x_min)
        """
        from intersection.convex_hull import convex_hull_2d, intersect_convex_hull_with_x_axis

        # Projection 1: (x, f) plane - draw on back wall at y = y_max
        proj_x = ctrl_points_3d[:, [0, 2]]  # (x, f) coordinates

        # Draw projected control points on back wall
        for pt in proj_x:
            ax.scatter([pt[0]], [y_max], [pt[1]], color=color, s=30, alpha=0.5, marker='o')

        # Compute and draw convex hull on back wall
        if len(proj_x) >= 3:
            try:
                hull_x = convex_hull_2d(proj_x)
                # Draw convex hull edges on back wall
                for i in range(len(hull_x)):
                    p1 = hull_x[i]
                    p2 = hull_x[(i + 1) % len(hull_x)]
                    ax.plot([p1[0], p2[0]], [y_max, y_max], [p1[1], p2[1]],
                           color=color, linewidth=2, alpha=0.7, linestyle='--')

                # Find and mark intersection with f=0 axis
                intersection = intersect_convex_hull_with_x_axis(proj_x)
                if intersection is not None:
                    x_int_min, x_int_max = intersection
                    # Draw the intersection range on back wall at f=0
                    ax.plot([x_int_min, x_int_max], [y_max, y_max], [0, 0],
                           color=color, linewidth=4, alpha=0.9)
            except:
                pass

        # Projection 2: (y, f) plane - draw on side wall at x = x_min
        proj_y = ctrl_points_3d[:, [1, 2]]  # (y, f) coordinates

        # Draw projected control points on side wall
        for pt in proj_y:
            ax.scatter([x_min], [pt[0]], [pt[1]], color=color, s=30, alpha=0.5, marker='o')

        # Compute and draw convex hull on side wall
        if len(proj_y) >= 3:
            try:
                hull_y = convex_hull_2d(proj_y)
                # Draw convex hull edges on side wall
                for i in range(len(hull_y)):
                    p1 = hull_y[i]
                    p2 = hull_y[(i + 1) % len(hull_y)]
                    ax.plot([x_min, x_min], [p1[0], p2[0]], [p1[1], p2[1]],
                           color=color, linewidth=2, alpha=0.7, linestyle='--')

                # Find and mark intersection with f=0 axis
                intersection = intersect_convex_hull_with_x_axis(proj_y)
                if intersection is not None:
                    y_int_min, y_int_max = intersection
                    # Draw the intersection range on side wall at f=0
                    ax.plot([x_min, x_min], [y_int_min, y_int_max], [0, 0],
                           color=color, linewidth=4, alpha=0.9)
            except:
                pass
    
    def visualize_box(self, depth, box_range, eq1_bern, eq2_bern, status, pp_result):
        """Create 3D visualization of current box state."""
        self.step_count += 1

        # Compute PP bounds for each equation individually
        pp_result_eq1 = find_root_box_pp_nd([eq1_bern], k=2, tolerance=self.subdivision_tolerance)
        pp_result_eq2 = find_root_box_pp_nd([eq2_bern], k=2, tolerance=self.subdivision_tolerance)

        fig = plt.figure(figsize=(16, 6))

        # Panel 1: Equation 1 surface with its own PP bounds
        ax1 = fig.add_subplot(131, projection='3d')
        self.plot_surface(ax1, box_range, eq1_bern, 'Equation 1: x² + y² - 1 = 0', 'blue', pp_result_eq1)

        # Panel 2: Equation 2 surface with its own PP bounds
        ax2 = fig.add_subplot(132, projection='3d')
        self.plot_surface(ax2, box_range, eq2_bern, 'Equation 2: x²/4 + 4y² - 1 = 0', 'red', pp_result_eq2)

        # Panel 3: Both surfaces together (no projections, combined PP bounds)
        ax3 = fig.add_subplot(133, projection='3d')
        self.plot_both_surfaces(ax3, box_range, eq1_bern, eq2_bern, pp_result)

        # Title
        status_color = {
            'SOLUTION': 'green',
            'PRUNED': 'red',
            'SUBDIVIDE': 'blue',
            'TIGHTEN': 'purple',
            'MAX_DEPTH': 'orange'
        }.get(status, 'black')

        fig.suptitle(f'Step {self.step_count}: Depth {depth} - {status}',
                    fontsize=16, fontweight='bold', color=status_color)

        plt.tight_layout()
        filename = f'debug_2d_step_{self.step_count:04d}_depth{depth}_{status.lower()}.png'
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()
    
    def plot_surface(self, ax, box_range, bern_coeffs, title, color, pp_result=None):
        """Plot polynomial surface over the box."""
        x_min, x_max = box_range[0]
        y_min, y_max = box_range[1]

        # Create grid
        x = np.linspace(x_min, x_max, 30)
        y = np.linspace(y_min, y_max, 30)
        X, Y = np.meshgrid(x, y)

        # Evaluate polynomial at grid points
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Evaluate using power form (more accurate)
                if color == 'blue':  # eq1
                    Z[i,j] = self.evaluate_polynomial(self.eq1_power, X[i,j], Y[i,j])
                else:  # eq2
                    Z[i,j] = self.evaluate_polynomial(self.eq2_power, X[i,j], Y[i,j])

        # Plot surface
        surf = ax.plot_surface(X, Y, Z, alpha=0.6, cmap=cm.coolwarm if color=='blue' else cm.Reds)

        # Plot zero level (z=0 plane)
        ax.plot_surface(X, Y, np.zeros_like(Z), alpha=0.2, color='green')

        # Plot the curve where surface intersects z=0 (the zero contour)
        contour = ax.contour(X, Y, Z, levels=[0], colors=color, linewidths=3, zdir='z', offset=0)

        # Plot Bernstein control points
        n, m = bern_coeffs.shape
        ctrl_points_3d = []
        for i in range(n):
            for j in range(m):
                # Map control point to box coordinates
                x_ctrl = x_min + (i / (n-1)) * (x_max - x_min) if n > 1 else (x_min + x_max) / 2
                y_ctrl = y_min + (j / (m-1)) * (y_max - y_min) if m > 1 else (y_min + y_max) / 2
                z_ctrl = bern_coeffs[i, j]
                ax.scatter([x_ctrl], [y_ctrl], [z_ctrl], color='black', s=20, alpha=0.8)
                ctrl_points_3d.append([x_ctrl, y_ctrl, z_ctrl])

        ctrl_points_3d = np.array(ctrl_points_3d)

        # Project control points and draw convex hulls on background walls
        self.draw_projections_on_walls(ax, ctrl_points_3d, color, x_min, x_max, y_min, y_max)

        # Draw current box boundary on z=0 plane
        box_corners_x = [x_min, x_max, x_max, x_min, x_min]
        box_corners_y = [y_min, y_min, y_max, y_max, y_min]
        ax.plot(box_corners_x, box_corners_y, [0]*5, 'k-', linewidth=2, alpha=0.5, label='Current box')

        # Draw PP bounding box on z=0 plane if available (use different color than curve)
        if pp_result is not None:
            (pp_x_min, pp_x_max), (pp_y_min, pp_y_max) = pp_result

            # Map to box coordinates
            pp_x_min_box = x_min + pp_x_min * (x_max - x_min)
            pp_x_max_box = x_min + pp_x_max * (x_max - x_min)
            pp_y_min_box = y_min + pp_y_min * (y_max - y_min)
            pp_y_max_box = y_min + pp_y_max * (y_max - y_min)

            # Use orange for blue curve, purple for red curve
            pp_box_color = 'orange' if color == 'blue' else 'purple'

            # Draw PP bounding box on z=0 plane
            pp_corners_x = [pp_x_min_box, pp_x_max_box, pp_x_max_box, pp_x_min_box, pp_x_min_box]
            pp_corners_y = [pp_y_min_box, pp_y_min_box, pp_y_max_box, pp_y_max_box, pp_y_min_box]
            ax.plot(pp_corners_x, pp_corners_y, [0]*5, color=pp_box_color, linewidth=3,
                   alpha=0.9, label='PP bounds')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x,y)')
        ax.set_title(title, fontsize=10)
        ax.legend(loc='upper right', fontsize=8)

        # Set consistent z-limits
        ax.set_zlim(-2, 4)
    
    def plot_both_surfaces(self, ax, box_range, eq1_bern, eq2_bern, pp_result):
        """Plot both surfaces and PP bounds (without projections on walls)."""
        x_min, x_max = box_range[0]
        y_min, y_max = box_range[1]

        # Create grid
        x = np.linspace(x_min, x_max, 30)
        y = np.linspace(y_min, y_max, 30)
        X, Y = np.meshgrid(x, y)

        # Evaluate both polynomials
        Z1 = np.zeros_like(X)
        Z2 = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z1[i,j] = self.evaluate_polynomial(self.eq1_power, X[i,j], Y[i,j])
                Z2[i,j] = self.evaluate_polynomial(self.eq2_power, X[i,j], Y[i,j])

        # Plot both surfaces
        ax.plot_surface(X, Y, Z1, alpha=0.3, color='blue', label='Eq1')
        ax.plot_surface(X, Y, Z2, alpha=0.3, color='red', label='Eq2')

        # Plot zero plane
        ax.plot_surface(X, Y, np.zeros_like(Z1), alpha=0.1, color='green')

        # Plot the curves where surfaces intersect z=0 (the zero contours)
        ax.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=3, zdir='z', offset=0)
        ax.contour(X, Y, Z2, levels=[0], colors='red', linewidths=3, zdir='z', offset=0)

        # Draw current box boundary on z=0 plane
        box_corners_x = [x_min, x_max, x_max, x_min, x_min]
        box_corners_y = [y_min, y_min, y_max, y_max, y_min]
        ax.plot(box_corners_x, box_corners_y, [0]*5, 'k-', linewidth=2, alpha=0.5, label='Current box')

        # Plot PP bounds if available (only on z=0 plane)
        if pp_result is not None:
            (pp_x_min, pp_x_max), (pp_y_min, pp_y_max) = pp_result

            # Map to box coordinates
            pp_x_min_box = x_min + pp_x_min * (x_max - x_min)
            pp_x_max_box = x_min + pp_x_max * (x_max - x_min)
            pp_y_min_box = y_min + pp_y_min * (y_max - y_min)
            pp_y_max_box = y_min + pp_y_max * (y_max - y_min)

            # Draw PP bounding box on z=0 plane only
            pp_corners_x = [pp_x_min_box, pp_x_max_box, pp_x_max_box, pp_x_min_box, pp_x_min_box]
            pp_corners_y = [pp_y_min_box, pp_y_min_box, pp_y_max_box, pp_y_max_box, pp_y_min_box]
            ax.plot(pp_corners_x, pp_corners_y, [0]*5, 'g-', linewidth=3, alpha=0.9, label='PP bounds')

        # Plot expected solution
        x_exact = 2 / np.sqrt(5)
        y_exact = 1 / np.sqrt(5)
        if x_min <= x_exact <= x_max and y_min <= y_exact <= y_max:
            ax.scatter([x_exact], [y_exact], [0], color='gold', s=200, marker='*',
                      edgecolors='black', linewidths=2, label='Expected solution', zorder=10)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x,y)')
        ax.set_title('Both Equations + Bounds', fontsize=10)
        ax.set_zlim(-2, 4)
        ax.legend(loc='upper right', fontsize=8)
    
    def solve_recursive(self, eq1_bern, eq2_bern, box_range, depth=0, max_depth=20, crit=0.5):
        """Recursively solve with visualization."""
        self.boxes_processed += 1
        
        print(f"\n{'  ' * depth}[Depth {depth}] Processing box {box_range}")
        
        # Step 1: Apply PP method
        pp_result = find_root_box_pp_nd([eq1_bern, eq2_bern], k=2, 
                                        tolerance=self.subdivision_tolerance)
        
        if pp_result is None:
            # No roots - PRUNE
            print(f"{'  ' * depth}  → PRUNED (PP method: no roots)")
            self.boxes_pruned += 1
            self.visualize_box(depth, box_range, eq1_bern, eq2_bern, 'PRUNED', None)
            return
        
        print(f"{'  ' * depth}  PP bounds: {pp_result}")
        
        # Calculate widths
        pp_widths = [t_max - t_min for t_min, t_max in pp_result]
        box_widths = [x_max - x_min for (x_min, x_max) in box_range]
        
        print(f"{'  ' * depth}  PP widths: {pp_widths}")
        print(f"{'  ' * depth}  Box widths: {box_widths}")
        
        # Map PP bounds to box coordinates
        pp_box_widths = [pp_widths[i] * box_widths[i] for i in range(2)]
        
        print(f"{'  ' * depth}  PP box widths: {pp_box_widths}")
        
        # Step 2: Check if small enough
        if all(w < self.tolerance for w in pp_box_widths):
            # Found solution
            x_sol = box_range[0][0] + (pp_result[0][0] + pp_result[0][1]) / 2 * box_widths[0]
            y_sol = box_range[1][0] + (pp_result[1][0] + pp_result[1][1]) / 2 * box_widths[1]
            print(f"{'  ' * depth}  → SOLUTION at ({x_sol:.6f}, {y_sol:.6f})")
            self.solutions.append((x_sol, y_sol))
            self.visualize_box(depth, box_range, eq1_bern, eq2_bern, 'SOLUTION', pp_result)
            return
        
        # Step 3: Check depth limit
        if depth >= max_depth:
            x_sol = box_range[0][0] + (pp_result[0][0] + pp_result[0][1]) / 2 * box_widths[0]
            y_sol = box_range[1][0] + (pp_result[1][0] + pp_result[1][1]) / 2 * box_widths[1]
            print(f"{'  ' * depth}  → MAX DEPTH at ({x_sol:.6f}, {y_sol:.6f})")
            self.solutions.append((x_sol, y_sol))
            self.visualize_box(depth, box_range, eq1_bern, eq2_bern, 'MAX_DEPTH', pp_result)
            return
        
        # Step 4: Check CRIT - should we subdivide or tighten?
        dims_to_subdivide = []
        for i, pp_width in enumerate(pp_widths):
            if pp_width > crit:
                dims_to_subdivide.append(i)
        
        if not dims_to_subdivide:
            # TIGHTEN all dimensions
            print(f"{'  ' * depth}  → TIGHTEN (all dims reduced ≥ {(1-crit)*100:.0f}%)")
            self.visualize_box(depth, box_range, eq1_bern, eq2_bern, 'TIGHTEN', pp_result)
            
            # Extract sub-box
            eq1_tight = extract_subbox_2d(eq1_bern, pp_result, tolerance=self.subdivision_tolerance)
            eq2_tight = extract_subbox_2d(eq2_bern, pp_result, tolerance=self.subdivision_tolerance)
            
            # Map to box coordinates
            tight_range = [
                (box_range[0][0] + pp_result[0][0] * box_widths[0],
                 box_range[0][0] + pp_result[0][1] * box_widths[0]),
                (box_range[1][0] + pp_result[1][0] * box_widths[1],
                 box_range[1][0] + pp_result[1][1] * box_widths[1])
            ]
            
            self.solve_recursive(eq1_tight, eq2_tight, tight_range, depth, max_depth, crit)
        else:
            # SUBDIVIDE along dimensions that didn't shrink enough
            print(f"{'  ' * depth}  → SUBDIVIDE dims {dims_to_subdivide}")
            self.boxes_subdivided += 1
            self.visualize_box(depth, box_range, eq1_bern, eq2_bern, 'SUBDIVIDE', pp_result)
            
            # For simplicity, subdivide along first dimension that needs it
            axis = dims_to_subdivide[0]
            t_mid = (pp_result[axis][0] + pp_result[axis][1]) / 2
            
            print(f"{'  ' * depth}  Subdividing axis {axis} at t={t_mid:.6f}")
            
            # Create sub-ranges
            left_ranges = list(pp_result)
            left_ranges[axis] = (pp_result[axis][0], t_mid)
            
            right_ranges = list(pp_result)
            right_ranges[axis] = (t_mid, pp_result[axis][1])
            
            # Extract coefficients
            eq1_left = extract_subbox_2d(eq1_bern, left_ranges, tolerance=self.subdivision_tolerance)
            eq2_left = extract_subbox_2d(eq2_bern, left_ranges, tolerance=self.subdivision_tolerance)
            
            eq1_right = extract_subbox_2d(eq1_bern, right_ranges, tolerance=self.subdivision_tolerance)
            eq2_right = extract_subbox_2d(eq2_bern, right_ranges, tolerance=self.subdivision_tolerance)
            
            # Map to box coordinates
            left_box_range = [
                (box_range[0][0] + left_ranges[0][0] * box_widths[0],
                 box_range[0][0] + left_ranges[0][1] * box_widths[0]),
                (box_range[1][0] + left_ranges[1][0] * box_widths[1],
                 box_range[1][0] + left_ranges[1][1] * box_widths[1])
            ]
            
            right_box_range = [
                (box_range[0][0] + right_ranges[0][0] * box_widths[0],
                 box_range[0][0] + right_ranges[0][1] * box_widths[0]),
                (box_range[1][0] + right_ranges[1][0] * box_widths[1],
                 box_range[1][0] + right_ranges[1][1] * box_widths[1])
            ]
            
            # Recursively process
            self.solve_recursive(eq1_left, eq2_left, left_box_range, depth + 1, max_depth, crit)
            self.solve_recursive(eq1_right, eq2_right, right_box_range, depth + 1, max_depth, crit)


if __name__ == "__main__":
    print("=" * 80)
    print("2D STEP-BY-STEP VISUALIZATION")
    print("=" * 80)
    
    # Define polynomials
    eq1_power = np.zeros((3, 3))
    eq1_power[0, 0] = -1
    eq1_power[2, 0] = 1
    eq1_power[0, 2] = 1
    
    eq2_power = np.zeros((3, 3))
    eq2_power[0, 0] = -1
    eq2_power[2, 0] = 0.25
    eq2_power[0, 2] = 4
    
    print("\nExpected solution: x ≈ 0.894427, y ≈ 0.447214")
    
    visualizer = Visualizer2D(eq1_power, eq2_power, tolerance=1e-3, subdivision_tolerance=1e-10)
    
    initial_range = [(0.0, 1.0), (0.0, 1.0)]
    
    visualizer.solve_recursive(
        visualizer.eq1_bern_initial,
        visualizer.eq2_bern_initial,
        initial_range,
        depth=0,
        max_depth=10,
        crit=0.5
    )
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Boxes processed: {visualizer.boxes_processed}")
    print(f"Boxes pruned: {visualizer.boxes_pruned}")
    print(f"Boxes subdivided: {visualizer.boxes_subdivided}")
    print(f"Solutions found: {len(visualizer.solutions)}")
    
    for i, (x, y) in enumerate(visualizer.solutions):
        print(f"\n  Solution {i+1}: x={x:.6f}, y={y:.6f}")
        
        # Verify
        eq1_val = x**2 + y**2 - 1
        eq2_val = x**2/4 + 4*y**2 - 1
        print(f"    eq1 residual: {eq1_val:.6e}")
        print(f"    eq2 residual: {eq2_val:.6e}")

