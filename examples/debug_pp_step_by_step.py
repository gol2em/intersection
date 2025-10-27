"""
Step-by-step visualization of PP method for debugging.

This creates a detailed visualization showing:
1. Each subdivision step
2. Bounding boxes at each level
3. PP method tightening
4. Which boxes are pruned vs. kept

SAFETY: Stops if more than 3 boxes are preserved (indicates bug).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from intersection.bernstein import polynomial_nd_to_bernstein
from intersection.convex_hull import find_root_box_pp_1d
from intersection.de_casteljau import de_casteljau_subdivide_1d


def expand_polynomial_product(roots):
    """Expand (x - r1)(x - r2)...(x - rn) into power series."""
    coeffs = np.array([1.0])
    for root in roots:
        new_coeffs = np.zeros(len(coeffs) + 1)
        new_coeffs[1:] += coeffs
        new_coeffs[:-1] -= root * coeffs
        coeffs = new_coeffs
    return coeffs


class PPDebugVisualizer:
    """Visualize PP method step-by-step."""
    
    def __init__(self, bern_coeffs, roots, tolerance=1e-6, subdivision_tolerance=1e-10):
        self.bern_coeffs = bern_coeffs
        self.roots = sorted(roots)
        self.tolerance = tolerance
        self.subdivision_tolerance = subdivision_tolerance
        self.degree = len(bern_coeffs) - 1
        
        # Track all boxes
        self.all_boxes = []  # List of (depth, box_range, coeffs, status, pp_bounds)
        self.solutions = []
        
        # Statistics
        self.boxes_processed = 0
        self.boxes_pruned = 0
        self.boxes_subdivided = 0
        
    def visualize_box(self, depth, box_range, coeffs, status, pp_bounds=None):
        """Visualize a single box."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left panel: Polynomial and control points
        t_plot = np.linspace(box_range[0], box_range[1], 200)
        
        # Evaluate Bernstein polynomial
        def eval_bernstein(t_local):
            """Evaluate Bernstein polynomial at t in [0,1]."""
            n = len(coeffs) - 1
            result = 0
            for i, b_i in enumerate(coeffs):
                # Binomial coefficient
                binom = 1
                for j in range(i):
                    binom = binom * (n - j) // (j + 1)
                # Bernstein basis
                basis = binom * (t_local ** i) * ((1 - t_local) ** (n - i))
                result += b_i * basis
            return result
        
        # Map to [0,1] for evaluation
        t_normalized = (t_plot - box_range[0]) / (box_range[1] - box_range[0])
        y_plot = np.array([eval_bernstein(t) for t in t_normalized])
        
        ax1.plot(t_plot, y_plot, 'b-', linewidth=2, label='Polynomial')
        ax1.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax1.grid(True, alpha=0.3)
        
        # Plot control points
        t_control = np.linspace(box_range[0], box_range[1], len(coeffs))
        ax1.plot(t_control, coeffs, 'ro-', markersize=8, linewidth=1.5, 
                label='Control points', alpha=0.7)
        
        # Plot roots
        for root in self.roots:
            if box_range[0] <= root <= box_range[1]:
                ax1.axvline(root, color='g', linestyle=':', linewidth=2, alpha=0.7)
        
        # Highlight box range
        y_min, y_max = ax1.get_ylim()
        ax1.axvspan(box_range[0], box_range[1], alpha=0.1, color='yellow')
        
        ax1.set_xlabel('t', fontsize=12)
        ax1.set_ylabel('f(t)', fontsize=12)
        ax1.set_title(f'Depth {depth}: Box [{box_range[0]:.4f}, {box_range[1]:.4f}]', 
                     fontsize=14, fontweight='bold')
        ax1.legend()
        
        # Right panel: Convex hull and PP bounds
        ax2.plot(t_control, coeffs, 'ro-', markersize=10, linewidth=2, 
                label='Control points')
        ax2.axhline(0, color='k', linestyle='--', linewidth=1, label='x-axis')
        ax2.grid(True, alpha=0.3)
        
        # Compute and show convex hull
        from intersection.convex_hull import convex_hull_2d
        points_2d = np.column_stack([t_control, coeffs])
        hull = convex_hull_2d(points_2d)
        
        # Plot convex hull
        hull_closed = np.vstack([hull, hull[0]])
        ax2.fill(hull_closed[:, 0], hull_closed[:, 1], alpha=0.2, color='orange', 
                label='Convex hull')
        ax2.plot(hull_closed[:, 0], hull_closed[:, 1], 'r-', linewidth=2)
        
        # Show PP bounds if available
        if pp_bounds is not None:
            t_min, t_max = pp_bounds
            t_min_orig = box_range[0] + t_min * (box_range[1] - box_range[0])
            t_max_orig = box_range[0] + t_max * (box_range[1] - box_range[0])
            
            ax2.axvspan(t_min_orig, t_max_orig, alpha=0.3, color='green', 
                       label=f'PP bounds [{t_min:.3f}, {t_max:.3f}]')
            
            # Show reduction
            original_width = box_range[1] - box_range[0]
            pp_width = t_max_orig - t_min_orig
            reduction = (1 - pp_width / original_width) * 100
            
            ax2.text(0.5, 0.95, f'PP reduction: {reduction:.1f}%', 
                    transform=ax2.transAxes, fontsize=12, fontweight='bold',
                    ha='center', va='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.8))
        
        ax2.set_xlabel('t', fontsize=12)
        ax2.set_ylabel('Coefficient value', fontsize=12)
        status_color = {
            'SOLUTION': 'green',
            'PRUNED': 'red',
            'SUBDIVIDE': 'blue',
            'TIGHTEN': 'purple',
            'MAX_DEPTH': 'orange'
        }.get(status, 'black')

        ax2.set_title(f'Status: {status}', fontsize=14, fontweight='bold',
                     color=status_color)
        ax2.legend()
        
        # Add info text
        info_text = f'Depth: {depth}\n'
        info_text += f'Box: [{box_range[0]:.6f}, {box_range[1]:.6f}]\n'
        info_text += f'Width: {box_range[1] - box_range[0]:.6f}\n'
        info_text += f'Coeffs: [{np.min(coeffs):.6f}, {np.max(coeffs):.6f}]\n'
        info_text += f'Tolerance: {self.tolerance:.6e}\n'
        info_text += f'Subdiv tol: {self.subdivision_tolerance:.6e}'
        
        fig.text(0.02, 0.98, info_text, fontsize=10, verticalalignment='top',
                fontfamily='monospace', bbox=dict(boxstyle='round', 
                facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        filename = f'debug_step_{self.boxes_processed:04d}_depth{depth}_{status.lower()}.png'
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filename}")
    
    def solve_recursive(self, coeffs, box_range, depth=0, max_depth=30, crit=0.8):
        """Recursively solve with visualization using FIXED PP method logic."""
        self.boxes_processed += 1

        print(f"\n{'  ' * depth}[Depth {depth}] Processing box [{box_range[0]:.6f}, {box_range[1]:.6f}]")

        # Step 1: Apply PP method to find tighter bounds
        pp_result = find_root_box_pp_1d(coeffs, tolerance=self.subdivision_tolerance)

        if pp_result is None:
            # No roots in this box - PRUNE
            print(f"{'  ' * depth}  → PRUNED (PP method: no roots)")
            self.boxes_pruned += 1
            self.visualize_box(depth, box_range, coeffs, 'PRUNED', None)
            return

        t_min, t_max = pp_result
        pp_width = t_max - t_min

        # Map to original box range
        box_width = box_range[1] - box_range[0]
        tight_min = box_range[0] + t_min * box_width
        tight_max = box_range[0] + t_max * box_width
        tight_width = tight_max - tight_min

        print(f"{'  ' * depth}  PP bounds: [{t_min:.6f}, {t_max:.6f}] in [0,1]")
        print(f"{'  ' * depth}  Mapped to: [{tight_min:.6f}, {tight_max:.6f}]")
        print(f"{'  ' * depth}  PP width: {pp_width:.6f} (CRIT: {crit:.6f})")
        print(f"{'  ' * depth}  Reduction: {(1 - pp_width) * 100:.2f}%")

        # Step 2: Check if box is small enough
        if tight_width < self.tolerance:
            # Found solution
            solution = (tight_min + tight_max) / 2
            print(f"{'  ' * depth}  → SOLUTION at t = {solution:.6f}")
            self.solutions.append(solution)
            self.visualize_box(depth, box_range, coeffs, 'SOLUTION', pp_result)
            return

        # Step 3: Check depth limit
        if depth >= max_depth:
            solution = (tight_min + tight_max) / 2
            print(f"{'  ' * depth}  → MAX DEPTH REACHED, claiming solution at t = {solution:.6f}")
            self.solutions.append(solution)
            self.visualize_box(depth, box_range, coeffs, 'MAX_DEPTH', pp_result)
            return

        # Step 4: Check CRIT - should we subdivide or tighten?
        if pp_width > crit:
            # PP didn't reduce enough (< (1-CRIT)% reduction) - SUBDIVIDE
            print(f"{'  ' * depth}  → PP width {pp_width:.6f} > CRIT {crit:.6f}")
            print(f"{'  ' * depth}  → SUBDIVIDE (PP reduction < {(1-crit)*100:.0f}%)")
            self.boxes_subdivided += 1
            self.visualize_box(depth, box_range, coeffs, 'SUBDIVIDE', pp_result)

            # Subdivide at midpoint of PP bounds (in [0,1] space)
            t_mid = (t_min + t_max) / 2

            # Subdivide coefficients
            left_coeffs, right_coeffs = de_casteljau_subdivide_1d(coeffs, t_mid)

            # Map to original box range
            mid_point = box_range[0] + t_mid * box_width

            print(f"{'  ' * depth}  Subdividing at t = {t_mid:.6f} (original: {mid_point:.6f})")

            # Recursively process left and right
            left_range = (box_range[0], mid_point)
            right_range = (mid_point, box_range[1])

            self.solve_recursive(left_coeffs, left_range, depth + 1, max_depth, crit)
            self.solve_recursive(right_coeffs, right_range, depth + 1, max_depth, crit)
        else:
            # PP reduced significantly (≥ (1-CRIT)% reduction) - TIGHTEN
            print(f"{'  ' * depth}  → PP width {pp_width:.6f} ≤ CRIT {crit:.6f}")
            print(f"{'  ' * depth}  → TIGHTEN (PP reduction ≥ {(1-crit)*100:.0f}%)")
            print(f"{'  ' * depth}  → Extract sub-box and apply PP again")
            self.visualize_box(depth, box_range, coeffs, 'TIGHTEN', pp_result)

            # Extract sub-box coefficients
            from intersection.de_casteljau import extract_subbox_1d
            tight_coeffs = extract_subbox_1d(coeffs, t_min, t_max,
                                            tolerance=self.subdivision_tolerance)

            # Recursively process the tightened box (SAME depth - we're tightening, not subdividing)
            tight_range = (tight_min, tight_max)
            self.solve_recursive(tight_coeffs, tight_range, depth, max_depth, crit)


if __name__ == "__main__":
    print("=" * 80)
    print("PP METHOD STEP-BY-STEP DEBUG VISUALIZATION")
    print("=" * 80)
    
    # Test polynomial: (x - 0.2)(x - 0.5)(x - 0.8)
    roots = [0.2, 0.5, 0.8]
    
    print(f"\nPolynomial: (x - 0.2)(x - 0.5)(x - 0.8) = 0")
    print(f"Expected roots: {roots}")
    
    # Expand to power form
    power_coeffs = expand_polynomial_product(roots)
    print(f"\nPower coefficients: {power_coeffs}")
    
    # Convert to Bernstein
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    print(f"Bernstein coefficients: {bern_coeffs}")
    print(f"Coefficient magnitude: {np.max(np.abs(bern_coeffs)):.6e}")
    
    # Create visualizer
    visualizer = PPDebugVisualizer(
        bern_coeffs,
        roots,
        tolerance=1e-6,
        subdivision_tolerance=1e-10
    )
    
    # Solve with visualization
    print("\n" + "=" * 80)
    print("STARTING SUBDIVISION")
    print("=" * 80)
    
    visualizer.solve_recursive(bern_coeffs, (0.0, 1.0), depth=0, max_depth=30, crit=0.8)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Boxes processed: {visualizer.boxes_processed}")
    print(f"Boxes pruned: {visualizer.boxes_pruned}")
    print(f"Boxes subdivided: {visualizer.boxes_subdivided}")
    print(f"Solutions found: {len(visualizer.solutions)}")

    print(f"\nFound solutions:")
    for i, sol in enumerate(sorted(visualizer.solutions)):
        print(f"  {i+1}. t = {sol:.6f}")

    print(f"\nExpected solutions:")
    for i, root in enumerate(roots):
        print(f"  {i+1}. t = {root:.6f}")

    # Check accuracy
    found_sorted = sorted(visualizer.solutions)
    expected_sorted = sorted(roots)

    if len(found_sorted) == len(expected_sorted):
        errors = [abs(f - e) for f, e in zip(found_sorted, expected_sorted)]
        print(f"\nErrors: {errors}")
        print(f"Max error: {max(errors):.6e}")
        print("\n✓ SUCCESS: All roots found!")
    else:
        print(f"\n✗ FAILURE: Expected {len(expected_sorted)} roots, found {len(found_sorted)}")

