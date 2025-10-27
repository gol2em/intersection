"""
Debug Subdivision Visualizer

Wraps the subdivision solver to capture and visualize each step.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.intersection.bernstein import evaluate_bernstein_kd
from src.intersection.convex_hull import convex_hull_2d, intersect_convex_hull_with_x_axis
from src.intersection.de_casteljau import de_casteljau_subdivide_1d


class SubdivisionVisualizer:
    """Captures subdivision steps for visualization."""
    
    def __init__(self, bern_coeffs, roots, tolerance=1e-7):
        self.bern_coeffs = bern_coeffs
        self.roots = roots
        self.tolerance = tolerance
        self.degree = len(bern_coeffs) - 1
        
        # Track all boxes processed
        self.boxes = []
        self.solutions = []
        
        # Statistics
        self.boxes_processed = 0
        self.boxes_pruned = 0
        self.subdivisions = 0
        
    def subdivide_recursive(self, coeffs, box_range, depth=0, max_depth=50):
        """Recursively subdivide with tracking."""
        self.boxes_processed += 1
        
        # Get control points
        control_points = np.array([[i / self.degree, coeffs[i]] for i in range(self.degree + 1)])
        
        # Compute convex hull intersection
        result = intersect_convex_hull_with_x_axis(control_points)
        
        # Store box info
        box_info = {
            'depth': depth,
            'range': box_range,
            'coeffs': coeffs.copy(),
            'control_points': control_points.copy(),
            'intersection': result,
            'pruned': result is None,
            'box_id': self.boxes_processed
        }
        self.boxes.append(box_info)
        
        if result is None:
            # Pruned
            self.boxes_pruned += 1
            if depth <= 5:
                print(f"  [Depth {depth}] Box {self.boxes_processed}: [{box_range[0]:.6f}, {box_range[1]:.6f}] → PRUNED")
            return
        
        t_min, t_max = result
        box_width = box_range[1] - box_range[0]
        
        if depth <= 5:
            print(f"  [Depth {depth}] Box {self.boxes_processed}: [{box_range[0]:.6f}, {box_range[1]:.6f}] → bounds [{t_min:.6f}, {t_max:.6f}]")
        
        # Check if small enough
        if box_width < self.tolerance:
            # Found solution
            solution = (box_range[0] + box_range[1]) / 2
            self.solutions.append(solution)
            if depth <= 5:
                print(f"    → SOLUTION at x = {solution:.8f}")
            return
        
        if depth >= max_depth:
            # Max depth reached
            solution = (box_range[0] + box_range[1]) / 2
            self.solutions.append(solution)
            if depth <= 5:
                print(f"    → MAX DEPTH, solution at x = {solution:.8f}")
            return
        
        # Subdivide
        self.subdivisions += 1
        mid = (box_range[0] + box_range[1]) / 2
        left_coeffs, right_coeffs = de_casteljau_subdivide_1d(coeffs, t=0.5)
        
        # Process left and right
        self.subdivide_recursive(left_coeffs, (box_range[0], mid), depth + 1, max_depth)
        self.subdivide_recursive(right_coeffs, (mid, box_range[1]), depth + 1, max_depth)
    
    def solve(self, max_depth=50):
        """Run subdivision and return solutions."""
        print("\n" + "=" * 80)
        print("STEP 3: SUBDIVISION PROCESS (with detailed logging)")
        print("=" * 80)
        print(f"\nTolerance: {self.tolerance}")
        print(f"Max depth: {max_depth}")
        print(f"\nStarting subdivision...")
        
        self.subdivide_recursive(self.bern_coeffs, (0.0, 1.0), depth=0, max_depth=max_depth)
        
        print(f"\n" + "=" * 80)
        print("SUBDIVISION STATISTICS")
        print("=" * 80)
        print(f"Boxes processed: {self.boxes_processed}")
        print(f"Boxes pruned: {self.boxes_pruned} ({self.boxes_pruned / self.boxes_processed * 100:.1f}%)")
        print(f"Subdivisions: {self.subdivisions}")
        print(f"Solutions found: {len(self.solutions)}")
        
        return sorted(self.solutions)
    
    def visualize_by_depth(self, max_depth_to_show=4):
        """Visualize boxes grouped by depth."""
        print("\n" + "=" * 80)
        print("STEP 4: VISUALIZATION BY DEPTH")
        print("=" * 80)
        
        # Group by depth
        max_depth = max(b['depth'] for b in self.boxes)
        
        for depth in range(min(max_depth_to_show, max_depth + 1)):
            boxes_at_depth = [b for b in self.boxes if b['depth'] == depth]
            
            if not boxes_at_depth:
                continue
            
            print(f"\nDepth {depth}: {len(boxes_at_depth)} boxes")
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            
            # Plot full polynomial
            x_vals = np.linspace(0, 1, 1000)
            y_vals = np.array([evaluate_bernstein_kd(self.bern_coeffs, x) for x in x_vals])
            ax.plot(x_vals, y_vals, 'b-', linewidth=1, alpha=0.3, label='Full polynomial', zorder=1)
            ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, zorder=2)
            
            # Plot each box
            colors = plt.cm.tab20(np.linspace(0, 1, len(boxes_at_depth)))
            
            for idx, (box, color) in enumerate(zip(boxes_at_depth, colors)):
                box_range = box['range']
                coeffs = box['coeffs']
                control_points = box['control_points']
                
                # Map control points to box range
                cp_x = box_range[0] + control_points[:, 0] * (box_range[1] - box_range[0])
                cp_y = control_points[:, 1]
                
                if box['pruned']:
                    # Pruned box - show in red
                    ax.axvspan(box_range[0], box_range[1], alpha=0.1, color='red', zorder=3)
                    ax.text((box_range[0] + box_range[1]) / 2, 0, 'X', 
                           ha='center', va='center', fontsize=8, color='red', fontweight='bold', zorder=10)
                else:
                    # Active box - show convex hull
                    hull_points = convex_hull_2d(control_points)
                    hull_x = box_range[0] + hull_points[:, 0] * (box_range[1] - box_range[0])
                    hull_y = hull_points[:, 1]
                    hull_x_closed = np.append(hull_x, hull_x[0])
                    hull_y_closed = np.append(hull_y, hull_y[0])
                    
                    ax.fill(hull_x_closed, hull_y_closed, color=color, alpha=0.2, zorder=4)
                    ax.plot(hull_x_closed, hull_y_closed, color=color, linewidth=1.5, zorder=5)
                    
                    # Show intersection
                    if box['intersection'] is not None:
                        t_min, t_max = box['intersection']
                        x_min = box_range[0] + t_min * (box_range[1] - box_range[0])
                        x_max = box_range[0] + t_max * (box_range[1] - box_range[0])
                        ax.plot([x_min, x_max], [0, 0], color=color, linewidth=3, zorder=6)
                
                # Mark box boundaries
                ax.axvline(box_range[0], color='gray', linestyle=':', linewidth=0.5, alpha=0.5, zorder=3)
                ax.axvline(box_range[1], color='gray', linestyle=':', linewidth=0.5, alpha=0.5, zorder=3)
            
            # Plot actual roots
            for root in self.roots:
                ax.plot(root, 0, 'r*', markersize=12, zorder=8)
            
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('p(x)', fontsize=12)
            ax.set_title(f'Depth {depth}: {len(boxes_at_depth)} boxes ({sum(1 for b in boxes_at_depth if b["pruned"])} pruned)', 
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            
            plt.tight_layout()
            filename = f'debug_step3_depth{depth}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved: {filename}")
            plt.close()
    
    def visualize_solution_accuracy(self):
        """Visualize how accurate the solutions are."""
        print("\n" + "=" * 80)
        print("STEP 5: SOLUTION ACCURACY ANALYSIS")
        print("=" * 80)
        
        if not self.solutions:
            print("No solutions found!")
            return
        
        # Match solutions to expected roots
        tolerance = 0.02
        matched = []
        unmatched_solutions = []
        unmatched_roots = list(self.roots)
        
        for sol in self.solutions:
            best_match = None
            best_dist = float('inf')
            
            for root in unmatched_roots:
                dist = abs(sol - root)
                if dist < best_dist:
                    best_dist = dist
                    best_match = root
            
            if best_dist < tolerance and best_match is not None:
                matched.append((sol, best_match, best_dist))
                unmatched_roots.remove(best_match)
            else:
                unmatched_solutions.append(sol)
        
        print(f"\nMatched: {len(matched)} / {len(self.roots)}")
        print(f"Unmatched solutions: {len(unmatched_solutions)}")
        print(f"Missing roots: {len(unmatched_roots)}")
        
        if matched:
            errors = [m[2] for m in matched]
            print(f"\nError statistics:")
            print(f"  Mean error: {np.mean(errors):.6e}")
            print(f"  Max error: {np.max(errors):.6e}")
            print(f"  Min error: {np.min(errors):.6e}")
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top: Solutions vs expected roots
        x_vals = np.linspace(0, 1, 1000)
        y_vals = np.array([evaluate_bernstein_kd(self.bern_coeffs, x) for x in x_vals])
        
        ax1.plot(x_vals, y_vals, 'b-', linewidth=2, alpha=0.5, label='Polynomial')
        ax1.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        
        # Expected roots
        for root in self.roots:
            ax1.plot(root, 0, 'go', markersize=10, label='Expected' if root == self.roots[0] else '', zorder=5)
        
        # Found solutions
        for sol in self.solutions:
            ax1.plot(sol, 0, 'rx', markersize=8, markeredgewidth=2, label='Found' if sol == self.solutions[0] else '', zorder=6)
        
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('p(x)', fontsize=12)
        ax1.set_title(f'Solutions: {len(self.solutions)} found, {len(self.roots)} expected', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        
        # Bottom: Error distribution
        if matched:
            errors = [m[2] for m in matched]
            ax2.bar(range(len(errors)), errors, color='blue', alpha=0.7)
            ax2.axhline(y=self.tolerance, color='r', linestyle='--', linewidth=2, label=f'Tolerance = {self.tolerance}')
            ax2.set_xlabel('Solution index', fontsize=12)
            ax2.set_ylabel('Error (|found - expected|)', fontsize=12)
            ax2.set_title('Error for Each Matched Solution', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.legend(fontsize=10)
            ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('debug_step5_accuracy.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved: debug_step5_accuracy.png")
        plt.close()

