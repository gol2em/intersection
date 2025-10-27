"""
Example: Projected Polyhedron (PP) Algorithm Demonstration

Shows how the PP algorithm works for finding polynomial roots.
"""

import sys
sys.path.insert(0, 'D:/Python/Intersection')

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb


def evaluate_bernstein(coeffs, t):
    """Evaluate Bernstein polynomial at t."""
    n = len(coeffs) - 1
    result = 0.0
    for i, b_i in enumerate(coeffs):
        basis = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        result += b_i * basis
    return result


def de_casteljau_subdivide(coeffs, t):
    """Subdivide Bernstein polynomial at t."""
    n = len(coeffs) - 1
    pyramid = [np.array(coeffs, dtype=float)]
    
    for j in range(1, n + 1):
        level = []
        prev_level = pyramid[-1]
        for i in range(n - j + 1):
            val = (1 - t) * prev_level[i] + t * prev_level[i + 1]
            level.append(val)
        pyramid.append(np.array(level))
    
    left_coeffs = np.array([pyramid[j][0] for j in range(n + 1)])
    right_coeffs = np.array([pyramid[n - j][j] for j in range(n + 1)])
    
    return left_coeffs, right_coeffs


def pp_bounds(coeffs):
    """Compute PP bounds: [min, max] of coefficients."""
    return np.min(coeffs), np.max(coeffs)


def solve_pp_1d(coeffs, tolerance=1e-6, max_depth=20, verbose=False):
    """
    Find roots using PP method.
    
    Returns
    -------
    solutions : list of (a, b) intervals
    iterations : list of (a, b, coeffs, p_min, p_max, action) for visualization
    """
    solutions = []
    iterations = []
    queue = [(0.0, 1.0, coeffs, 0)]
    
    iteration_count = 0
    
    while queue:
        a, b, c, depth = queue.pop(0)
        iteration_count += 1
        
        # Compute PP bounds
        p_min, p_max = pp_bounds(c)
        
        if verbose:
            print(f"\nIteration {iteration_count}: Box [{a:.6f}, {b:.6f}], Depth {depth}")
            print(f"  Coeffs: {c}")
            print(f"  Bounds: [{p_min:.6f}, {p_max:.6f}]")
        
        # Exclusion test
        if p_min > 0:
            action = "EXCLUDE (all positive)"
            if verbose:
                print(f"  → {action}")
            iterations.append((a, b, c, p_min, p_max, action))
            continue
        
        if p_max < 0:
            action = "EXCLUDE (all negative)"
            if verbose:
                print(f"  → {action}")
            iterations.append((a, b, c, p_min, p_max, action))
            continue
        
        # Convergence test
        if (b - a) < tolerance or depth >= max_depth:
            action = "SOLUTION"
            if verbose:
                print(f"  → {action}")
            solutions.append((a, b))
            iterations.append((a, b, c, p_min, p_max, action))
            continue
        
        # Subdivide
        action = "SUBDIVIDE"
        if verbose:
            print(f"  → {action}")
        iterations.append((a, b, c, p_min, p_max, action))
        
        left_coeffs, right_coeffs = de_casteljau_subdivide(c, 0.5)
        mid = (a + b) / 2
        
        queue.append((a, mid, left_coeffs, depth + 1))
        queue.append((mid, b, right_coeffs, depth + 1))
    
    return solutions, iterations


def example_simple_root():
    """Example 1: Find root of p(t) = t - 0.5."""
    
    print("=" * 80)
    print("EXAMPLE 1: Simple Root Finding")
    print("=" * 80)
    
    # p(t) = t - 0.5 in Bernstein form (degree 1)
    # p(t) = b_0 * (1-t) + b_1 * t
    # p(t) = -0.5 * (1-t) + 0.5 * t = -0.5 + t
    coeffs = np.array([-0.5, 0.5])
    
    print(f"\nPolynomial: p(t) = t - 0.5")
    print(f"Bernstein coefficients: {coeffs}")
    print(f"Expected root: t = 0.5")
    
    # Solve using PP
    solutions, iterations = solve_pp_1d(coeffs, tolerance=1e-3, verbose=True)
    
    print(f"\n--- Results ---")
    print(f"Number of iterations: {len(iterations)}")
    print(f"Solutions found: {len(solutions)}")
    for i, (a, b) in enumerate(solutions):
        mid = (a + b) / 2
        print(f"  Solution {i+1}: [{a:.6f}, {b:.6f}], midpoint = {mid:.6f}")


def example_quadratic():
    """Example 2: Find roots of p(t) = t^2 - 0.5."""
    
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Quadratic Root Finding")
    print("=" * 80)
    
    # p(t) = t^2 - 0.5 in Bernstein form (degree 2)
    # Convert from power form to Bernstein form
    # p(t) = -0.5 + 0*t + 1*t^2
    # Bernstein: b_0 = -0.5, b_1 = 0, b_2 = 0.5
    coeffs = np.array([-0.5, 0.0, 0.5])
    
    print(f"\nPolynomial: p(t) = t^2 - 0.5")
    print(f"Bernstein coefficients: {coeffs}")
    print(f"Expected root: t = √0.5 ≈ 0.707")
    
    # Solve using PP
    solutions, iterations = solve_pp_1d(coeffs, tolerance=1e-3, verbose=False)
    
    print(f"\n--- Results ---")
    print(f"Number of iterations: {len(iterations)}")
    print(f"Solutions found: {len(solutions)}")
    for i, (a, b) in enumerate(solutions):
        mid = (a + b) / 2
        p_mid = evaluate_bernstein(coeffs, mid)
        print(f"  Solution {i+1}: [{a:.6f}, {b:.6f}], midpoint = {mid:.6f}, p(mid) = {p_mid:.6e}")
    
    # Show first few iterations
    print(f"\n--- First 5 Iterations ---")
    for i, (a, b, c, p_min, p_max, action) in enumerate(iterations[:5]):
        print(f"Iter {i+1}: [{a:.4f}, {b:.4f}] → bounds=[{p_min:.4f}, {p_max:.4f}] → {action}")


def visualize_pp_algorithm():
    """Visualize the PP algorithm in action."""
    
    print("\n\n" + "=" * 80)
    print("EXAMPLE 3: Visualization")
    print("=" * 80)
    
    # Polynomial: p(t) = (t - 0.3)(t - 0.7) = t^2 - t + 0.21
    # Bernstein form (degree 2): b_0 = 0.21, b_1 = -0.29, b_2 = 0.21
    coeffs = np.array([0.21, -0.29, 0.21])
    
    print(f"\nPolynomial: p(t) = (t - 0.3)(t - 0.7)")
    print(f"Bernstein coefficients: {coeffs}")
    print(f"Expected roots: t = 0.3, 0.7")
    
    # Solve
    solutions, iterations = solve_pp_1d(coeffs, tolerance=1e-2, max_depth=10, verbose=False)
    
    print(f"\nSolutions found: {len(solutions)}")
    for i, (a, b) in enumerate(solutions):
        mid = (a + b) / 2
        print(f"  Root {i+1}: t ≈ {mid:.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Polynomial and roots
    ax1 = axes[0, 0]
    t_vals = np.linspace(0, 1, 200)
    p_vals = [evaluate_bernstein(coeffs, t) for t in t_vals]
    
    ax1.plot(t_vals, p_vals, 'b-', linewidth=2, label='p(t)')
    ax1.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax1.plot([0, 1, 2], coeffs, 'ro', markersize=10, label='Bernstein coeffs')
    
    # Mark true roots
    ax1.axvline(0.3, color='g', linestyle='--', alpha=0.5, label='True roots')
    ax1.axvline(0.7, color='g', linestyle='--', alpha=0.5)
    
    # Mark found solutions
    for a, b in solutions:
        mid = (a + b) / 2
        ax1.plot(mid, 0, 'r*', markersize=15)
    
    ax1.set_xlabel('t')
    ax1.set_ylabel('p(t)')
    ax1.set_title('Polynomial and Roots')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Subdivision tree (first 15 iterations)
    ax2 = axes[0, 1]
    for i, (a, b, c, p_min, p_max, action) in enumerate(iterations[:15]):
        color = {'EXCLUDE (all positive)': 'red',
                 'EXCLUDE (all negative)': 'orange',
                 'SUBDIVIDE': 'blue',
                 'SOLUTION': 'green'}[action]
        
        ax2.barh(i, b - a, left=a, height=0.8, color=color, alpha=0.6, edgecolor='black')
        ax2.text((a + b) / 2, i, f'{action[:3]}', ha='center', va='center', fontsize=8)
    
    ax2.set_xlabel('t')
    ax2.set_ylabel('Iteration')
    ax2.set_title('Subdivision Tree (First 15 Iterations)')
    ax2.set_ylim(-1, 15)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Bounds convergence
    ax3 = axes[1, 0]
    subdivide_iters = [(i, a, b, p_min, p_max) for i, (a, b, c, p_min, p_max, action) 
                       in enumerate(iterations) if action == 'SUBDIVIDE']
    
    if subdivide_iters:
        iters, a_vals, b_vals, p_mins, p_maxs = zip(*subdivide_iters)
        widths = [b - a for a, b in zip(a_vals, b_vals)]
        bound_widths = [p_max - p_min for p_min, p_max in zip(p_mins, p_maxs)]
        
        ax3.semilogy(iters, widths, 'b-o', label='Box width', markersize=4)
        ax3.semilogy(iters, bound_widths, 'r-s', label='Bound width', markersize=4)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Width (log scale)')
        ax3.set_title('Convergence: Box and Bound Widths')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics
    ax4 = axes[1, 1]
    actions = [action for _, _, _, _, _, action in iterations]
    action_counts = {
        'EXCLUDE': actions.count('EXCLUDE (all positive)') + actions.count('EXCLUDE (all negative)'),
        'SUBDIVIDE': actions.count('SUBDIVIDE'),
        'SOLUTION': actions.count('SOLUTION')
    }
    
    colors_pie = ['red', 'blue', 'green']
    ax4.pie(action_counts.values(), labels=action_counts.keys(), autopct='%1.1f%%',
            colors=colors_pie, startangle=90)
    ax4.set_title('Action Distribution')
    
    plt.tight_layout()
    plt.savefig('pp_algorithm_demo.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved to: pp_algorithm_demo.png")
    plt.close()


if __name__ == '__main__':
    example_simple_root()
    example_quadratic()
    visualize_pp_algorithm()
    
    print("\n\n" + "=" * 80)
    print("PP ALGORITHM DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Points:")
    print("1. PP uses min/max of Bernstein coefficients for bounding")
    print("2. Boxes are excluded if bounds don't contain 0")
    print("3. Subdivision uses de Casteljau for automatic renormalization")
    print("4. Simple but effective for root finding!")

