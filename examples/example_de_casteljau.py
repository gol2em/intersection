"""
Example: de Casteljau Algorithm Demonstration

Shows how de Casteljau works for:
1. Evaluation at a point
2. Subdivision into left and right pieces
3. Subdivision to arbitrary interval
"""

import sys
sys.path.insert(0, 'D:/Python/Intersection')

import numpy as np
import matplotlib.pyplot as plt


def de_casteljau_eval(coeffs, t):
    """
    Evaluate Bernstein polynomial at t using de Casteljau algorithm.
    
    Parameters
    ----------
    coeffs : array-like
        Bernstein coefficients [b_0, b_1, ..., b_n]
    t : float
        Parameter value in [0, 1]
        
    Returns
    -------
    float
        Polynomial value p(t)
    """
    n = len(coeffs) - 1
    b = np.array(coeffs, dtype=float)
    
    for j in range(1, n + 1):
        for i in range(n - j + 1):
            b[i] = (1 - t) * b[i] + t * b[i + 1]
    
    return b[0]


def de_casteljau_subdivide(coeffs, t):
    """
    Subdivide Bernstein polynomial at t using de Casteljau algorithm.
    
    Parameters
    ----------
    coeffs : array-like
        Bernstein coefficients [b_0, b_1, ..., b_n]
    t : float
        Subdivision point in [0, 1]
        
    Returns
    -------
    left_coeffs : ndarray
        Bernstein coefficients on [0, t] (renormalized to [0, 1])
    right_coeffs : ndarray
        Bernstein coefficients on [t, 1] (renormalized to [0, 1])
    """
    n = len(coeffs) - 1
    
    # Build de Casteljau pyramid
    pyramid = [np.array(coeffs, dtype=float)]
    
    for j in range(1, n + 1):
        level = []
        prev_level = pyramid[-1]
        for i in range(n - j + 1):
            val = (1 - t) * prev_level[i] + t * prev_level[i + 1]
            level.append(val)
        pyramid.append(np.array(level))
    
    # Extract left coefficients (left diagonal)
    left_coeffs = np.array([pyramid[j][0] for j in range(n + 1)])
    
    # Extract right coefficients (right diagonal)
    right_coeffs = np.array([pyramid[n - j][j] for j in range(n + 1)])
    
    return left_coeffs, right_coeffs


def de_casteljau_interval(coeffs, a, b):
    """
    Get Bernstein coefficients on interval [a, b] ⊂ [0, 1].
    
    Parameters
    ----------
    coeffs : array-like
        Bernstein coefficients on [0, 1]
    a, b : float
        Interval endpoints, 0 ≤ a < b ≤ 1
        
    Returns
    -------
    ndarray
        Bernstein coefficients on [a, b] (renormalized to [0, 1])
    """
    # Step 1: Subdivide at a, take right piece [a, 1]
    _, right_coeffs = de_casteljau_subdivide(coeffs, a)
    
    # Step 2: Subdivide at (b-a)/(1-a), take left piece [a, b]
    if a < 1:
        t_relative = (b - a) / (1 - a)
        left_coeffs, _ = de_casteljau_subdivide(right_coeffs, t_relative)
    else:
        left_coeffs = right_coeffs
    
    return left_coeffs


def evaluate_bernstein(coeffs, t):
    """Evaluate Bernstein polynomial using basis functions (for verification)."""
    from scipy.special import comb
    n = len(coeffs) - 1
    result = 0.0
    for i, b_i in enumerate(coeffs):
        basis = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        result += b_i * basis
    return result


def example_evaluation():
    """Example 1: Evaluate polynomial at a point."""
    
    print("=" * 80)
    print("EXAMPLE 1: Evaluation Using de Casteljau")
    print("=" * 80)
    
    # Polynomial: p(t) with Bernstein coefficients [1, 3, 2]
    coeffs = np.array([1.0, 3.0, 2.0])
    t = 0.5
    
    print(f"\nBernstein coefficients: {coeffs}")
    print(f"Evaluate at t = {t}")
    
    # Evaluate using de Casteljau
    print("\n--- de Casteljau Algorithm ---")
    n = len(coeffs) - 1
    b = coeffs.copy()
    
    print(f"Level 0: {b}")
    
    for j in range(1, n + 1):
        for i in range(n - j + 1):
            b[i] = (1 - t) * b[i] + t * b[i + 1]
        print(f"Level {j}: {b[:n-j+1]}")
    
    result_de_casteljau = b[0]
    print(f"\nResult (de Casteljau): p({t}) = {result_de_casteljau}")
    
    # Verify using direct evaluation
    result_direct = evaluate_bernstein(coeffs, t)
    print(f"Result (direct):       p({t}) = {result_direct}")
    print(f"Match: {np.isclose(result_de_casteljau, result_direct)}")


def example_subdivision():
    """Example 2: Subdivide polynomial at t = 0.5."""
    
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Subdivision at t = 0.5")
    print("=" * 80)
    
    # Polynomial with coefficients [1, 3, 2]
    coeffs = np.array([1.0, 3.0, 2.0])
    t = 0.5
    
    print(f"\nOriginal Bernstein coefficients: {coeffs}")
    print(f"Subdivide at t = {t}")
    
    # Subdivide
    left_coeffs, right_coeffs = de_casteljau_subdivide(coeffs, t)
    
    print(f"\nLeft piece [0, {t}]:")
    print(f"  Coefficients: {left_coeffs}")
    print(f"  (These represent the polynomial on [0, {t}], renormalized to [0, 1])")
    
    print(f"\nRight piece [{t}, 1]:")
    print(f"  Coefficients: {right_coeffs}")
    print(f"  (These represent the polynomial on [{t}, 1], renormalized to [0, 1])")
    
    # Verify: evaluate at boundary
    print(f"\n--- Verification ---")
    
    # Original polynomial at t = 0.5
    p_mid_original = evaluate_bernstein(coeffs, t)
    print(f"Original p({t}) = {p_mid_original}")
    
    # Left piece at t = 1 (end of [0, 0.5])
    p_mid_left = evaluate_bernstein(left_coeffs, 1.0)
    print(f"Left piece p(1) = {p_mid_left}")
    
    # Right piece at t = 0 (start of [0.5, 1])
    p_mid_right = evaluate_bernstein(right_coeffs, 0.0)
    print(f"Right piece p(0) = {p_mid_right}")
    
    print(f"\nAll three should match: {np.allclose([p_mid_original, p_mid_left, p_mid_right], p_mid_original)}")


def example_arbitrary_interval():
    """Example 3: Get coefficients on arbitrary interval [0.25, 0.75]."""
    
    print("\n\n" + "=" * 80)
    print("EXAMPLE 3: Subdivision to Arbitrary Interval [0.25, 0.75]")
    print("=" * 80)
    
    coeffs = np.array([1.0, 3.0, 2.0])
    a, b = 0.25, 0.75
    
    print(f"\nOriginal coefficients: {coeffs}")
    print(f"Get coefficients on [{a}, {b}]")
    
    # Get coefficients on [a, b]
    interval_coeffs = de_casteljau_interval(coeffs, a, b)
    
    print(f"\nCoefficients on [{a}, {b}]:")
    print(f"  {interval_coeffs}")
    print(f"  (Renormalized to [0, 1])")
    
    # Verify: evaluate at several points
    print(f"\n--- Verification ---")
    test_points = [0.0, 0.5, 1.0]
    
    for s in test_points:
        # Map s ∈ [0, 1] to t ∈ [a, b]
        t = a + s * (b - a)
        
        # Evaluate original polynomial at t
        p_original = evaluate_bernstein(coeffs, t)
        
        # Evaluate interval polynomial at s
        p_interval = evaluate_bernstein(interval_coeffs, s)
        
        print(f"  s={s:.2f} (t={t:.2f}): original={p_original:.4f}, interval={p_interval:.4f}, match={np.isclose(p_original, p_interval)}")


def visualize_subdivision():
    """Visualize subdivision graphically."""
    
    print("\n\n" + "=" * 80)
    print("EXAMPLE 4: Visual Demonstration")
    print("=" * 80)
    
    # Polynomial with coefficients [0, 4, 1]
    coeffs = np.array([0.0, 4.0, 1.0])
    
    print(f"\nBernstein coefficients: {coeffs}")
    print("Creating visualization...")
    
    # Subdivide at t = 0.5
    left_coeffs, right_coeffs = de_casteljau_subdivide(coeffs, 0.5)
    
    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Original polynomial
    ax1 = axes[0]
    t_vals = np.linspace(0, 1, 100)
    p_vals = [evaluate_bernstein(coeffs, t) for t in t_vals]
    
    ax1.plot(t_vals, p_vals, 'b-', linewidth=2, label='p(t)')
    ax1.plot([0, 1, 2], coeffs, 'ro', markersize=10, label='Bernstein coeffs')
    ax1.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Subdivision at t=0.5')
    ax1.set_xlabel('t')
    ax1.set_ylabel('p(t)')
    ax1.set_title('Original Polynomial on [0, 1]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Left piece
    ax2 = axes[1]
    t_left = np.linspace(0, 0.5, 50)
    p_left_original = [evaluate_bernstein(coeffs, t) for t in t_left]
    
    s_left = np.linspace(0, 1, 50)
    p_left_renorm = [evaluate_bernstein(left_coeffs, s) for s in s_left]
    
    ax2.plot(t_left, p_left_original, 'b-', linewidth=2, label='Original on [0, 0.5]')
    ax2.plot(s_left, p_left_renorm, 'g--', linewidth=2, label='Renormalized to [0, 1]')
    ax2.plot([0, 1, 2], left_coeffs, 'ro', markersize=10, label='New coeffs')
    ax2.set_xlabel('t (blue) / s (green)')
    ax2.set_ylabel('p(t)')
    ax2.set_title('Left Piece [0, 0.5]')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Right piece
    ax3 = axes[2]
    t_right = np.linspace(0.5, 1, 50)
    p_right_original = [evaluate_bernstein(coeffs, t) for t in t_right]
    
    s_right = np.linspace(0, 1, 50)
    p_right_renorm = [evaluate_bernstein(right_coeffs, s) for s in s_right]
    
    ax3.plot(t_right, p_right_original, 'b-', linewidth=2, label='Original on [0.5, 1]')
    ax3.plot(s_right, p_right_renorm, 'r--', linewidth=2, label='Renormalized to [0, 1]')
    ax3.plot([0, 1, 2], right_coeffs, 'ro', markersize=10, label='New coeffs')
    ax3.set_xlabel('t (blue) / s (red)')
    ax3.set_ylabel('p(t)')
    ax3.set_title('Right Piece [0.5, 1]')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('de_casteljau_subdivision.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved to: de_casteljau_subdivision.png")
    plt.close()


if __name__ == '__main__':
    example_evaluation()
    example_subdivision()
    example_arbitrary_interval()
    visualize_subdivision()
    
    print("\n\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. de Casteljau evaluates polynomials using recursive linear interpolation")
    print("2. Subdivision gives coefficients on sub-intervals, renormalized to [0,1]")
    print("3. No explicit renormalization needed - it's automatic!")
    print("4. Perfect for LP/PP methods that assume [0,1] domain")

