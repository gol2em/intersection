"""
Visualize de Casteljau Subdivision Algorithm

This script creates presentation-quality visualizations showing:
1. Control points before subdivision
2. Control points after subdivision (left and right)
3. The Bernstein polynomial curve
4. How subdivision preserves the curve exactly
"""

import sys
sys.path.insert(0, 'D:/Python/Intersection')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from src.intersection.de_casteljau import de_casteljau_subdivide_1d, de_casteljau_eval_1d
from src.intersection.bernstein import evaluate_bernstein_kd


def plot_bernstein_curve_1d(ax, coeffs, color='blue', label='Curve', linewidth=2, alpha=1.0):
    """Plot a 1D Bernstein polynomial curve."""
    t_vals = np.linspace(0, 1, 200)
    y_vals = [de_casteljau_eval_1d(coeffs, t) for t in t_vals]
    ax.plot(t_vals, y_vals, color=color, linewidth=linewidth, label=label, alpha=alpha)


def plot_control_polygon_1d(ax, coeffs, color='red', label='Control Points', 
                            show_polygon=True, marker='o', markersize=8, alpha=1.0):
    """Plot control points and control polygon for 1D Bernstein polynomial."""
    n = len(coeffs) - 1
    t_vals = np.linspace(0, 1, n + 1)
    
    # Plot control polygon
    if show_polygon:
        ax.plot(t_vals, coeffs, color=color, linestyle='--', linewidth=1.5, 
                alpha=alpha*0.6, label=label + ' (polygon)')
    
    # Plot control points
    ax.scatter(t_vals, coeffs, color=color, s=markersize**2, marker=marker, 
               zorder=5, alpha=alpha, edgecolors='black', linewidths=1.5,
               label=label if not show_polygon else None)


def visualize_1d_subdivision_single(coeffs, t_split=0.5, title="de Casteljau Subdivision"):
    """
    Visualize 1D subdivision in a single comprehensive figure.
    
    Shows:
    - Original curve and control points
    - Subdivision point
    - Left and right subdivided curves with their control points
    """
    # Perform subdivision
    left_coeffs, right_coeffs = de_casteljau_subdivide_1d(coeffs, t_split)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ========== Panel 1: Original Polynomial ==========
    ax = axes[0]
    ax.set_title('Original Polynomial', fontsize=14, fontweight='bold')
    
    # Plot curve
    plot_bernstein_curve_1d(ax, coeffs, color='blue', label='Original curve', linewidth=3)
    
    # Plot control points
    plot_control_polygon_1d(ax, coeffs, color='red', label='Control points')
    
    # Mark subdivision point
    y_split = de_casteljau_eval_1d(coeffs, t_split)
    ax.axvline(t_split, color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'Split at t={t_split}')
    ax.scatter([t_split], [y_split], color='green', s=150, marker='*', 
               zorder=10, edgecolors='black', linewidths=2, label='Split point')
    
    ax.set_xlabel('t', fontsize=12)
    ax.set_ylabel('p(t)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    # ========== Panel 2: Left Subdivision ==========
    ax = axes[1]
    ax.set_title(f'Left: [0, {t_split}] → [0, 1]', fontsize=14, fontweight='bold', color='darkgreen')
    
    # Plot original curve (faded)
    plot_bernstein_curve_1d(ax, coeffs, color='lightgray', label='Original (faded)', 
                           linewidth=2, alpha=0.3)
    
    # Plot left curve
    plot_bernstein_curve_1d(ax, left_coeffs, color='darkgreen', label='Left curve', linewidth=3)
    
    # Plot left control points
    plot_control_polygon_1d(ax, left_coeffs, color='darkred', label='Left control points')
    
    # Shade the region
    t_vals = np.linspace(0, 1, 200)
    y_vals = [de_casteljau_eval_1d(left_coeffs, t) for t in t_vals]
    ax.fill_between(t_vals, y_vals, alpha=0.1, color='green')
    
    ax.set_xlabel('t (renormalized)', fontsize=12)
    ax.set_ylabel('p(t)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    # ========== Panel 3: Right Subdivision ==========
    ax = axes[2]
    ax.set_title(f'Right: [{t_split}, 1] → [0, 1]', fontsize=14, fontweight='bold', color='darkblue')
    
    # Plot original curve (faded)
    plot_bernstein_curve_1d(ax, coeffs, color='lightgray', label='Original (faded)', 
                           linewidth=2, alpha=0.3)
    
    # Plot right curve
    plot_bernstein_curve_1d(ax, right_coeffs, color='darkblue', label='Right curve', linewidth=3)
    
    # Plot right control points
    plot_control_polygon_1d(ax, right_coeffs, color='darkorange', label='Right control points')
    
    # Shade the region
    t_vals = np.linspace(0, 1, 200)
    y_vals = [de_casteljau_eval_1d(right_coeffs, t) for t in t_vals]
    ax.fill_between(t_vals, y_vals, alpha=0.1, color='blue')
    
    ax.set_xlabel('t (renormalized)', fontsize=12)
    ax.set_ylabel('p(t)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


def visualize_1d_subdivision_overlay(coeffs, t_split=0.5, title="de Casteljau Subdivision - Overlay"):
    """
    Visualize 1D subdivision with all curves overlaid.
    
    Shows how the left and right curves exactly match the original.
    """
    # Perform subdivision
    left_coeffs, right_coeffs = de_casteljau_subdivide_1d(coeffs, t_split)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot original curve
    plot_bernstein_curve_1d(ax, coeffs, color='blue', label='Original curve', linewidth=4, alpha=0.5)
    
    # Plot original control points
    plot_control_polygon_1d(ax, coeffs, color='red', label='Original control points', 
                           show_polygon=True, markersize=10, alpha=0.5)
    
    # Plot left curve (on [0, t_split] mapped to [0, 1])
    t_left = np.linspace(0, t_split, 200)
    y_left = [de_casteljau_eval_1d(left_coeffs, t / t_split) for t in t_left]
    ax.plot(t_left, y_left, color='darkgreen', linewidth=3, label='Left subdivision', linestyle='-')
    
    # Plot left control points (mapped back to original domain)
    n_left = len(left_coeffs) - 1
    t_left_ctrl = np.linspace(0, t_split, n_left + 1)
    ax.plot(t_left_ctrl, left_coeffs, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7)
    ax.scatter(t_left_ctrl, left_coeffs, color='darkgreen', s=100, marker='s', 
               zorder=5, edgecolors='black', linewidths=1.5, label='Left control points')
    
    # Plot right curve (on [t_split, 1] mapped to [0, 1])
    t_right = np.linspace(t_split, 1, 200)
    y_right = [de_casteljau_eval_1d(right_coeffs, (t - t_split) / (1 - t_split)) for t in t_right]
    ax.plot(t_right, y_right, color='darkblue', linewidth=3, label='Right subdivision', linestyle='-')
    
    # Plot right control points (mapped back to original domain)
    n_right = len(right_coeffs) - 1
    t_right_ctrl = np.linspace(t_split, 1, n_right + 1)
    ax.plot(t_right_ctrl, right_coeffs, color='darkblue', linestyle='--', linewidth=2, alpha=0.7)
    ax.scatter(t_right_ctrl, right_coeffs, color='darkblue', s=100, marker='^', 
               zorder=5, edgecolors='black', linewidths=1.5, label='Right control points')
    
    # Mark subdivision point
    y_split = de_casteljau_eval_1d(coeffs, t_split)
    ax.axvline(t_split, color='purple', linestyle=':', linewidth=3, alpha=0.7, 
               label=f'Subdivision at t={t_split}')
    ax.scatter([t_split], [y_split], color='purple', s=300, marker='*', 
               zorder=10, edgecolors='black', linewidths=2, label='Split point')
    
    ax.set_xlabel('t', fontsize=14, fontweight='bold')
    ax.set_ylabel('p(t)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    plt.tight_layout()
    return fig


def main():
    """Generate de Casteljau subdivision visualizations."""
    print("=" * 80)
    print("DE CASTELJAU SUBDIVISION VISUALIZATION")
    print("=" * 80)
    
    # Example 1: Quadratic polynomial p(t) = t^2
    print("\nExample 1: Quadratic p(t) = t²")
    coeffs_quad = np.array([0.0, 0.0, 1.0])
    
    fig1 = visualize_1d_subdivision_single(coeffs_quad, t_split=0.5, 
                                           title="de Casteljau Subdivision: p(t) = t²")
    fig1.savefig('de_casteljau_quadratic_panels.png', dpi=150, bbox_inches='tight')
    print("Saved: de_casteljau_quadratic_panels.png")
    plt.close(fig1)
    
    fig2 = visualize_1d_subdivision_overlay(coeffs_quad, t_split=0.5,
                                            title="de Casteljau Subdivision: p(t) = t² (Overlay)")
    fig2.savefig('de_casteljau_quadratic_overlay.png', dpi=150, bbox_inches='tight')
    print("Saved: de_casteljau_quadratic_overlay.png")
    plt.close(fig2)
    
    # Example 2: Cubic polynomial with interesting shape
    print("\nExample 2: Cubic p(t) = (t-0.2)(t-0.5)(t-0.8)")
    # Expand: t³ - 1.5t² + 0.62t - 0.08
    # Bernstein form for degree 3
    from src.intersection.bernstein import polynomial_nd_to_bernstein
    power_coeffs = np.array([-0.08, 0.62, -1.5, 1.0])
    coeffs_cubic = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    fig3 = visualize_1d_subdivision_single(coeffs_cubic, t_split=0.5,
                                           title="de Casteljau Subdivision: Cubic Polynomial")
    fig3.savefig('de_casteljau_cubic_panels.png', dpi=150, bbox_inches='tight')
    print("Saved: de_casteljau_cubic_panels.png")
    plt.close(fig3)
    
    fig4 = visualize_1d_subdivision_overlay(coeffs_cubic, t_split=0.5,
                                            title="de Casteljau Subdivision: Cubic Polynomial (Overlay)")
    fig4.savefig('de_casteljau_cubic_overlay.png', dpi=150, bbox_inches='tight')
    print("Saved: de_casteljau_cubic_overlay.png")
    plt.close(fig4)
    
    # Example 3: Higher degree polynomial
    print("\nExample 3: Degree 5 polynomial")
    coeffs_deg5 = np.array([0.1, 0.3, 0.8, 0.6, 0.4, 0.2])
    
    fig5 = visualize_1d_subdivision_single(coeffs_deg5, t_split=0.5,
                                           title="de Casteljau Subdivision: Degree 5 Polynomial")
    fig5.savefig('de_casteljau_deg5_panels.png', dpi=150, bbox_inches='tight')
    print("Saved: de_casteljau_deg5_panels.png")
    plt.close(fig5)
    
    fig6 = visualize_1d_subdivision_overlay(coeffs_deg5, t_split=0.5,
                                            title="de Casteljau Subdivision: Degree 5 Polynomial (Overlay)")
    fig6.savefig('de_casteljau_deg5_overlay.png', dpi=150, bbox_inches='tight')
    print("Saved: de_casteljau_deg5_overlay.png")
    plt.close(fig6)
    
    print("\n" + "=" * 80)
    print("✅ All visualizations generated successfully!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - de_casteljau_quadratic_panels.png")
    print("  - de_casteljau_quadratic_overlay.png")
    print("  - de_casteljau_cubic_panels.png")
    print("  - de_casteljau_cubic_overlay.png")
    print("  - de_casteljau_deg5_panels.png")
    print("  - de_casteljau_deg5_overlay.png")


if __name__ == "__main__":
    main()

