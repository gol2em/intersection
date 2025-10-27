"""
Visualize the multidimensional PP method logic.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np


def create_2d_hybrid_diagram():
    """Show 2D hybrid tighten+subdivide in one step."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Common settings
    for ax in axes:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.set_xlabel('Dimension 0 (x)', fontsize=11)
        ax.set_ylabel('Dimension 1 (y)', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # ========== Panel 1: Original Box ==========
    ax = axes[0]
    ax.set_title('Step 1: Original Box', fontsize=14, fontweight='bold')
    
    # Original box [0,1] × [0,1]
    rect = Rectangle((0, 0), 1, 1, linewidth=3, edgecolor='blue', 
                     facecolor='lightblue', alpha=0.3)
    ax.add_patch(rect)
    ax.text(0.5, 0.5, '[0, 1] × [0, 1]', ha='center', va='center',
           fontsize=12, fontweight='bold')
    
    # ========== Panel 2: PP Bounds ==========
    ax = axes[1]
    ax.set_title('Step 2: Apply PP Method', fontsize=14, fontweight='bold')
    
    # Original box (faded)
    rect = Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='gray', 
                     facecolor='lightgray', alpha=0.1, linestyle='--')
    ax.add_patch(rect)
    
    # PP bounds
    pp_x = (0.1, 0.95)
    pp_y = (0.3, 0.5)
    rect = Rectangle((pp_x[0], pp_y[0]), pp_x[1]-pp_x[0], pp_y[1]-pp_y[0],
                     linewidth=3, edgecolor='purple', facecolor='plum', alpha=0.3)
    ax.add_patch(rect)
    
    # Annotations
    ax.text(0.525, 0.4, f'PP bounds:\nx ∈ [{pp_x[0]}, {pp_x[1]}]\ny ∈ [{pp_y[0]}, {pp_y[1]}]',
           ha='center', va='center', fontsize=10, 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Show reductions
    ax.text(0.525, 0.05, f'Dim 0: width = 0.85 > 0.8 → SUBDIVIDE',
           ha='center', fontsize=9, color='red', fontweight='bold')
    ax.text(0.525, -0.02, f'Dim 1: width = 0.2 ≤ 0.8 → TIGHTEN',
           ha='center', fontsize=9, color='green', fontweight='bold')
    
    # ========== Panel 3: Hybrid Result ==========
    ax = axes[2]
    ax.set_title('Step 3: Hybrid (One Step!)', fontsize=14, fontweight='bold')
    
    # Original box (faded)
    rect = Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='gray', 
                     facecolor='lightgray', alpha=0.1, linestyle='--')
    ax.add_patch(rect)
    
    # Subdivision point
    x_mid = (pp_x[0] + pp_x[1]) / 2
    
    # Left box
    rect_left = Rectangle((pp_x[0], pp_y[0]), x_mid - pp_x[0], pp_y[1] - pp_y[0],
                          linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.5)
    ax.add_patch(rect_left)
    ax.text((pp_x[0] + x_mid) / 2, 0.4, 'Box 0', ha='center', va='center',
           fontsize=11, fontweight='bold')
    
    # Right box
    rect_right = Rectangle((x_mid, pp_y[0]), pp_x[1] - x_mid, pp_y[1] - pp_y[0],
                           linewidth=2, edgecolor='orange', facecolor='lightyellow', alpha=0.5)
    ax.add_patch(rect_right)
    ax.text((x_mid + pp_x[1]) / 2, 0.4, 'Box 1', ha='center', va='center',
           fontsize=11, fontweight='bold')
    
    # Subdivision line
    ax.plot([x_mid, x_mid], [pp_y[0], pp_y[1]], 'r--', linewidth=2)
    ax.text(x_mid, pp_y[1] + 0.03, f'x = {x_mid:.3f}', ha='center', fontsize=9,
           color='red', fontweight='bold')
    
    # Annotations
    ax.text(0.525, 0.05, f'Dim 0: SUBDIVIDED at {x_mid:.3f}',
           ha='center', fontsize=9, color='red', fontweight='bold')
    ax.text(0.525, -0.02, f'Dim 1: TIGHTENED to [{pp_y[0]}, {pp_y[1]}]',
           ha='center', fontsize=9, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('multidim_hybrid_2d.png', dpi=150, bbox_inches='tight')
    print("Saved: multidim_hybrid_2d.png")
    plt.close()


def create_comparison_diagram():
    """Compare traditional vs PP hybrid subdivision."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Common settings
    for ax in axes:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.set_xlabel('Dimension 0 (x)', fontsize=11)
        ax.set_ylabel('Dimension 1 (y)', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # ========== Panel 1: Traditional Subdivision ==========
    ax = axes[0]
    ax.set_title('Traditional Subdivision\n(No PP)', fontsize=14, fontweight='bold', color='red')
    
    # 4 boxes
    boxes = [
        ((0, 0), (0.5, 0.5), 'Box 0'),
        ((0.5, 0), (1, 0.5), 'Box 1'),
        ((0, 0.5), (0.5, 1), 'Box 2'),
        ((0.5, 0.5), (1, 1), 'Box 3'),
    ]
    
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    
    for i, ((x0, y0), (x1, y1), label) in enumerate(boxes):
        rect = Rectangle((x0, y0), x1-x0, y1-y0,
                        linewidth=2, edgecolor='blue', facecolor=colors[i], alpha=0.5)
        ax.add_patch(rect)
        ax.text((x0+x1)/2, (y0+y1)/2, label, ha='center', va='center',
               fontsize=10, fontweight='bold')
    
    # Subdivision lines
    ax.plot([0.5, 0.5], [0, 1], 'k--', linewidth=2)
    ax.plot([0, 1], [0.5, 0.5], 'k--', linewidth=2)
    
    ax.text(0.5, -0.02, 'Result: 4 boxes', ha='center', fontsize=11,
           fontweight='bold', color='red')
    
    # ========== Panel 2: PP Hybrid Subdivision ==========
    ax = axes[1]
    ax.set_title('PP Hybrid Subdivision\n(Tighten + Subdivide)', fontsize=14, fontweight='bold', color='green')
    
    # Original box (faded)
    rect = Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='gray', 
                     facecolor='lightgray', alpha=0.1, linestyle='--')
    ax.add_patch(rect)
    
    # PP bounds
    pp_x = (0.1, 0.95)
    pp_y = (0.3, 0.5)
    x_mid = (pp_x[0] + pp_x[1]) / 2
    
    # Left box
    rect_left = Rectangle((pp_x[0], pp_y[0]), x_mid - pp_x[0], pp_y[1] - pp_y[0],
                          linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.5)
    ax.add_patch(rect_left)
    ax.text((pp_x[0] + x_mid) / 2, 0.4, 'Box 0', ha='center', va='center',
           fontsize=11, fontweight='bold')
    
    # Right box
    rect_right = Rectangle((x_mid, pp_y[0]), pp_x[1] - x_mid, pp_y[1] - pp_y[0],
                           linewidth=2, edgecolor='orange', facecolor='lightyellow', alpha=0.5)
    ax.add_patch(rect_right)
    ax.text((x_mid + pp_x[1]) / 2, 0.4, 'Box 1', ha='center', va='center',
           fontsize=11, fontweight='bold')
    
    # Subdivision line
    ax.plot([x_mid, x_mid], [pp_y[0], pp_y[1]], 'r--', linewidth=2)
    
    # Annotations
    ax.text(0.525, 0.15, 'x: Subdivided', ha='center', fontsize=9, color='red')
    ax.text(0.525, 0.08, 'y: Tightened 80%!', ha='center', fontsize=9, color='green')
    ax.text(0.5, -0.02, 'Result: 2 boxes (tighter!)', ha='center', fontsize=11,
           fontweight='bold', color='green')
    
    plt.tight_layout()
    plt.savefig('multidim_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: multidim_comparison.png")
    plt.close()


def create_3d_example_diagram():
    """Show 3D example with 2 dimensions subdivided, 1 tightened."""
    
    fig = plt.figure(figsize=(14, 10))
    
    # Title
    fig.suptitle('3D Example: 2 Dimensions Subdivided, 1 Tightened', 
                fontsize=16, fontweight='bold')
    
    # Create text-based visualization
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    y = 9.5
    
    # Original box
    ax.text(5, y, 'Original Box: [0, 1]³', ha='center', fontsize=14, fontweight='bold')
    y -= 0.8
    
    # PP results
    ax.text(5, y, 'PP Method Finds:', ha='center', fontsize=12, fontweight='bold')
    y -= 0.6
    ax.text(5, y, 'Dim 0: [0.1, 0.95] → width = 0.85 > 0.8 → SUBDIVIDE', 
           ha='center', fontsize=11, color='red')
    y -= 0.5
    ax.text(5, y, 'Dim 1: [0.2, 0.4] → width = 0.2 ≤ 0.8 → TIGHTEN', 
           ha='center', fontsize=11, color='green')
    y -= 0.5
    ax.text(5, y, 'Dim 2: [0.05, 0.92] → width = 0.87 > 0.8 → SUBDIVIDE', 
           ha='center', fontsize=11, color='red')
    y -= 0.8
    
    # Decision
    ax.text(5, y, 'dims_to_subdivide = [0, 2]', ha='center', fontsize=12,
           fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow'))
    y -= 0.6
    ax.text(5, y, 'Number of boxes = 2² = 4', ha='center', fontsize=11, style='italic')
    y -= 1.0
    
    # Boxes
    ax.text(5, y, 'Boxes Created (in ONE step):', ha='center', fontsize=12, fontweight='bold')
    y -= 0.8
    
    boxes = [
        ('Box 0', '[(0.1, 0.525), (0.2, 0.4), (0.05, 0.485)]', 'Left-Left'),
        ('Box 1', '[(0.1, 0.525), (0.2, 0.4), (0.485, 0.92)]', 'Left-Right'),
        ('Box 2', '[(0.525, 0.95), (0.2, 0.4), (0.05, 0.485)]', 'Right-Left'),
        ('Box 3', '[(0.525, 0.95), (0.2, 0.4), (0.485, 0.92)]', 'Right-Right'),
    ]
    
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    
    for i, (name, bounds, desc) in enumerate(boxes):
        ax.text(1.5, y, f'{name}:', ha='left', fontsize=10, fontweight='bold')
        ax.text(3, y, bounds, ha='left', fontsize=9, family='monospace',
               bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.5))
        ax.text(8.5, y, f'({desc})', ha='right', fontsize=9, style='italic')
        y -= 0.6
    
    y -= 0.4
    
    # Key observation
    ax.text(5, y, 'Key: All boxes have Dim 1 tightened to [0.2, 0.4]!', 
           ha='center', fontsize=12, fontweight='bold', color='green',
           bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green', linewidth=2))
    y -= 0.8
    
    # Efficiency
    ax.text(5, y, 'Efficiency Gain:', ha='center', fontsize=12, fontweight='bold')
    y -= 0.6
    ax.text(5, y, 'Traditional: 2³ = 8 boxes with full [0,1]³ coverage', 
           ha='center', fontsize=10, color='red')
    y -= 0.5
    ax.text(5, y, 'PP Hybrid: 2² = 4 boxes with Dim 1 reduced by 80%', 
           ha='center', fontsize=10, color='green')
    y -= 0.5
    ax.text(5, y, 'Volume reduction: 4 boxes × 0.2 = 0.8 vs 8 boxes × 1.0 = 8.0', 
           ha='center', fontsize=10, color='green', fontweight='bold')
    y -= 0.5
    ax.text(5, y, '→ 10× more efficient!', ha='center', fontsize=11, 
           color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('multidim_3d_example.png', dpi=150, bbox_inches='tight')
    print("Saved: multidim_3d_example.png")
    plt.close()


if __name__ == "__main__":
    print("Creating multidimensional logic diagrams...")
    create_2d_hybrid_diagram()
    create_comparison_diagram()
    create_3d_example_diagram()
    print("\nDone! Created:")
    print("  1. multidim_hybrid_2d.png - 2D hybrid tighten+subdivide")
    print("  2. multidim_comparison.png - Traditional vs PP hybrid")
    print("  3. multidim_3d_example.png - 3D example with mixed operations")

