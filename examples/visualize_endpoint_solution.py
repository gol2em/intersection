"""
Visualize how endpoint solutions are processed.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np


def create_endpoint_diagram():
    """Create diagram showing how endpoint solutions are found twice."""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # ========== Panel 1: Initial Subdivision ==========
    ax = axes[0]
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.5, 2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.text(0.5, 1.8, 'Step 1: Subdivide at 0.5', fontsize=16, fontweight='bold', ha='center')
    
    # Original box
    box = FancyBboxPatch((0.121, 1.2), 0.758, 0.3, 
                        edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(box)
    ax.text(0.5, 1.35, '[0.121, 0.879]', ha='center', fontsize=11, fontweight='bold')
    
    # Arrow down
    arrow = FancyArrowPatch((0.5, 1.15), (0.5, 0.85),
                           arrowstyle='->', mutation_scale=20, linewidth=2)
    ax.add_patch(arrow)
    ax.text(0.55, 1.0, 'Subdivide at 0.5', fontsize=10, style='italic')
    
    # Left box
    box_left = FancyBboxPatch((0.121, 0.4), 0.379, 0.3,
                             edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax.add_patch(box_left)
    ax.text(0.31, 0.55, '[0.121, 0.500]', ha='center', fontsize=10, fontweight='bold')
    ax.text(0.31, 0.25, 'Left box', ha='center', fontsize=9, style='italic')
    
    # Right box
    box_right = FancyBboxPatch((0.500, 0.4), 0.379, 0.3,
                              edgecolor='orange', facecolor='lightyellow', linewidth=2)
    ax.add_patch(box_right)
    ax.text(0.69, 0.55, '[0.500, 0.879]', ha='center', fontsize=10, fontweight='bold')
    ax.text(0.69, 0.25, 'Right box', ha='center', fontsize=9, style='italic')
    
    # Highlight the shared endpoint
    circle = Circle((0.5, 0.55), 0.02, color='red', zorder=10)
    ax.add_patch(circle)
    ax.text(0.5, 0.05, '0.5 is included in BOTH boxes!', ha='center', fontsize=11,
           color='red', fontweight='bold')
    
    # ========== Panel 2: Left Box Finds Root ==========
    ax = axes[1]
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.5, 2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.text(0.5, 1.8, 'Step 8: Left Branch Finds Root at 0.5', fontsize=16, fontweight='bold', ha='center')
    
    # Show the box [0.340, 0.500]
    box = FancyBboxPatch((0.340, 1.2), 0.160, 0.3,
                        edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax.add_patch(box)
    ax.text(0.42, 1.35, '[0.340, 0.500]', ha='center', fontsize=11, fontweight='bold')
    
    # Show [0,1] space
    ax.text(0.42, 1.0, 'In [0,1] space:', ha='center', fontsize=10, style='italic')
    
    # [0,1] box
    box_01 = FancyBboxPatch((0.25, 0.6), 0.34, 0.2,
                           edgecolor='gray', facecolor='white', linewidth=1, linestyle='--')
    ax.add_patch(box_01)
    ax.text(0.42, 0.7, '[0, 1]', ha='center', fontsize=9)
    
    # Control points
    t_vals = np.linspace(0, 1, 4)
    for i, t in enumerate(t_vals):
        x = 0.25 + t * 0.34
        y = 0.7
        if i == 3:  # Last point (t=1.0)
            circle = Circle((x, y), 0.015, color='red', zorder=10)
            ax.add_patch(circle)
            ax.text(x, y-0.15, f't={t:.1f}\n(root!)', ha='center', fontsize=8,
                   color='red', fontweight='bold')
        else:
            circle = Circle((x, y), 0.01, color='blue', zorder=5)
            ax.add_patch(circle)
            ax.text(x, y-0.1, f't={t:.1f}', ha='center', fontsize=7)
    
    # Arrow to result
    arrow = FancyArrowPatch((0.42, 0.5), (0.42, 0.2),
                           arrowstyle='->', mutation_scale=20, linewidth=2, color='green')
    ax.add_patch(arrow)
    
    # Result
    ax.text(0.42, 0.05, 'PP bounds: [1.0, 1.0] → Maps to [0.5, 0.5]', ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green', linewidth=2))
    ax.text(0.42, -0.2, '✓ SOLUTION at t = 0.500 (right endpoint)', ha='center', fontsize=10,
           color='green', fontweight='bold')
    
    # ========== Panel 3: Right Box Finds Root ==========
    ax = axes[2]
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.5, 2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.text(0.5, 1.8, 'Step 10: Right Branch Finds Root at 0.5 (Duplicate!)', 
           fontsize=16, fontweight='bold', ha='center', color='red')
    
    # Show the box [0.500, 0.660]
    box = FancyBboxPatch((0.500, 1.2), 0.160, 0.3,
                        edgecolor='orange', facecolor='lightyellow', linewidth=2)
    ax.add_patch(box)
    ax.text(0.58, 1.35, '[0.500, 0.660]', ha='center', fontsize=11, fontweight='bold')
    
    # Show [0,1] space
    ax.text(0.58, 1.0, 'In [0,1] space:', ha='center', fontsize=10, style='italic')
    
    # [0,1] box
    box_01 = FancyBboxPatch((0.41, 0.6), 0.34, 0.2,
                           edgecolor='gray', facecolor='white', linewidth=1, linestyle='--')
    ax.add_patch(box_01)
    ax.text(0.58, 0.7, '[0, 1]', ha='center', fontsize=9)
    
    # Control points
    t_vals = np.linspace(0, 1, 4)
    for i, t in enumerate(t_vals):
        x = 0.41 + t * 0.34
        y = 0.7
        if i == 0:  # First point (t=0.0)
            circle = Circle((x, y), 0.015, color='red', zorder=10)
            ax.add_patch(circle)
            ax.text(x, y-0.15, f't={t:.1f}\n(root!)', ha='center', fontsize=8,
                   color='red', fontweight='bold')
        else:
            circle = Circle((x, y), 0.01, color='blue', zorder=5)
            ax.add_patch(circle)
            ax.text(x, y-0.1, f't={t:.1f}', ha='center', fontsize=7)
    
    # Arrow to result
    arrow = FancyArrowPatch((0.58, 0.5), (0.58, 0.2),
                           arrowstyle='->', mutation_scale=20, linewidth=2, color='orange')
    ax.add_patch(arrow)
    
    # Result
    ax.text(0.58, 0.05, 'PP bounds: [0.0, 0.0] → Maps to [0.5, 0.5]', ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', linewidth=2))
    ax.text(0.58, -0.2, '✓ SOLUTION at t = 0.500 (left endpoint)', ha='center', fontsize=10,
           color='orange', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('endpoint_solution_diagram.png', dpi=150, bbox_inches='tight')
    print("Saved: endpoint_solution_diagram.png")
    plt.close()


def create_deduplication_diagram():
    """Show how deduplication works."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    ax.text(5, 5.5, 'Deduplication Process', fontsize=18, fontweight='bold', ha='center')
    
    # Before deduplication
    y = 4.5
    ax.text(2, y, 'Before Deduplication:', fontsize=12, fontweight='bold')
    
    solutions_before = [
        ('0.200000', 'green'),
        ('0.500000', 'orange'),
        ('0.500000', 'red'),
        ('0.800000', 'green')
    ]
    
    y -= 0.8
    for i, (sol, color) in enumerate(solutions_before):
        x = 1 + i * 0.8
        circle = Circle((x, y), 0.15, color=color, alpha=0.7, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, f'{i+1}', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        ax.text(x, y-0.5, sol, ha='center', fontsize=8)
    
    # Highlight duplicates
    ax.plot([1.8, 2.6], [y, y], 'r--', linewidth=2)
    ax.text(2.2, y+0.4, 'Duplicates!', ha='center', fontsize=9, color='red', fontweight='bold')
    
    # Arrow
    arrow = FancyArrowPatch((5, y-0.8), (5, y-1.5),
                           arrowstyle='->', mutation_scale=30, linewidth=3, color='blue')
    ax.add_patch(arrow)
    ax.text(5.5, y-1.15, 'Remove duplicates\nwithin tolerance', fontsize=10, style='italic')
    
    # After deduplication
    y = 1.5
    ax.text(2, y, 'After Deduplication:', fontsize=12, fontweight='bold')
    
    solutions_after = [
        ('0.200000', 'green'),
        ('0.500000', 'green'),
        ('0.800000', 'green')
    ]
    
    y -= 0.8
    for i, (sol, color) in enumerate(solutions_after):
        x = 1.5 + i * 1.2
        circle = Circle((x, y), 0.15, color=color, alpha=0.7, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, f'{i+1}', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        ax.text(x, y-0.5, sol, ha='center', fontsize=8)
    
    ax.text(5, 0.2, '✓ 3 unique solutions', ha='center', fontsize=12, 
           color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('deduplication_diagram.png', dpi=150, bbox_inches='tight')
    print("Saved: deduplication_diagram.png")
    plt.close()


if __name__ == "__main__":
    print("Creating endpoint solution diagrams...")
    create_endpoint_diagram()
    create_deduplication_diagram()
    print("\nDone! Created:")
    print("  1. endpoint_solution_diagram.png - How endpoint solutions are found twice")
    print("  2. deduplication_diagram.png - How duplicates are removed")

