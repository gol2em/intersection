"""
Create a visual comparison diagram showing before vs after fix.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def create_comparison_diagram():
    """Create side-by-side comparison of buggy vs fixed workflow."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Common settings
    for ax in [ax1, ax2]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')
    
    # ========== LEFT: BUGGY WORKFLOW ==========
    ax1.text(5, 11.5, 'BEFORE FIX (Buggy)', fontsize=18, fontweight='bold',
            ha='center', color='red')
    
    y = 10.5
    
    # Step 1
    box1 = FancyBboxPatch((1, y-0.4), 8, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax1.add_patch(box1)
    ax1.text(5, y, 'Box [0, 1]', ha='center', fontsize=11, fontweight='bold')
    
    y -= 1.2
    ax1.text(5, y, 'PP: [0.121, 0.879] (24% reduction)', ha='center', fontsize=10, style='italic')
    
    y -= 0.8
    ax1.text(5, y, '✗ WRONG: 0.758 < 0.8 → "small enough"', ha='center', fontsize=10, color='red')
    
    y -= 0.6
    ax1.text(5, y, '✗ Force subdivision anyway!', ha='center', fontsize=10, 
            color='red', fontweight='bold')
    
    y -= 1.0
    # Two children
    box2a = FancyBboxPatch((0.5, y-0.4), 3.5, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='blue', facecolor='lightblue', linewidth=2)
    box2b = FancyBboxPatch((6, y-0.4), 3.5, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax1.add_patch(box2a)
    ax1.add_patch(box2b)
    ax1.text(2.25, y, '[0, 0.5]', ha='center', fontsize=10)
    ax1.text(7.75, y, '[0.5, 1]', ha='center', fontsize=10)
    
    y -= 1.2
    ax1.text(2.25, y, 'PP: 24% reduction', ha='center', fontsize=9, style='italic')
    ax1.text(7.75, y, 'PP: 24% reduction', ha='center', fontsize=9, style='italic')
    
    y -= 0.6
    ax1.text(2.25, y, '✗ Subdivide again!', ha='center', fontsize=9, color='red')
    ax1.text(7.75, y, '✗ Subdivide again!', ha='center', fontsize=9, color='red')
    
    y -= 1.0
    # Four children
    for i, x in enumerate([0.3, 1.7, 6.3, 7.7]):
        box = FancyBboxPatch((x-0.6, y-0.3), 1.2, 0.6, boxstyle="round,pad=0.05",
                            edgecolor='blue', facecolor='lightblue', linewidth=1)
        ax1.add_patch(box)
    
    y -= 1.0
    ax1.text(5, y, '...continues subdividing...', ha='center', fontsize=11, 
            style='italic', color='red')
    
    y -= 1.0
    ax1.text(5, y, 'Result: 64,400+ boxes', ha='center', fontsize=12, 
            fontweight='bold', color='red')
    ax1.text(5, y-0.6, 'Exponential explosion!', ha='center', fontsize=11, color='red')
    
    # ========== RIGHT: FIXED WORKFLOW ==========
    ax2.text(5, 11.5, 'AFTER FIX (Correct)', fontsize=18, fontweight='bold',
            ha='center', color='green')
    
    y = 10.5
    
    # Step 1
    box1 = FancyBboxPatch((1, y-0.4), 8, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax2.add_patch(box1)
    ax2.text(5, y, 'Box [0, 1]', ha='center', fontsize=11, fontweight='bold')
    
    y -= 1.2
    ax2.text(5, y, 'PP: [0.121, 0.879] (24% reduction)', ha='center', fontsize=10, style='italic')
    
    y -= 0.8
    ax2.text(5, y, '✓ CORRECT: 0.758 ≤ 0.8 → PP helped!', ha='center', fontsize=10, color='green')
    
    y -= 0.6
    ax2.text(5, y, '✓ TIGHTEN to [0.121, 0.879]', ha='center', fontsize=10, 
            color='green', fontweight='bold')
    
    y -= 1.0
    # Tightened box
    box2 = FancyBboxPatch((1.5, y-0.4), 7, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='purple', facecolor='plum', linewidth=2)
    ax2.add_patch(box2)
    ax2.text(5, y, 'Box [0.121, 0.879]', ha='center', fontsize=11, fontweight='bold')
    
    y -= 1.2
    ax2.text(5, y, 'PP: [0.181, 0.819] (15.7% reduction)', ha='center', fontsize=10, style='italic')
    
    y -= 0.8
    ax2.text(5, y, '✓ CORRECT: 0.843 > 0.8 → PP didn\'t help much', ha='center', fontsize=10, color='green')
    
    y -= 0.6
    ax2.text(5, y, '✓ SUBDIVIDE', ha='center', fontsize=10, 
            color='blue', fontweight='bold')
    
    y -= 1.0
    # Two children
    box3a = FancyBboxPatch((0.5, y-0.4), 3.5, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='blue', facecolor='lightblue', linewidth=2)
    box3b = FancyBboxPatch((6, y-0.4), 3.5, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax2.add_patch(box3a)
    ax2.add_patch(box3b)
    ax2.text(2.25, y, 'Left', ha='center', fontsize=10)
    ax2.text(7.75, y, 'Right', ha='center', fontsize=10)
    
    y -= 1.0
    ax2.text(2.25, y, '→ Tighten 4×', ha='center', fontsize=9, color='purple', fontweight='bold')
    ax2.text(7.75, y, '→ Tighten 4×', ha='center', fontsize=9, color='purple', fontweight='bold')
    
    y -= 0.6
    ax2.text(2.25, y, '→ Root at 0.2 ✓', ha='center', fontsize=9, color='green')
    ax2.text(7.75, y, '→ Root at 0.8 ✓', ha='center', fontsize=9, color='green')
    
    y -= 1.0
    ax2.text(5, y, 'Result: 14 boxes', ha='center', fontsize=12, 
            fontweight='bold', color='green')
    ax2.text(5, y-0.6, 'Efficient convergence!', ha='center', fontsize=11, color='green')
    
    plt.tight_layout()
    plt.savefig('workflow_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: workflow_comparison.png")
    plt.close()


def create_crit_logic_diagram():
    """Create diagram explaining CRIT logic."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'CRIT Logic (CRIT = 0.8)', fontsize=18, fontweight='bold',
            ha='center')
    
    # Decision tree
    y = 8.5
    
    # Root box
    box = FancyBboxPatch((3, y-0.4), 4, 0.8, boxstyle="round,pad=0.1",
                        edgecolor='black', facecolor='lightgray', linewidth=2)
    ax.add_patch(box)
    ax.text(5, y, 'Apply PP Method', ha='center', fontsize=12, fontweight='bold')
    
    # Arrow down
    arrow = FancyArrowPatch((5, y-0.5), (5, y-1.3),
                           arrowstyle='->', mutation_scale=20, linewidth=2)
    ax.add_patch(arrow)
    
    y -= 1.8
    ax.text(5, y, 'PP width = ?', ha='center', fontsize=11, style='italic')
    
    # Split
    y -= 1.0
    
    # Left branch: width > CRIT
    arrow_left = FancyArrowPatch((5, y), (2, y-1.5),
                                arrowstyle='->', mutation_scale=20, linewidth=2,
                                color='blue')
    ax.add_patch(arrow_left)
    ax.text(3.2, y-0.7, 'width > 0.8', ha='center', fontsize=10, color='blue',
           bbox=dict(boxstyle='round', facecolor='lightblue'))
    
    y_left = y - 2.0
    box_left = FancyBboxPatch((0.5, y_left-0.5), 3, 1.0, boxstyle="round,pad=0.1",
                             edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(box_left)
    ax.text(2, y_left-0.1, 'SUBDIVIDE', ha='center', fontsize=12, 
           fontweight='bold', color='blue')
    ax.text(2, y_left-0.4, '(PP didn\'t help much)', ha='center', fontsize=9,
           style='italic')
    
    # Right branch: width ≤ CRIT
    arrow_right = FancyArrowPatch((5, y), (8, y-1.5),
                                 arrowstyle='->', mutation_scale=20, linewidth=2,
                                 color='purple')
    ax.add_patch(arrow_right)
    ax.text(6.8, y-0.7, 'width ≤ 0.8', ha='center', fontsize=10, color='purple',
           bbox=dict(boxstyle='round', facecolor='plum'))
    
    y_right = y - 2.0
    box_right = FancyBboxPatch((6.5, y_right-0.5), 3, 1.0, boxstyle="round,pad=0.1",
                              edgecolor='purple', facecolor='plum', linewidth=2)
    ax.add_patch(box_right)
    ax.text(8, y_right-0.1, 'TIGHTEN', ha='center', fontsize=12, 
           fontweight='bold', color='purple')
    ax.text(8, y_right-0.4, '(Extract sub-box,', ha='center', fontsize=9,
           style='italic')
    ax.text(8, y_right-0.65, 'apply PP again)', ha='center', fontsize=9,
           style='italic')
    
    # Examples
    y_ex = 1.5
    ax.text(5, y_ex+0.5, 'Examples:', fontsize=12, fontweight='bold', ha='center')
    
    # Example 1
    ax.text(2, y_ex, 'width = 0.843 > 0.8', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax.text(2, y_ex-0.4, '→ Reduction: 15.7%', ha='center', fontsize=9)
    ax.text(2, y_ex-0.7, '→ SUBDIVIDE', ha='center', fontsize=9, 
           color='blue', fontweight='bold')
    
    # Example 2
    ax.text(8, y_ex, 'width = 0.758 ≤ 0.8', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='plum'))
    ax.text(8, y_ex-0.4, '→ Reduction: 24.2%', ha='center', fontsize=9)
    ax.text(8, y_ex-0.7, '→ TIGHTEN', ha='center', fontsize=9, 
           color='purple', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('crit_logic_diagram.png', dpi=150, bbox_inches='tight')
    print("Saved: crit_logic_diagram.png")
    plt.close()


if __name__ == "__main__":
    print("Creating comparison diagrams...")
    create_comparison_diagram()
    create_crit_logic_diagram()
    print("\nDone! Created:")
    print("  1. workflow_comparison.png - Before vs After fix")
    print("  2. crit_logic_diagram.png - CRIT logic explanation")

