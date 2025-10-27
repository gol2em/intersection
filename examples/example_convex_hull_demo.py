"""
Demonstration of convex hull method for PP bounds.

Shows how convex hull intersection gives tighter bounds than simple min/max.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.intersection.bernstein import polynomial_nd_to_bernstein
from src.intersection.convex_hull import find_root_box_pp_1d


def demo_convex_hull_method():
    """Demonstrate convex hull method with various polynomials."""
    
    print("=" * 80)
    print("CONVEX HULL METHOD FOR PP - DEMONSTRATION")
    print("=" * 80)
    print("\nThe convex hull method finds tighter bounds on where roots can exist")
    print("by computing the intersection of the convex hull of Bernstein control")
    print("points with the x-axis (y=0).")
    print()
    
    test_cases = [
        {
            'name': 'Linear: f(t) = t - 0.5',
            'power': np.array([-0.5, 1.0]),
            'true_roots': [0.5]
        },
        {
            'name': 'Quadratic: f(t) = (t - 0.3)(t - 0.7)',
            'power': np.array([0.21, -1.0, 1.0]),
            'true_roots': [0.3, 0.7]
        },
        {
            'name': 'Quadratic: f(t) = t² + 0.5 (no roots)',
            'power': np.array([0.5, 0.0, 1.0]),
            'true_roots': []
        },
        {
            'name': 'Cubic: f(t) = (t - 0.5)³',
            'power': np.array([-0.125, 0.75, -1.5, 1.0]),
            'true_roots': [0.5]
        },
        {
            'name': 'Cubic: f(t) = (t - 0.4)(t - 0.6)(t - 0.8)',
            'power': np.array([-0.192, 1.08, -1.8, 1.0]),
            'true_roots': [0.4, 0.6, 0.8]
        },
        {
            'name': 'Quartic: f(t) = (t - 0.25)(t - 0.5)(t - 0.75)²',
            'power': np.array([0.0703125, -0.5625, 1.5625, -1.875, 1.0]),
            'true_roots': [0.25, 0.5, 0.75]
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print("\n" + "=" * 80)
        print(f"Example {i}: {case['name']}")
        print("=" * 80)
        
        power_coeffs = case['power']
        true_roots = case['true_roots']
        
        # Convert to Bernstein basis
        bernstein_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
        
        print(f"\nPolynomial degree: {len(power_coeffs) - 1}")
        print(f"Bernstein coefficients: {bernstein_coeffs}")
        
        # Simple method: check if 0 is in [min, max]
        min_coeff = np.min(bernstein_coeffs)
        max_coeff = np.max(bernstein_coeffs)
        
        print(f"\nCoefficient range: [{min_coeff:.6f}, {max_coeff:.6f}]")
        
        simple_can_prune = (min_coeff > 1e-10) or (max_coeff < -1e-10)
        simple_box = None if simple_can_prune else (0.0, 1.0)
        
        # Convex hull method
        ch_box = find_root_box_pp_1d(bernstein_coeffs)
        
        print(f"\n{'Method':<25} {'Result':<30} {'Width':<10}")
        print("-" * 70)
        
        if simple_box:
            print(f"{'Simple (min/max)':<25} [{simple_box[0]:.6f}, {simple_box[1]:.6f}]     {1.0:.6f}")
        else:
            print(f"{'Simple (min/max)':<25} {'PRUNED (no roots)':<30}")
        
        if ch_box:
            width = ch_box[1] - ch_box[0]
            print(f"{'Convex Hull':<25} [{ch_box[0]:.6f}, {ch_box[1]:.6f}]     {width:.6f}")
        else:
            print(f"{'Convex Hull':<25} {'PRUNED (no roots)':<30}")
        
        # Show improvement
        if simple_box and ch_box:
            width = ch_box[1] - ch_box[0]
            improvement = (1.0 - width) / 1.0 * 100
            print(f"\n✅ Improvement: {improvement:.1f}% tighter bounds")
        elif simple_box and not ch_box:
            print(f"\n✅ Improvement: Convex hull correctly pruned (simple method failed)")
        elif not simple_box and not ch_box:
            print(f"\n✅ Both methods correctly pruned")
        
        # Show true roots
        if true_roots:
            print(f"\nTrue roots: {true_roots}")
            if ch_box:
                all_contained = all(ch_box[0] <= r <= ch_box[1] for r in true_roots)
                if all_contained:
                    print(f"✅ All roots contained in convex hull bounds")
                else:
                    print(f"❌ WARNING: Some roots outside bounds!")
        else:
            print(f"\nTrue roots: None")
            if ch_box is None:
                print(f"✅ Correctly identified no roots")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nThe convex hull method provides:")
    print("  1. ✅ Tighter bounds than simple min/max")
    print("  2. ✅ Better pruning (can eliminate boxes that simple method cannot)")
    print("  3. ✅ Guaranteed to contain all roots (convex hull property)")
    print("  4. ✅ Efficient computation (O(n log n) for convex hull)")
    print("\nThis is essential for the PP (Projected Polyhedron) method to")
    print("efficiently find roots of polynomial systems using subdivision.")
    print("=" * 80)


if __name__ == "__main__":
    demo_convex_hull_method()

