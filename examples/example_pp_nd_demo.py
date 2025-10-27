"""
Demonstration of enhanced PP method with n-D convex hull bounding.

Shows how the enhanced PP method finds tighter bounding boxes by intersecting
convex hulls for each dimension.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.intersection.bernstein import polynomial_nd_to_bernstein
from src.intersection.convex_hull import find_root_box_pp_nd


def demo_1d_system():
    """Demonstrate 1D system."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: 1D System")
    print("=" * 80)
    
    # f(t) = (t - 0.3)(t - 0.7)
    power_coeffs = np.array([0.21, -1.0, 1.0])
    coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    print("\nEquation: f(t) = (t - 0.3)(t - 0.7)")
    print(f"Bernstein coefficients: {coeffs}")
    
    # Simple method: min/max
    min_val = np.min(coeffs)
    max_val = np.max(coeffs)
    simple_box = None if (min_val > 0 or max_val < 0) else [(0.0, 1.0)]
    
    # Enhanced PP method
    pp_box = find_root_box_pp_nd([coeffs], k=1)
    
    print(f"\nSimple method (min/max): {simple_box}")
    print(f"Enhanced PP method:      {pp_box}")
    
    if simple_box and pp_box:
        simple_width = 1.0
        pp_width = pp_box[0][1] - pp_box[0][0]
        improvement = (simple_width - pp_width) / simple_width * 100
        print(f"\n✅ Improvement: {improvement:.1f}% tighter bounds")
        print(f"   Simple width: {simple_width:.6f}")
        print(f"   PP width:     {pp_width:.6f}")


def demo_1d_multiple_equations():
    """Demonstrate 1D system with multiple equations."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: 1D System with Multiple Equations")
    print("=" * 80)
    
    # f1(t) = t - 0.5
    # f2(t) = (t - 0.4)(t - 0.6)
    
    power_coeffs_1 = np.array([-0.5, 1.0])
    coeffs_1 = polynomial_nd_to_bernstein(power_coeffs_1, k=1)
    
    power_coeffs_2 = np.array([0.24, -1.0, 1.0])
    coeffs_2 = polynomial_nd_to_bernstein(power_coeffs_2, k=1)
    
    print("\nEquation 1: f1(t) = t - 0.5")
    print(f"  Bernstein coefficients: {coeffs_1}")
    
    print("\nEquation 2: f2(t) = (t - 0.4)(t - 0.6)")
    print(f"  Bernstein coefficients: {coeffs_2}")
    
    # Enhanced PP method
    pp_box = find_root_box_pp_nd([coeffs_1, coeffs_2], k=1)
    
    print(f"\nEnhanced PP method: {pp_box}")
    
    if pp_box:
        pp_width = pp_box[0][1] - pp_box[0][0]
        print(f"\n✅ Box width: {pp_width:.6f}")
        print(f"   The intersection of ranges from both equations")
        print(f"   gives a very tight bound!")


def demo_2d_single_equation():
    """Demonstrate 2D system with single equation."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: 2D System - Single Equation")
    print("=" * 80)
    
    # f(u,v) = u - 0.5
    power_coeffs = np.array([
        [-0.5, 0.0],
        [1.0, 0.0]
    ])
    coeffs = polynomial_nd_to_bernstein(power_coeffs, k=2)
    
    print("\nEquation: f(u,v) = u - 0.5")
    print(f"Bernstein coefficients shape: {coeffs.shape}")
    print(f"Coefficients:\n{coeffs}")
    
    # Enhanced PP method
    pp_box = find_root_box_pp_nd([coeffs], k=2)
    
    print(f"\nEnhanced PP method: {pp_box}")
    
    if pp_box:
        u_width = pp_box[0][1] - pp_box[0][0]
        v_width = pp_box[1][1] - pp_box[1][0]
        
        print(f"\n✅ U range: [{pp_box[0][0]:.6f}, {pp_box[0][1]:.6f}] (width: {u_width:.6f})")
        print(f"   V range: [{pp_box[1][0]:.6f}, {pp_box[1][1]:.6f}] (width: {v_width:.6f})")
        print(f"\n   U is tightly bounded (polynomial depends on u)")
        print(f"   V is full range (polynomial independent of v)")


def demo_2d_multiple_equations():
    """Demonstrate 2D system with multiple equations."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: 2D System - Multiple Equations")
    print("=" * 80)
    
    # f1(u,v) = u - 0.5
    # f2(u,v) = v - 0.5
    # Common root at (0.5, 0.5)
    
    power_coeffs_1 = np.array([
        [-0.5, 0.0],
        [1.0, 0.0]
    ])
    coeffs_1 = polynomial_nd_to_bernstein(power_coeffs_1, k=2)
    
    power_coeffs_2 = np.array([
        [-0.5, 1.0],
        [0.0, 0.0]
    ])
    coeffs_2 = polynomial_nd_to_bernstein(power_coeffs_2, k=2)
    
    print("\nEquation 1: f1(u,v) = u - 0.5")
    print(f"  Coefficients shape: {coeffs_1.shape}")
    
    print("\nEquation 2: f2(u,v) = v - 0.5")
    print(f"  Coefficients shape: {coeffs_2.shape}")
    
    # Enhanced PP method
    pp_box = find_root_box_pp_nd([coeffs_1, coeffs_2], k=2)
    
    print(f"\nEnhanced PP method: {pp_box}")
    
    if pp_box:
        u_width = pp_box[0][1] - pp_box[0][0]
        v_width = pp_box[1][1] - pp_box[1][0]
        
        print(f"\n✅ U range: [{pp_box[0][0]:.6f}, {pp_box[0][1]:.6f}] (width: {u_width:.6f})")
        print(f"   V range: [{pp_box[1][0]:.6f}, {pp_box[1][1]:.6f}] (width: {v_width:.6f})")
        print(f"\n   Both dimensions tightly bounded!")
        print(f"   Root at (0.5, 0.5) is in the box")


def demo_comparison():
    """Compare simple vs enhanced PP method."""
    print("\n" + "=" * 80)
    print("COMPARISON: Simple vs Enhanced PP Method")
    print("=" * 80)
    
    test_cases = [
        {
            'name': '1D: f(t) = (t - 0.3)(t - 0.7)',
            'k': 1,
            'coeffs_list': [polynomial_nd_to_bernstein(np.array([0.21, -1.0, 1.0]), k=1)]
        },
        {
            'name': '1D: f(t) = (t - 0.5)^3',
            'k': 1,
            'coeffs_list': [polynomial_nd_to_bernstein(np.array([-0.125, 0.75, -1.5, 1.0]), k=1)]
        },
        {
            'name': '2D: f(u,v) = (u - 0.5)(v - 0.5)',
            'k': 2,
            'coeffs_list': [polynomial_nd_to_bernstein(np.array([[0.25, -0.5], [-0.5, 1.0]]), k=2)]
        }
    ]
    
    print(f"\n{'Test Case':<40} {'Simple':<15} {'Enhanced PP':<20} {'Improvement':<15}")
    print("-" * 90)
    
    for case in test_cases:
        name = case['name']
        k = case['k']
        coeffs_list = case['coeffs_list']
        
        # Simple method
        can_have_root = True
        for coeffs in coeffs_list:
            min_val = np.min(coeffs)
            max_val = np.max(coeffs)
            if min_val > 0 or max_val < 0:
                can_have_root = False
                break
        
        simple_box = [(0.0, 1.0) for _ in range(k)] if can_have_root else None
        
        # Enhanced PP method
        pp_box = find_root_box_pp_nd(coeffs_list, k=k)
        
        # Calculate volumes
        if simple_box:
            simple_volume = np.prod([r[1] - r[0] for r in simple_box])
        else:
            simple_volume = 0.0
        
        if pp_box:
            pp_volume = np.prod([r[1] - r[0] for r in pp_box])
        else:
            pp_volume = 0.0
        
        # Format output
        simple_str = f"{simple_volume:.6f}" if simple_box else "PRUNED"
        pp_str = f"{pp_volume:.6f}" if pp_box else "PRUNED"
        
        if simple_box and pp_box and simple_volume > 0:
            improvement = (simple_volume - pp_volume) / simple_volume * 100
            improvement_str = f"{improvement:.1f}%"
        else:
            improvement_str = "N/A"
        
        print(f"{name:<40} {simple_str:<15} {pp_str:<20} {improvement_str:<15}")


def main():
    """Run all demonstrations."""
    print("=" * 80)
    print("ENHANCED PP METHOD WITH N-D CONVEX HULL BOUNDING")
    print("=" * 80)
    print("\nFor each dimension j:")
    print("  1. Extract j-th component from each equation (univariate polynomial)")
    print("  2. Compute convex hull intersection with x-axis for each equation")
    print("  3. Intersect all ranges to get tightest bound for dimension j")
    print("  4. Construct k-dimensional bounding box")
    
    demo_1d_system()
    demo_1d_multiple_equations()
    demo_2d_single_equation()
    demo_2d_multiple_equations()
    demo_comparison()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nThe enhanced PP method provides:")
    print("  ✅ Tighter bounds than simple min/max")
    print("  ✅ Dimension-wise analysis using convex hull intersection")
    print("  ✅ Intersection of ranges from multiple equations")
    print("  ✅ Handles dimension independence correctly")
    print("  ✅ Significant reduction in search space volume")
    print("\nThis leads to:")
    print("  🚀 Faster convergence in subdivision methods")
    print("  🚀 Fewer boxes to process")
    print("  🚀 More efficient root finding")
    print("=" * 80)


if __name__ == "__main__":
    main()

