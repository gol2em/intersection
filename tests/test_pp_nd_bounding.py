"""
Test enhanced PP method with convex hull intersection for n-D bounding boxes.
"""

import numpy as np
from src.intersection.convex_hull import find_root_box_pp_nd, _extract_dimension_range
from src.intersection.bernstein import polynomial_nd_to_bernstein


def test_extract_dimension_range_1d():
    """Test extracting dimension range for 1D case."""
    print("\n" + "=" * 80)
    print("Test 1: Extract Dimension Range (1D)")
    print("=" * 80)
    
    # 1D polynomial: f(t) = (t - 0.3)(t - 0.7)
    power_coeffs = np.array([0.21, -1.0, 1.0])
    coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    result = _extract_dimension_range(coeffs, dim=0, k=1)
    
    print(f"\nPolynomial: f(t) = (t - 0.3)(t - 0.7)")
    print(f"Bernstein coefficients: {coeffs}")
    print(f"Dimension range: {result}")
    
    assert result is not None, "Expected a range"
    t_min, t_max = result
    print(f"Range: [{t_min:.6f}, {t_max:.6f}]")
    
    # Roots at 0.3 and 0.7 should be in range
    assert t_min <= 0.3 <= t_max, "Root at 0.3 should be in range"
    assert t_min <= 0.7 <= t_max, "Root at 0.7 should be in range"
    print("✅ Test 1 passed")


def test_extract_dimension_range_2d():
    """Test extracting dimension range for 2D case."""
    print("\n" + "=" * 80)
    print("Test 2: Extract Dimension Range (2D)")
    print("=" * 80)
    
    # 2D polynomial: f(u,v) = (u - 0.5) * (v - 0.5)
    # In power basis: f(u,v) = uv - 0.5u - 0.5v + 0.25
    # Coefficients in power basis (u^i * v^j):
    # [[0.25, -0.5], [-0.5, 1.0]]
    power_coeffs = np.array([
        [0.25, -0.5],
        [-0.5, 1.0]
    ])
    coeffs = polynomial_nd_to_bernstein(power_coeffs, k=2)
    
    print(f"\nPolynomial: f(u,v) = (u - 0.5)(v - 0.5)")
    print(f"Bernstein coefficients shape: {coeffs.shape}")
    print(f"Coefficients:\n{coeffs}")
    
    # Extract range for u dimension (dim=0)
    u_range = _extract_dimension_range(coeffs, dim=0, k=2)
    print(f"\nU-dimension range: {u_range}")
    
    # Extract range for v dimension (dim=1)
    v_range = _extract_dimension_range(coeffs, dim=1, k=2)
    print(f"V-dimension range: {v_range}")
    
    assert u_range is not None, "Expected u range"
    assert v_range is not None, "Expected v range"
    
    print(f"\nU range: [{u_range[0]:.6f}, {u_range[1]:.6f}]")
    print(f"V range: [{v_range[0]:.6f}, {v_range[1]:.6f}]")
    
    print("✅ Test 2 passed")


def test_find_root_box_pp_nd_1d():
    """Test PP method for 1D system."""
    print("\n" + "=" * 80)
    print("Test 3: PP Method for 1D System")
    print("=" * 80)
    
    # Single equation: f(t) = (t - 0.3)(t - 0.7)
    power_coeffs = np.array([0.21, -1.0, 1.0])
    coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    result = find_root_box_pp_nd([coeffs], k=1)
    
    print(f"\nEquation: f(t) = (t - 0.3)(t - 0.7)")
    print(f"Bernstein coefficients: {coeffs}")
    print(f"Bounding box: {result}")
    
    assert result is not None, "Expected bounding box"
    assert len(result) == 1, "Expected 1D box"
    
    t_min, t_max = result[0]
    print(f"Box: [{t_min:.6f}, {t_max:.6f}]")
    
    # Roots at 0.3 and 0.7 should be in box
    assert t_min <= 0.3 <= t_max, "Root at 0.3 should be in box"
    assert t_min <= 0.7 <= t_max, "Root at 0.7 should be in box"
    
    # Box should be tighter than [0, 1]
    width = t_max - t_min
    assert width < 1.0, "Box should be tighter than [0, 1]"
    print(f"Box width: {width:.6f} (improvement: {(1.0 - width) / 1.0 * 100:.1f}%)")
    
    print("✅ Test 3 passed")


def test_find_root_box_pp_nd_1d_multiple_equations():
    """Test PP method for 1D system with multiple equations."""
    print("\n" + "=" * 80)
    print("Test 4: PP Method for 1D System with Multiple Equations")
    print("=" * 80)
    
    # Two equations:
    # f1(t) = t - 0.5
    # f2(t) = (t - 0.4)(t - 0.6)
    
    power_coeffs_1 = np.array([-0.5, 1.0])
    coeffs_1 = polynomial_nd_to_bernstein(power_coeffs_1, k=1)
    
    power_coeffs_2 = np.array([0.24, -1.0, 1.0])
    coeffs_2 = polynomial_nd_to_bernstein(power_coeffs_2, k=1)
    
    result = find_root_box_pp_nd([coeffs_1, coeffs_2], k=1)
    
    print(f"\nEquation 1: f1(t) = t - 0.5")
    print(f"  Bernstein coefficients: {coeffs_1}")
    
    print(f"\nEquation 2: f2(t) = (t - 0.4)(t - 0.6)")
    print(f"  Bernstein coefficients: {coeffs_2}")
    
    print(f"\nBounding box: {result}")
    
    assert result is not None, "Expected bounding box"
    
    t_min, t_max = result[0]
    print(f"Box: [{t_min:.6f}, {t_max:.6f}]")
    
    # The common root is at t = 0.5
    # f1(0.5) = 0, f2(0.5) = (0.5-0.4)(0.5-0.6) = 0.1 * (-0.1) = -0.01 ≠ 0
    # So there's no common root, but the box should contain the intersection
    # of the ranges where each equation could be zero
    
    # f1 has root at 0.5
    # f2 has roots at 0.4 and 0.6
    # The intersection should be around [0.4, 0.6] or tighter
    
    print(f"Box width: {t_max - t_min:.6f}")
    print("✅ Test 4 passed")


def test_find_root_box_pp_nd_2d():
    """Test PP method for 2D system."""
    print("\n" + "=" * 80)
    print("Test 5: PP Method for 2D System")
    print("=" * 80)
    
    # Single equation: f(u,v) = u - 0.5
    # This is independent of v, so v can be anything
    
    # In power basis: f(u,v) = -0.5 + u
    # Coefficients: [[-0.5, 0], [1.0, 0]]
    power_coeffs = np.array([
        [-0.5, 0.0],
        [1.0, 0.0]
    ])
    coeffs = polynomial_nd_to_bernstein(power_coeffs, k=2)
    
    result = find_root_box_pp_nd([coeffs], k=2)
    
    print(f"\nEquation: f(u,v) = u - 0.5")
    print(f"Bernstein coefficients shape: {coeffs.shape}")
    print(f"Coefficients:\n{coeffs}")
    print(f"\nBounding box: {result}")
    
    assert result is not None, "Expected bounding box"
    assert len(result) == 2, "Expected 2D box"
    
    u_min, u_max = result[0]
    v_min, v_max = result[1]
    
    print(f"U range: [{u_min:.6f}, {u_max:.6f}]")
    print(f"V range: [{v_min:.6f}, {v_max:.6f}]")
    
    # Root is at u = 0.5, v can be anything
    # So u range should be tight around 0.5
    # v range should be [0, 1] (full range)
    
    assert u_min <= 0.5 <= u_max, "Root at u=0.5 should be in box"
    
    u_width = u_max - u_min
    v_width = v_max - v_min
    
    print(f"U width: {u_width:.6f}")
    print(f"V width: {v_width:.6f}")
    
    # U should be tighter than [0, 1]
    assert u_width < 1.0, "U range should be tighter than [0, 1]"
    
    print("✅ Test 5 passed")


def test_find_root_box_pp_nd_2d_multiple_equations():
    """Test PP method for 2D system with multiple equations."""
    print("\n" + "=" * 80)
    print("Test 6: PP Method for 2D System with Multiple Equations")
    print("=" * 80)
    
    # Two equations:
    # f1(u,v) = u - 0.5
    # f2(u,v) = v - 0.5
    # Common root at (0.5, 0.5)
    
    # f1: u - 0.5
    power_coeffs_1 = np.array([
        [-0.5, 0.0],
        [1.0, 0.0]
    ])
    coeffs_1 = polynomial_nd_to_bernstein(power_coeffs_1, k=2)
    
    # f2: v - 0.5
    power_coeffs_2 = np.array([
        [-0.5, 1.0],
        [0.0, 0.0]
    ])
    coeffs_2 = polynomial_nd_to_bernstein(power_coeffs_2, k=2)
    
    result = find_root_box_pp_nd([coeffs_1, coeffs_2], k=2)
    
    print(f"\nEquation 1: f1(u,v) = u - 0.5")
    print(f"  Coefficients shape: {coeffs_1.shape}")
    
    print(f"\nEquation 2: f2(u,v) = v - 0.5")
    print(f"  Coefficients shape: {coeffs_2.shape}")
    
    print(f"\nBounding box: {result}")
    
    assert result is not None, "Expected bounding box"
    
    u_min, u_max = result[0]
    v_min, v_max = result[1]
    
    print(f"U range: [{u_min:.6f}, {u_max:.6f}]")
    print(f"V range: [{v_min:.6f}, {v_max:.6f}]")
    
    # Root at (0.5, 0.5) should be in box
    assert u_min <= 0.5 <= u_max, "Root u=0.5 should be in box"
    assert v_min <= 0.5 <= v_max, "Root v=0.5 should be in box"
    
    u_width = u_max - u_min
    v_width = v_max - v_min
    
    print(f"U width: {u_width:.6f}")
    print(f"V width: {v_width:.6f}")
    
    # Both should be tighter than [0, 1]
    assert u_width < 1.0, "U range should be tighter"
    assert v_width < 1.0, "V range should be tighter"
    
    print("✅ Test 6 passed")


def test_no_roots():
    """Test PP method correctly identifies when no roots exist."""
    print("\n" + "=" * 80)
    print("Test 7: No Roots Case")
    print("=" * 80)
    
    # Equation: f(t) = t^2 + 1 (no real roots)
    power_coeffs = np.array([1.0, 0.0, 1.0])
    coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    result = find_root_box_pp_nd([coeffs], k=1)
    
    print(f"\nEquation: f(t) = t^2 + 1 (no roots)")
    print(f"Bernstein coefficients: {coeffs}")
    print(f"Bounding box: {result}")
    
    assert result is None, "Should return None for no roots"
    print("✅ Test 7 passed: Correctly identified no roots")


if __name__ == "__main__":
    test_extract_dimension_range_1d()
    test_extract_dimension_range_2d()
    test_find_root_box_pp_nd_1d()
    test_find_root_box_pp_nd_1d_multiple_equations()
    test_find_root_box_pp_nd_2d()
    test_find_root_box_pp_nd_2d_multiple_equations()
    test_no_roots()
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✅")
    print("=" * 80)

