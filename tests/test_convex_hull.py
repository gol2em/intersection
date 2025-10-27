"""
Test convex hull utilities for PP method.
"""

import numpy as np
from src.intersection.convex_hull import (
    convex_hull_2d,
    intersect_convex_hull_with_x_axis,
    find_root_box_pp_1d
)


def test_convex_hull_2d():
    """Test 2D convex hull computation."""
    print("\n" + "=" * 80)
    print("Test 1: Convex Hull 2D")
    print("=" * 80)
    
    # Test 1: Square with interior point
    points = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0.5, 0.5]  # Interior point
    ])
    
    hull = convex_hull_2d(points)
    print(f"\nInput points:\n{points}")
    print(f"\nConvex hull vertices:\n{hull}")
    print(f"Number of hull vertices: {len(hull)}")
    
    # Should have 4 vertices (the square corners)
    assert len(hull) == 4, f"Expected 4 vertices, got {len(hull)}"
    print("✅ Test 1 passed: Square hull has 4 vertices")
    
    # Test 2: Triangle
    points = np.array([
        [0, 0],
        [1, 0],
        [0.5, 1]
    ])
    
    hull = convex_hull_2d(points)
    print(f"\nTriangle hull vertices: {len(hull)}")
    assert len(hull) == 3, f"Expected 3 vertices, got {len(hull)}"
    print("✅ Test 2 passed: Triangle hull has 3 vertices")
    
    # Test 3: Collinear points
    points = np.array([
        [0, 0],
        [0.5, 0.5],
        [1, 1]
    ])
    
    hull = convex_hull_2d(points)
    print(f"\nCollinear points hull vertices: {len(hull)}")
    # Collinear points should give 2 vertices (endpoints)
    assert len(hull) == 2, f"Expected 2 vertices for collinear points, got {len(hull)}"
    print("✅ Test 3 passed: Collinear points give 2 vertices")


def test_intersect_x_axis_basic():
    """Test basic intersection with x-axis."""
    print("\n" + "=" * 80)
    print("Test 2: Intersection with X-axis - Basic Cases")
    print("=" * 80)
    
    # Test 1: Triangle crossing x-axis
    points = np.array([
        [0, 1],
        [0.5, -1],
        [1, 1]
    ])
    
    result = intersect_convex_hull_with_x_axis(points)
    print(f"\nTest 1: Triangle crossing x-axis")
    print(f"Points:\n{points}")
    print(f"Intersection: {result}")
    
    assert result is not None, "Expected intersection"
    x_min, x_max = result
    print(f"X-range: [{x_min:.6f}, {x_max:.6f}]")
    
    # The triangle crosses x-axis between 0.25 and 0.75
    assert 0.2 < x_min < 0.3, f"Expected x_min around 0.25, got {x_min}"
    assert 0.7 < x_max < 0.8, f"Expected x_max around 0.75, got {x_max}"
    print("✅ Test 1 passed: Triangle intersection correct")
    
    # Test 2: All points above x-axis (no intersection)
    points = np.array([
        [0, 1],
        [0.5, 2],
        [1, 1]
    ])
    
    result = intersect_convex_hull_with_x_axis(points)
    print(f"\nTest 2: All points above x-axis")
    print(f"Intersection: {result}")
    
    assert result is None, "Expected no intersection"
    print("✅ Test 2 passed: No intersection when all points above")
    
    # Test 3: All points below x-axis (no intersection)
    points = np.array([
        [0, -1],
        [0.5, -2],
        [1, -1]
    ])
    
    result = intersect_convex_hull_with_x_axis(points)
    print(f"\nTest 3: All points below x-axis")
    print(f"Intersection: {result}")
    
    assert result is None, "Expected no intersection"
    print("✅ Test 3 passed: No intersection when all points below")
    
    # Test 4: Point on x-axis
    points = np.array([
        [0, 1],
        [0.5, 0],
        [1, 1]
    ])
    
    result = intersect_convex_hull_with_x_axis(points)
    print(f"\nTest 4: Point on x-axis")
    print(f"Intersection: {result}")
    
    assert result is not None, "Expected intersection"
    x_min, x_max = result
    print(f"X-range: [{x_min:.6f}, {x_max:.6f}]")
    assert abs(x_min - 0.5) < 1e-6, f"Expected x_min = 0.5, got {x_min}"
    assert abs(x_max - 0.5) < 1e-6, f"Expected x_max = 0.5, got {x_max}"
    print("✅ Test 4 passed: Single point intersection")


def test_intersect_x_axis_bernstein():
    """Test intersection with Bernstein control points."""
    print("\n" + "=" * 80)
    print("Test 3: Intersection with Bernstein Control Points")
    print("=" * 80)
    
    # Test 1: Polynomial with sign change
    # f(t) = t - 0.5, Bernstein coeffs: [-0.5, 0.5]
    coeffs = np.array([-0.5, 0.5])
    t_values = np.linspace(0, 1, len(coeffs))
    points = np.column_stack([t_values, coeffs])
    
    result = intersect_convex_hull_with_x_axis(points)
    print(f"\nTest 1: f(t) = t - 0.5")
    print(f"Bernstein coefficients: {coeffs}")
    print(f"Intersection: {result}")
    
    assert result is not None, "Expected intersection"
    x_min, x_max = result
    print(f"Root must be in [{x_min:.6f}, {x_max:.6f}]")
    
    # The actual root is at t = 0.5
    # Convex hull should give tight bounds
    assert x_min <= 0.5 <= x_max, "Root should be in the range"
    print("✅ Test 1 passed: Linear polynomial")
    
    # Test 2: Quadratic with two roots
    # f(t) = (t - 0.3)(t - 0.7) = t^2 - t + 0.21
    # Bernstein form (degree 2): [0.21, -0.095, 0.21]
    from src.intersection.bernstein import polynomial_nd_to_bernstein
    power_coeffs = np.array([0.21, -1.0, 1.0])
    coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    t_values = np.linspace(0, 1, len(coeffs))
    points = np.column_stack([t_values, coeffs])
    
    result = intersect_convex_hull_with_x_axis(points)
    print(f"\nTest 2: f(t) = (t - 0.3)(t - 0.7)")
    print(f"Bernstein coefficients: {coeffs}")
    print(f"Intersection: {result}")
    
    assert result is not None, "Expected intersection"
    x_min, x_max = result
    print(f"Roots must be in [{x_min:.6f}, {x_max:.6f}]")
    
    # Roots are at 0.3 and 0.7
    assert x_min <= 0.3, f"x_min should be <= 0.3, got {x_min}"
    assert x_max >= 0.7, f"x_max should be >= 0.7, got {x_max}"
    print("✅ Test 2 passed: Quadratic with two roots")
    
    # Test 3: Polynomial with no roots (all positive)
    # f(t) = t^2 + 1
    power_coeffs = np.array([1.0, 0.0, 1.0])
    coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    t_values = np.linspace(0, 1, len(coeffs))
    points = np.column_stack([t_values, coeffs])
    
    result = intersect_convex_hull_with_x_axis(points)
    print(f"\nTest 3: f(t) = t^2 + 1 (no roots)")
    print(f"Bernstein coefficients: {coeffs}")
    print(f"Intersection: {result}")
    
    assert result is None, "Expected no intersection"
    print("✅ Test 3 passed: No roots detected")


def test_find_root_box_pp_1d():
    """Test PP method for finding root box in 1D."""
    print("\n" + "=" * 80)
    print("Test 4: Find Root Box using PP Method (1D)")
    print("=" * 80)
    
    # Test 1: Linear polynomial
    from src.intersection.bernstein import polynomial_nd_to_bernstein
    
    power_coeffs = np.array([-0.5, 1.0])  # f(t) = t - 0.5
    coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    result = find_root_box_pp_1d(coeffs)
    print(f"\nTest 1: f(t) = t - 0.5")
    print(f"Bernstein coefficients: {coeffs}")
    print(f"Root box: {result}")
    
    assert result is not None, "Expected root box"
    t_min, t_max = result
    print(f"Root must be in [{t_min:.6f}, {t_max:.6f}]")
    assert t_min <= 0.5 <= t_max, "Root at 0.5 should be in range"
    print("✅ Test 1 passed")
    
    # Test 2: Quadratic with two roots
    power_coeffs = np.array([0.21, -1.0, 1.0])  # f(t) = (t-0.3)(t-0.7)
    coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    result = find_root_box_pp_1d(coeffs)
    print(f"\nTest 2: f(t) = (t - 0.3)(t - 0.7)")
    print(f"Bernstein coefficients: {coeffs}")
    print(f"Root box: {result}")
    
    assert result is not None, "Expected root box"
    t_min, t_max = result
    print(f"Roots must be in [{t_min:.6f}, {t_max:.6f}]")
    assert t_min <= 0.3, f"t_min should be <= 0.3, got {t_min}"
    assert t_max >= 0.7, f"t_max should be >= 0.7, got {t_max}"
    print("✅ Test 2 passed")
    
    # Test 3: No roots
    power_coeffs = np.array([1.0, 0.0, 1.0])  # f(t) = t^2 + 1
    coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    result = find_root_box_pp_1d(coeffs)
    print(f"\nTest 3: f(t) = t^2 + 1 (no roots)")
    print(f"Bernstein coefficients: {coeffs}")
    print(f"Root box: {result}")
    
    assert result is None, "Expected no root box"
    print("✅ Test 3 passed")
    
    # Test 4: High degree polynomial
    # f(t) = (t - 0.5)^5
    power_coeffs = np.array([
        -0.03125, 0.15625, -0.3125, 0.3125, -0.15625, 0.03125
    ])
    coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    result = find_root_box_pp_1d(coeffs)
    print(f"\nTest 4: f(t) = (t - 0.5)^5")
    print(f"Bernstein coefficients: {coeffs}")
    print(f"Root box: {result}")
    
    assert result is not None, "Expected root box"
    t_min, t_max = result
    print(f"Root must be in [{t_min:.6f}, {t_max:.6f}]")
    assert t_min <= 0.5 <= t_max, "Root at 0.5 should be in range"
    
    # Check that bounds are reasonably tight
    width = t_max - t_min
    print(f"Box width: {width:.6f}")
    assert width < 1.0, "Box should be tighter than [0, 1]"
    print("✅ Test 4 passed")


def test_comparison_with_simple_bounds():
    """Compare convex hull method with simple min/max bounds."""
    print("\n" + "=" * 80)
    print("Test 5: Comparison with Simple Min/Max Bounds")
    print("=" * 80)
    
    from src.intersection.bernstein import polynomial_nd_to_bernstein
    
    # Polynomial with roots at 0.3 and 0.7
    power_coeffs = np.array([0.21, -1.0, 1.0])
    coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    # Simple method: just check if 0 is in [min, max]
    simple_can_have_root = np.min(coeffs) <= 0 <= np.max(coeffs)
    simple_box = (0.0, 1.0) if simple_can_have_root else None
    
    # Convex hull method
    ch_box = find_root_box_pp_1d(coeffs)
    
    print(f"\nPolynomial: f(t) = (t - 0.3)(t - 0.7)")
    print(f"Bernstein coefficients: {coeffs}")
    print(f"Min coefficient: {np.min(coeffs):.6f}")
    print(f"Max coefficient: {np.max(coeffs):.6f}")
    print(f"\nSimple method (min/max): {simple_box}")
    print(f"Convex hull method: {ch_box}")
    
    if ch_box:
        t_min, t_max = ch_box
        width = t_max - t_min
        print(f"\nConvex hull box width: {width:.6f}")
        print(f"Simple box width: 1.0")
        print(f"Improvement: {(1.0 - width) / 1.0 * 100:.1f}% tighter")
        
        assert width < 1.0, "Convex hull should give tighter bounds"
        print("✅ Convex hull method gives tighter bounds!")


if __name__ == "__main__":
    test_convex_hull_2d()
    test_intersect_x_axis_basic()
    test_intersect_x_axis_bernstein()
    test_find_root_box_pp_1d()
    test_comparison_with_simple_bounds()
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✅")
    print("=" * 80)

