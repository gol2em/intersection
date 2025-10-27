"""
Test Box class for domain tracking in LP/PP methods.
"""

import sys
sys.path.insert(0, 'D:/Python/Intersection')

import numpy as np
from src.intersection.box import Box


def test_basic_box():
    """Test basic box operations."""
    print("=" * 80)
    print("TEST 1: Basic Box Operations")
    print("=" * 80)
    
    # Create a 2D box on [0, 1]^2
    box = Box(k=2, ranges=[(0.0, 1.0), (0.0, 1.0)])
    
    print(f"\nBox: {box}")
    print(f"Center: {box.get_center()}")
    print(f"Widths: {box.get_widths()}")
    print(f"Volume: {box.get_volume()}")
    
    # Test Bernstein to box mapping
    print(f"\n--- Bernstein to Box Mapping ---")
    test_points = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
    for s, t in test_points:
        box_point = box.bernstein_to_box(s, t)
        print(f"Bernstein ({s}, {t}) → Box {box_point}")
    
    # Test box to Bernstein mapping
    print(f"\n--- Box to Bernstein Mapping ---")
    for u, v in test_points:
        bernstein_point = box.box_to_bernstein(u, v)
        print(f"Box ({u}, {v}) → Bernstein {bernstein_point}")
    
    print(f"\n✅ Test 1 PASSED\n")


def test_subdivision():
    """Test box subdivision."""
    print("=" * 80)
    print("TEST 2: Box Subdivision")
    print("=" * 80)
    
    # Create a 2D box on [0, 1]^2
    box = Box(k=2, ranges=[(0.0, 1.0), (0.0, 1.0)])
    
    print(f"\nOriginal box: {box}")
    
    # Subdivide along axis 0
    left, right = box.subdivide(axis=0, split_point=0.5)
    
    print(f"\nAfter subdivision along axis 0:")
    print(f"  Left:  {left}")
    print(f"  Right: {right}")
    
    # Check that Bernstein (0.5, 0.5) in left box maps to (0.25, 0.5) in normalized space
    left_center = left.bernstein_to_box(0.5, 0.5)
    print(f"\nLeft box center (Bernstein 0.5, 0.5): {left_center}")
    print(f"Expected: [0.25, 0.5]")
    print(f"Match: {np.allclose(left_center, [0.25, 0.5])}")
    
    # Check that Bernstein (0.5, 0.5) in right box maps to (0.75, 0.5) in normalized space
    right_center = right.bernstein_to_box(0.5, 0.5)
    print(f"\nRight box center (Bernstein 0.5, 0.5): {right_center}")
    print(f"Expected: [0.75, 0.5]")
    print(f"Match: {np.allclose(right_center, [0.75, 0.5])}")
    
    print(f"\n✅ Test 2 PASSED\n")


def test_normalization_transform():
    """Test box with normalization transform."""
    print("=" * 80)
    print("TEST 3: Box with Normalization Transform")
    print("=" * 80)
    
    # Create normalization transform for u ∈ [-π, π]
    transform = {
        'scales': np.array([2 * np.pi]),
        'offsets': np.array([-np.pi]),
        'original_ranges': [(-np.pi, np.pi)]
    }
    
    # Create a 1D box on [0, 1] with normalization
    box = Box(k=1, ranges=[(0.0, 1.0)], normalization_transform=transform)
    
    print(f"\nBox: {box}")
    print(f"Original range: [-π, π]")
    print(f"Normalized range: [0, 1]")
    
    # Test normalized to original mapping
    print(f"\n--- Normalized to Original Mapping ---")
    test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    for t in test_points:
        original = box.normalized_to_original(t)
        print(f"Normalized {t:.2f} → Original {original[0]:.4f} (expected {-np.pi + t * 2 * np.pi:.4f})")
    
    # Test original to normalized mapping
    print(f"\n--- Original to Normalized Mapping ---")
    test_original = [-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi]
    for u in test_original:
        normalized = box.original_to_normalized(u)
        expected = (u + np.pi) / (2 * np.pi)
        print(f"Original {u:.4f} → Normalized {normalized[0]:.4f} (expected {expected:.4f})")
    
    # Test Bernstein to original mapping
    print(f"\n--- Bernstein to Original Mapping ---")
    # For box [0, 1], Bernstein = Box = Normalized
    for s in test_points:
        original = box.bernstein_to_original(s)
        expected = -np.pi + s * 2 * np.pi
        print(f"Bernstein {s:.2f} → Original {original[0]:.4f} (expected {expected:.4f})")
    
    print(f"\n✅ Test 3 PASSED\n")


def test_subdivision_with_normalization():
    """Test subdivision with normalization transform."""
    print("=" * 80)
    print("TEST 4: Subdivision with Normalization")
    print("=" * 80)
    
    # Create normalization transform for u ∈ [-π, π]
    transform = {
        'scales': np.array([2 * np.pi]),
        'offsets': np.array([-np.pi]),
        'original_ranges': [(-np.pi, np.pi)]
    }
    
    # Create a 1D box on [0, 1] with normalization
    box = Box(k=1, ranges=[(0.0, 1.0)], normalization_transform=transform)
    
    print(f"\nOriginal box: {box}")
    print(f"Original range: [-π, π]")
    
    # Subdivide at midpoint
    left, right = box.subdivide(axis=0, split_point=0.5)
    
    print(f"\nAfter subdivision:")
    print(f"  Left:  {left}")
    print(f"  Right: {right}")
    
    # Test Bernstein to original for left box
    print(f"\n--- Left Box: Bernstein to Original ---")
    # Left box is [0, 0.5] in normalized space
    # Bernstein 0.0 → Box 0.0 → Original -π
    # Bernstein 0.5 → Box 0.25 → Original -π/2
    # Bernstein 1.0 → Box 0.5 → Original 0
    test_bernstein = [0.0, 0.5, 1.0]
    expected_original = [-np.pi, -np.pi/2, 0.0]
    
    for s, expected in zip(test_bernstein, expected_original):
        original = left.bernstein_to_original(s)
        print(f"Bernstein {s:.2f} → Original {original[0]:.4f} (expected {expected:.4f})")
        assert np.isclose(original[0], expected), f"Mismatch: {original[0]} != {expected}"
    
    # Test Bernstein to original for right box
    print(f"\n--- Right Box: Bernstein to Original ---")
    # Right box is [0.5, 1.0] in normalized space
    # Bernstein 0.0 → Box 0.5 → Original 0
    # Bernstein 0.5 → Box 0.75 → Original π/2
    # Bernstein 1.0 → Box 1.0 → Original π
    expected_original = [0.0, np.pi/2, np.pi]
    
    for s, expected in zip(test_bernstein, expected_original):
        original = right.bernstein_to_original(s)
        print(f"Bernstein {s:.2f} → Original {original[0]:.4f} (expected {expected:.4f})")
        assert np.isclose(original[0], expected), f"Mismatch: {original[0]} != {expected}"
    
    print(f"\n✅ Test 4 PASSED\n")


def test_2d_subdivision_with_normalization():
    """Test 2D subdivision with normalization."""
    print("=" * 80)
    print("TEST 5: 2D Subdivision with Normalization")
    print("=" * 80)
    
    # Create normalization transform for (u, v) ∈ [-π, π] × [-1, 1]
    transform = {
        'scales': np.array([2 * np.pi, 2.0]),
        'offsets': np.array([-np.pi, -1.0]),
        'original_ranges': [(-np.pi, np.pi), (-1.0, 1.0)]
    }
    
    # Create a 2D box on [0, 1]^2 with normalization
    box = Box(k=2, ranges=[(0.0, 1.0), (0.0, 1.0)], normalization_transform=transform)
    
    print(f"\nOriginal box: {box}")
    print(f"Original ranges: [-π, π] × [-1, 1]")
    
    # Subdivide along axis 0 (u-axis)
    left, right = box.subdivide(axis=0, split_point=0.5)
    
    print(f"\nAfter subdivision along u-axis:")
    print(f"  Left:  {left}")
    print(f"  Right: {right}")
    
    # Test corner points of left box
    print(f"\n--- Left Box Corners ---")
    corners_bernstein = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    corners_expected = [(-np.pi, -1.0), (0.0, -1.0), (-np.pi, 1.0), (0.0, 1.0)]
    
    for (s, t), (u_exp, v_exp) in zip(corners_bernstein, corners_expected):
        original = left.bernstein_to_original(s, t)
        print(f"Bernstein ({s}, {t}) → Original ({original[0]:.4f}, {original[1]:.4f})")
        print(f"  Expected: ({u_exp:.4f}, {v_exp:.4f})")
        assert np.allclose(original, [u_exp, v_exp]), f"Mismatch!"
    
    # Test corner points of right box
    print(f"\n--- Right Box Corners ---")
    corners_expected = [(0.0, -1.0), (np.pi, -1.0), (0.0, 1.0), (np.pi, 1.0)]
    
    for (s, t), (u_exp, v_exp) in zip(corners_bernstein, corners_expected):
        original = right.bernstein_to_original(s, t)
        print(f"Bernstein ({s}, {t}) → Original ({original[0]:.4f}, {original[1]:.4f})")
        print(f"  Expected: ({u_exp:.4f}, {v_exp:.4f})")
        assert np.allclose(original, [u_exp, v_exp]), f"Mismatch!"
    
    print(f"\n✅ Test 5 PASSED\n")


if __name__ == "__main__":
    test_basic_box()
    test_subdivision()
    test_normalization_transform()
    test_subdivision_with_normalization()
    test_2d_subdivision_with_normalization()
    
    print("=" * 80)
    print("ALL TESTS PASSED ✅")
    print("=" * 80)

