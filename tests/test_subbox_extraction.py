"""
Test sub-box extraction for de Casteljau subdivision.
"""

import sys
sys.path.insert(0, 'D:/Python/Intersection')

import numpy as np
from scipy.special import comb
from src.intersection.de_casteljau import (
    extract_subbox_1d,
    extract_subbox_2d,
    extract_subbox_kd,
    extract_subbox_with_box
)
from src.intersection.box import Box


def evaluate_bernstein_1d(coeffs, t):
    """Evaluate 1D Bernstein polynomial."""
    n = len(coeffs) - 1
    result = 0.0
    for i, b_i in enumerate(coeffs):
        basis = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        result += b_i * basis
    return result


def evaluate_bernstein_2d(coeffs, u, v):
    """Evaluate 2D Bernstein polynomial."""
    n_u, n_v = coeffs.shape
    degree_u = n_u - 1
    degree_v = n_v - 1
    
    result = 0.0
    for i in range(n_u):
        for j in range(n_v):
            basis_u = comb(degree_u, i) * (u ** i) * ((1 - u) ** (degree_u - i))
            basis_v = comb(degree_v, j) * (v ** j) * ((1 - v) ** (degree_v - j))
            result += coeffs[i, j] * basis_u * basis_v
    
    return result


def test_1d_subbox_extraction():
    """Test 1D sub-box extraction."""
    print("=" * 80)
    print("TEST 1: 1D Sub-box Extraction")
    print("=" * 80)
    
    # Test p(t) = t on [0.25, 0.75]
    print("\n--- p(t) = t on [0.25, 0.75] ---")
    coeffs = np.array([0.0, 1.0])
    sub_coeffs = extract_subbox_1d(coeffs, 0.25, 0.75, verbose=True)
    
    print(f"\nExpected: [0.25, 0.75]")
    print(f"Got:      {sub_coeffs}")
    
    # Verify: sub_coeffs on [0,1] should give values from [0.25, 0.75]
    print(f"\nVerification:")
    test_points = [0.0, 0.5, 1.0]
    all_match = True
    
    for s in test_points:
        sub_val = evaluate_bernstein_1d(sub_coeffs, s)
        expected = 0.25 + 0.5 * s  # Map [0,1] → [0.25, 0.75]
        match = np.isclose(sub_val, expected)
        all_match = all_match and match
        print(f"s={s:.2f}: sub_val={sub_val:.6f}, expected={expected:.6f}, match={match}")
    
    # Test p(t) = t^2 on [0.5, 1.0]
    print("\n--- p(t) = t^2 on [0.5, 1.0] ---")
    coeffs = np.array([0.0, 0.0, 1.0])
    sub_coeffs = extract_subbox_1d(coeffs, 0.5, 1.0, verbose=False)
    
    print(f"Sub-box coefficients: {sub_coeffs}")
    
    # Verify
    for s in test_points:
        sub_val = evaluate_bernstein_1d(sub_coeffs, s)
        # Original: t^2, on [0.5, 1.0] mapped to [0, 1]
        t_original = 0.5 + 0.5 * s
        expected = t_original ** 2
        match = np.isclose(sub_val, expected)
        all_match = all_match and match
        print(f"s={s:.2f}: sub_val={sub_val:.6f}, expected={expected:.6f} (t={t_original:.2f}), match={match}")
    
    print(f"\n✅ Test 1: {'PASS' if all_match else 'FAIL'}\n")
    return all_match


def test_2d_subbox_extraction():
    """Test 2D sub-box extraction."""
    print("=" * 80)
    print("TEST 2: 2D Sub-box Extraction")
    print("=" * 80)
    
    # Test p(u,v) = u on [0.25, 0.75] × [0, 1]
    print("\n--- p(u,v) = u on [0.25, 0.75] × [0, 1] ---")
    coeffs = np.array([[0.0, 0.0], [1.0, 1.0]])
    sub_coeffs = extract_subbox_2d(coeffs, [(0.25, 0.75), (0.0, 1.0)], verbose=True)
    
    print(f"\nExpected: u ∈ [0.25, 0.75], constant in v")
    
    # Verify
    test_points = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
    all_match = True
    
    for s, t in test_points:
        sub_val = evaluate_bernstein_2d(sub_coeffs, s, t)
        expected = 0.25 + 0.5 * s  # u ∈ [0.25, 0.75]
        match = np.isclose(sub_val, expected)
        all_match = all_match and match
        print(f"(s,t)=({s:.2f},{t:.2f}): sub_val={sub_val:.6f}, expected={expected:.6f}, match={match}")
    
    # Test p(u,v) = u+v on [0.25, 0.75] × [0.5, 1.0]
    print("\n--- p(u,v) = u+v on [0.25, 0.75] × [0.5, 1.0] ---")
    # Bernstein coefficients for u+v with degree (1, 1)
    coeffs = np.array([[0.0, 1.0], [1.0, 2.0]])
    sub_coeffs = extract_subbox_2d(coeffs, [(0.25, 0.75), (0.5, 1.0)], verbose=False)
    
    print(f"Sub-box coefficients:\n{sub_coeffs}")
    
    # Verify
    for s, t in test_points:
        sub_val = evaluate_bernstein_2d(sub_coeffs, s, t)
        u_original = 0.25 + 0.5 * s
        v_original = 0.5 + 0.5 * t
        expected = u_original + v_original
        match = np.isclose(sub_val, expected)
        all_match = all_match and match
        print(f"(s,t)=({s:.2f},{t:.2f}): sub_val={sub_val:.6f}, expected={expected:.6f}, match={match}")
    
    print(f"\n✅ Test 2: {'PASS' if all_match else 'FAIL'}\n")
    return all_match


def test_subbox_with_box():
    """Test sub-box extraction with Box tracking."""
    print("=" * 80)
    print("TEST 3: Sub-box Extraction with Box Tracking")
    print("=" * 80)
    
    # 1D example
    print("\n--- 1D: p(t) = t on [0.25, 0.75] ---")
    coeffs = np.array([0.0, 1.0])
    box = Box(k=1, ranges=[(0.0, 1.0)])
    
    sub_coeffs, sub_box = extract_subbox_with_box(coeffs, box, [(0.25, 0.75)], verbose=True)
    
    print(f"\nVerification:")
    print(f"Sub-box ranges: {sub_box.ranges} (expected [(0.25, 0.75)])")
    print(f"Sub-coefficients: {sub_coeffs} (expected [0.25, 0.75])")
    
    box_match = (sub_box.ranges == [(0.25, 0.75)])
    coeff_match = np.allclose(sub_coeffs, [0.25, 0.75])
    
    # 2D example
    print("\n--- 2D: p(u,v) = u on [0.25, 0.75] × [0, 1] ---")
    coeffs = np.array([[0.0, 0.0], [1.0, 1.0]])
    box = Box(k=2, ranges=[(0.0, 1.0), (0.0, 1.0)])
    
    sub_coeffs, sub_box = extract_subbox_with_box(coeffs, box, [(0.25, 0.75), (0.0, 1.0)], verbose=True)
    
    print(f"\nVerification:")
    print(f"Sub-box ranges: {sub_box.ranges} (expected [(0.25, 0.75), (0.0, 1.0)])")
    
    box_match_2d = (sub_box.ranges == [(0.25, 0.75), (0.0, 1.0)])
    
    all_match = box_match and coeff_match and box_match_2d
    
    print(f"\n✅ Test 3: {'PASS' if all_match else 'FAIL'}\n")
    return all_match


def test_subbox_with_normalization():
    """Test sub-box extraction with normalization transform."""
    print("=" * 80)
    print("TEST 4: Sub-box Extraction with Normalization")
    print("=" * 80)
    
    # Create normalization transform for u ∈ [-π, π]
    transform = {
        'scales': np.array([2 * np.pi]),
        'offsets': np.array([-np.pi]),
        'original_ranges': [(-np.pi, np.pi)]
    }
    
    # Create box with normalization
    box = Box(k=1, ranges=[(0.0, 1.0)], normalization_transform=transform)
    
    # Coefficients for identity: t on [0, 1]
    coeffs = np.array([0.0, 1.0])
    
    print(f"\nOriginal box: {box}")
    print(f"Original range: [-π, π]")
    print(f"Normalized range: [0, 1]")
    
    # Extract sub-box [0.25, 0.75] in normalized space
    # This corresponds to [-π/2, π/2] in original space
    sub_coeffs, sub_box = extract_subbox_with_box(coeffs, box, [(0.25, 0.75)], verbose=True)
    
    print(f"\nVerification:")
    
    # Check that sub-box maps correctly
    # Bernstein 0.0 in sub-box → Box 0.25 → Normalized 0.25 → Original -π/2
    t_bern = 0.0
    u_original = sub_box.bernstein_to_original(t_bern)[0]
    expected_u = -np.pi / 2
    
    print(f"Sub-box: Bernstein {t_bern} → Original {u_original:.6f} (expected {expected_u:.6f})")
    match_0 = np.isclose(u_original, expected_u)
    
    # Bernstein 1.0 in sub-box → Box 0.75 → Normalized 0.75 → Original π/2
    t_bern = 1.0
    u_original = sub_box.bernstein_to_original(t_bern)[0]
    expected_u = np.pi / 2
    
    print(f"Sub-box: Bernstein {t_bern} → Original {u_original:.6f} (expected {expected_u:.6f})")
    match_1 = np.isclose(u_original, expected_u)
    
    # Check coefficients
    print(f"\nSub-coefficients: {sub_coeffs} (expected [0.25, 0.75])")
    coeff_match = np.allclose(sub_coeffs, [0.25, 0.75])
    
    all_match = match_0 and match_1 and coeff_match
    
    print(f"\n✅ Test 4: {'PASS' if all_match else 'FAIL'}\n")
    return all_match


def test_nested_subboxes():
    """Test extracting sub-box from a sub-box."""
    print("=" * 80)
    print("TEST 5: Nested Sub-box Extraction")
    print("=" * 80)
    
    # Start with p(t) = t on [0, 1]
    print("\n--- Extract [0.25, 0.75], then [0.5, 1.0] from that ---")
    coeffs = np.array([0.0, 1.0])
    box = Box(k=1, ranges=[(0.0, 1.0)])
    
    # First extraction: [0.25, 0.75]
    sub1_coeffs, sub1_box = extract_subbox_with_box(coeffs, box, [(0.25, 0.75)], verbose=False)
    print(f"First sub-box: {sub1_box}")
    print(f"First sub-coefficients: {sub1_coeffs}")
    
    # Second extraction: [0.5, 1.0] relative to sub1_box
    # This should give [0.5, 0.75] in original space
    sub2_coeffs, sub2_box = extract_subbox_with_box(sub1_coeffs, sub1_box, [(0.5, 1.0)], verbose=False)
    print(f"\nSecond sub-box: {sub2_box}")
    print(f"Second sub-coefficients: {sub2_coeffs}")
    
    # Verify
    print(f"\nVerification:")
    print(f"Expected ranges: [(0.5, 0.75)] (0.25 + 0.5*0.5 = 0.5, 0.25 + 0.5*1.0 = 0.75)")
    print(f"Got ranges:      {sub2_box.ranges}")
    
    box_match = np.allclose(sub2_box.ranges[0], (0.5, 0.75))
    
    # Expected coefficients: [0.5, 0.75]
    print(f"Expected coefficients: [0.5, 0.75]")
    print(f"Got coefficients:      {sub2_coeffs}")
    
    coeff_match = np.allclose(sub2_coeffs, [0.5, 0.75])
    
    all_match = box_match and coeff_match
    
    print(f"\n✅ Test 5: {'PASS' if all_match else 'FAIL'}\n")
    return all_match


if __name__ == "__main__":
    test1 = test_1d_subbox_extraction()
    test2 = test_2d_subbox_extraction()
    test3 = test_subbox_with_box()
    test4 = test_subbox_with_normalization()
    test5 = test_nested_subboxes()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Test 1 (1D Sub-box):              {'PASS' if test1 else 'FAIL'}")
    print(f"Test 2 (2D Sub-box):              {'PASS' if test2 else 'FAIL'}")
    print(f"Test 3 (Sub-box with Box):        {'PASS' if test3 else 'FAIL'}")
    print(f"Test 4 (Sub-box with Transform):  {'PASS' if test4 else 'FAIL'}")
    print(f"Test 5 (Nested Sub-boxes):        {'PASS' if test5 else 'FAIL'}")
    
    all_pass = test1 and test2 and test3 and test4 and test5
    print(f"\nOverall: {'ALL TESTS PASSED ✅' if all_pass else 'SOME TESTS FAILED ✗'}")

