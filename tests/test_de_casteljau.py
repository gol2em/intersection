"""
Test de Casteljau subdivision algorithm.
"""

import sys
sys.path.insert(0, 'D:/Python/Intersection')

import numpy as np
from scipy.special import comb
from src.intersection.de_casteljau import (
    de_casteljau_eval_1d,
    de_casteljau_subdivide_1d,
    de_casteljau_subdivide_2d,
    de_casteljau_subdivide_kd,
    subdivide_with_box
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


def test_1d_evaluation():
    """Test 1D de Casteljau evaluation."""
    print("=" * 80)
    print("TEST 1: 1D de Casteljau Evaluation")
    print("=" * 80)
    
    # Test p(t) = t
    print("\n--- p(t) = t ---")
    coeffs = np.array([0.0, 1.0])
    test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    all_match = True
    for t in test_points:
        dc_val = de_casteljau_eval_1d(coeffs, t)
        bern_val = evaluate_bernstein_1d(coeffs, t)
        expected = t
        match = np.isclose(dc_val, expected) and np.isclose(bern_val, expected)
        all_match = all_match and match
        print(f"t={t:.2f}: de_casteljau={dc_val:.6f}, bernstein={bern_val:.6f}, expected={expected:.6f}, match={match}")
    
    # Test p(t) = t^2
    print("\n--- p(t) = t^2 ---")
    coeffs = np.array([0.0, 0.0, 1.0])
    
    for t in test_points:
        dc_val = de_casteljau_eval_1d(coeffs, t)
        bern_val = evaluate_bernstein_1d(coeffs, t)
        expected = t**2
        match = np.isclose(dc_val, expected) and np.isclose(bern_val, expected)
        all_match = all_match and match
        print(f"t={t:.2f}: de_casteljau={dc_val:.6f}, bernstein={bern_val:.6f}, expected={expected:.6f}, match={match}")
    
    print(f"\n✅ Test 1: {'PASS' if all_match else 'FAIL'}\n")
    return all_match


def test_1d_subdivision():
    """Test 1D de Casteljau subdivision."""
    print("=" * 80)
    print("TEST 2: 1D de Casteljau Subdivision")
    print("=" * 80)
    
    # Test p(t) = t, subdivide at 0.5
    print("\n--- p(t) = t, subdivide at t=0.5 ---")
    coeffs = np.array([0.0, 1.0])
    left, right = de_casteljau_subdivide_1d(coeffs, 0.5, verbose=True)
    
    print(f"\nExpected:")
    print(f"  Left:  [0, 0.5] (p on [0, 0.5] renormalized to [0, 1])")
    print(f"  Right: [0.5, 1] (p on [0.5, 1] renormalized to [0, 1])")
    
    # Verify: left polynomial on [0,1] should give values from [0, 0.5]
    # right polynomial on [0,1] should give values from [0.5, 1]
    print(f"\nVerification:")
    test_points = [0.0, 0.5, 1.0]
    all_match = True
    
    for s in test_points:
        left_val = de_casteljau_eval_1d(left, s)
        expected_left = 0.5 * s  # Map [0,1] → [0, 0.5]
        match_left = np.isclose(left_val, expected_left)
        
        right_val = de_casteljau_eval_1d(right, s)
        expected_right = 0.5 + 0.5 * s  # Map [0,1] → [0.5, 1]
        match_right = np.isclose(right_val, expected_right)
        
        all_match = all_match and match_left and match_right
        print(f"s={s:.2f}: left={left_val:.6f} (expected {expected_left:.6f}, {match_left}), "
              f"right={right_val:.6f} (expected {expected_right:.6f}, {match_right})")
    
    # Test p(t) = t^2, subdivide at 0.5
    print("\n--- p(t) = t^2, subdivide at t=0.5 ---")
    coeffs = np.array([0.0, 0.0, 1.0])
    left, right = de_casteljau_subdivide_1d(coeffs, 0.5, verbose=False)
    
    print(f"Left coefficients:  {left}")
    print(f"Right coefficients: {right}")
    
    # Verify continuity at subdivision point
    left_at_1 = de_casteljau_eval_1d(left, 1.0)
    right_at_0 = de_casteljau_eval_1d(right, 0.0)
    parent_at_05 = de_casteljau_eval_1d(coeffs, 0.5)
    
    print(f"\nContinuity check:")
    print(f"  Parent at 0.5: {parent_at_05:.6f}")
    print(f"  Left at 1.0:   {left_at_1:.6f}")
    print(f"  Right at 0.0:  {right_at_0:.6f}")
    continuity = np.allclose([left_at_1, right_at_0], parent_at_05)
    print(f"  Match: {continuity}")
    
    all_match = all_match and continuity
    
    print(f"\n✅ Test 2: {'PASS' if all_match else 'FAIL'}\n")
    return all_match


def test_2d_subdivision():
    """Test 2D de Casteljau subdivision."""
    print("=" * 80)
    print("TEST 3: 2D de Casteljau Subdivision")
    print("=" * 80)
    
    # Test p(u,v) = u, subdivide along u-axis
    print("\n--- p(u,v) = u, subdivide along u-axis at t=0.5 ---")
    # Bernstein coefficients for u with degree (1, 1)
    coeffs = np.array([[0.0, 0.0],
                       [1.0, 1.0]])
    
    left, right = de_casteljau_subdivide_2d(coeffs, axis=0, t=0.5, verbose=True)
    
    # Verify: left should give u ∈ [0, 0.5], right should give u ∈ [0.5, 1]
    print(f"\nVerification:")
    test_points = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
    all_match = True
    
    for s, t in test_points:
        left_val = evaluate_bernstein_2d(left, s, t)
        expected_left = 0.5 * s  # u ∈ [0, 0.5]
        match_left = np.isclose(left_val, expected_left)
        
        right_val = evaluate_bernstein_2d(right, s, t)
        expected_right = 0.5 + 0.5 * s  # u ∈ [0.5, 1]
        match_right = np.isclose(right_val, expected_right)
        
        all_match = all_match and match_left and match_right
        print(f"(s,t)=({s:.2f},{t:.2f}): left={left_val:.6f} (expected {expected_left:.6f}, {match_left}), "
              f"right={right_val:.6f} (expected {expected_right:.6f}, {match_right})")
    
    # Test p(u,v) = v, subdivide along v-axis
    print("\n--- p(u,v) = v, subdivide along v-axis at t=0.5 ---")
    coeffs = np.array([[0.0, 1.0],
                       [0.0, 1.0]])
    
    left, right = de_casteljau_subdivide_2d(coeffs, axis=1, t=0.5, verbose=False)
    
    print(f"Left coefficients:\n{left}")
    print(f"Right coefficients:\n{right}")
    
    # Verify
    for s, t in test_points:
        left_val = evaluate_bernstein_2d(left, s, t)
        expected_left = 0.5 * t  # v ∈ [0, 0.5]
        match_left = np.isclose(left_val, expected_left)
        
        right_val = evaluate_bernstein_2d(right, s, t)
        expected_right = 0.5 + 0.5 * t  # v ∈ [0.5, 1]
        match_right = np.isclose(right_val, expected_right)
        
        all_match = all_match and match_left and match_right
        print(f"(s,t)=({s:.2f},{t:.2f}): left={left_val:.6f} (expected {expected_left:.6f}, {match_left}), "
              f"right={right_val:.6f} (expected {expected_right:.6f}, {match_right})")
    
    print(f"\n✅ Test 3: {'PASS' if all_match else 'FAIL'}\n")
    return all_match


def test_subdivision_with_box():
    """Test subdivision with Box tracking."""
    print("=" * 80)
    print("TEST 4: Subdivision with Box Tracking")
    print("=" * 80)
    
    # 1D example
    print("\n--- 1D: p(t) = t on [0, 1] ---")
    coeffs = np.array([0.0, 1.0])
    box = Box(k=1, ranges=[(0.0, 1.0)])
    
    left_c, left_b, right_c, right_b = subdivide_with_box(coeffs, box, axis=0, t=0.5, verbose=True)
    
    print(f"\nVerification:")
    # Check that boxes are correct
    print(f"Left box ranges:  {left_b.ranges} (expected [(0.0, 0.5)])")
    print(f"Right box ranges: {right_b.ranges} (expected [(0.5, 1.0)])")
    
    box_match = (left_b.ranges == [(0.0, 0.5)] and right_b.ranges == [(0.5, 1.0)])
    
    # Check that coefficients are correct
    print(f"Left coefficients:  {left_c} (expected [0, 0.5])")
    print(f"Right coefficients: {right_c} (expected [0.5, 1])")
    
    coeff_match = (np.allclose(left_c, [0, 0.5]) and np.allclose(right_c, [0.5, 1]))
    
    # 2D example
    print("\n--- 2D: p(u,v) = u on [0, 1]^2, subdivide along u ---")
    coeffs = np.array([[0.0, 0.0], [1.0, 1.0]])
    box = Box(k=2, ranges=[(0.0, 1.0), (0.0, 1.0)])
    
    left_c, left_b, right_c, right_b = subdivide_with_box(coeffs, box, axis=0, t=0.5, verbose=True)
    
    print(f"\nVerification:")
    print(f"Left box ranges:  {left_b.ranges} (expected [(0.0, 0.5), (0.0, 1.0)])")
    print(f"Right box ranges: {right_b.ranges} (expected [(0.5, 1.0), (0.0, 1.0)])")
    
    box_match_2d = (left_b.ranges == [(0.0, 0.5), (0.0, 1.0)] and 
                    right_b.ranges == [(0.5, 1.0), (0.0, 1.0)])
    
    all_match = box_match and coeff_match and box_match_2d
    
    print(f"\n✅ Test 4: {'PASS' if all_match else 'FAIL'}\n")
    return all_match


def test_subdivision_with_normalization():
    """Test subdivision with normalization transform."""
    print("=" * 80)
    print("TEST 5: Subdivision with Normalization Transform")
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
    
    # Subdivide
    left_c, left_b, right_c, right_b = subdivide_with_box(coeffs, box, axis=0, t=0.5, verbose=True)
    
    print(f"\nVerification:")
    
    # Check that left box maps correctly
    # Bernstein 0.5 in left box → Box 0.25 → Normalized 0.25 → Original -π/2
    t_bern = 0.5
    u_original = left_b.bernstein_to_original(t_bern)[0]
    expected_u = -np.pi / 2
    
    print(f"Left box: Bernstein {t_bern} → Original {u_original:.6f} (expected {expected_u:.6f})")
    left_match = np.isclose(u_original, expected_u)
    
    # Check that right box maps correctly
    u_original = right_b.bernstein_to_original(t_bern)[0]
    expected_u = np.pi / 2
    
    print(f"Right box: Bernstein {t_bern} → Original {u_original:.6f} (expected {expected_u:.6f})")
    right_match = np.isclose(u_original, expected_u)
    
    # Check that coefficients are correct
    print(f"\nLeft coefficients:  {left_c} (expected [0, 0.5])")
    print(f"Right coefficients: {right_c} (expected [0.5, 1])")
    coeff_match = np.allclose(left_c, [0, 0.5]) and np.allclose(right_c, [0.5, 1])
    
    all_match = left_match and right_match and coeff_match
    
    print(f"\n✅ Test 5: {'PASS' if all_match else 'FAIL'}\n")
    return all_match


if __name__ == "__main__":
    test1 = test_1d_evaluation()
    test2 = test_1d_subdivision()
    test3 = test_2d_subdivision()
    test4 = test_subdivision_with_box()
    test5 = test_subdivision_with_normalization()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Test 1 (1D Evaluation):              {'PASS' if test1 else 'FAIL'}")
    print(f"Test 2 (1D Subdivision):             {'PASS' if test2 else 'FAIL'}")
    print(f"Test 3 (2D Subdivision):             {'PASS' if test3 else 'FAIL'}")
    print(f"Test 4 (Subdivision with Box):       {'PASS' if test4 else 'FAIL'}")
    print(f"Test 5 (Subdivision with Transform): {'PASS' if test5 else 'FAIL'}")
    
    all_pass = test1 and test2 and test3 and test4 and test5
    print(f"\nOverall: {'ALL TESTS PASSED ✅' if all_pass else 'SOME TESTS FAILED ✗'}")

