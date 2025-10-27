"""
Diagnostic: Test power-to-Bernstein conversion.
"""

import sys
sys.path.insert(0, 'D:/Python/Intersection')

import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import comb
from src.intersection.bernstein import _polynomial_1d_to_bernstein


def evaluate_bernstein_1d(coeffs, t):
    """Evaluate 1D Bernstein polynomial."""
    n = len(coeffs) - 1
    result = 0.0
    for i, b_i in enumerate(coeffs):
        basis = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        result += b_i * basis
    return result


print("=" * 80)
print("DIAGNOSTIC: Power to Bernstein Conversion")
print("=" * 80)

# Test 1: Simple polynomial p(t) = t
print("\n--- Test 1: p(t) = t ---")
poly = Polynomial([0, 1])  # 0 + 1*t
print(f"Power coefficients: {poly.coef}")

bern = _polynomial_1d_to_bernstein(poly, verbose=True)
print(f"Bernstein coefficients: {bern}")

# Expected: [0, 1] for degree 1
# Or if elevated to higher degree: [0, 0.5, 1] for degree 2, etc.
print(f"Expected (degree 1): [0, 1]")

# Test evaluation
test_points = [0.0, 0.5, 1.0]
print(f"\nEvaluation test:")
for t in test_points:
    power_val = poly(t)
    bern_val = evaluate_bernstein_1d(bern, t)
    print(f"t={t}: power={power_val:.6f}, bernstein={bern_val:.6f}, match={np.isclose(power_val, bern_val)}")

# Test 2: Polynomial p(t) = t^2
print("\n\n--- Test 2: p(t) = t^2 ---")
poly = Polynomial([0, 0, 1])  # 0 + 0*t + 1*t^2
print(f"Power coefficients: {poly.coef}")

bern = _polynomial_1d_to_bernstein(poly, verbose=True)
print(f"Bernstein coefficients: {bern}")

# Expected: [0, 0.5, 1] for degree 2
print(f"Expected (degree 2): [0, 0.5, 1]")

# Test evaluation
print(f"\nEvaluation test:")
for t in test_points:
    power_val = poly(t)
    bern_val = evaluate_bernstein_1d(bern, t)
    print(f"t={t}: power={power_val:.6f}, bernstein={bern_val:.6f}, match={np.isclose(power_val, bern_val)}")

# Test 3: Polynomial p(t) = 1 - 2t + t^2 = (1-t)^2
print("\n\n--- Test 3: p(t) = (1-t)^2 = 1 - 2t + t^2 ---")
poly = Polynomial([1, -2, 1])
print(f"Power coefficients: {poly.coef}")

bern = _polynomial_1d_to_bernstein(poly, verbose=True)
print(f"Bernstein coefficients: {bern}")

# Expected: [1, 0, 0] for degree 2 (since (1-t)^2 = B_0^2(t))
print(f"Expected (degree 2): [1, 0, 0]")

# Test evaluation
print(f"\nEvaluation test:")
for t in test_points:
    power_val = poly(t)
    bern_val = evaluate_bernstein_1d(bern, t)
    print(f"t={t}: power={power_val:.6f}, bernstein={bern_val:.6f}, match={np.isclose(power_val, bern_val)}")

