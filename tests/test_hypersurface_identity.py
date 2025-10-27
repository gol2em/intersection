"""
Diagnostic test: Check if hypersurface correctly represents identity function.
"""

import sys
sys.path.insert(0, 'D:/Python/Intersection')

import numpy as np
from scipy.special import comb
from src.intersection.geometry import Hypersurface


def evaluate_bernstein_1d(coeffs, t):
    """Evaluate 1D Bernstein polynomial."""
    n = len(coeffs) - 1
    result = 0.0
    for i, b_i in enumerate(coeffs):
        basis = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        result += b_i * basis
    return result


print("=" * 80)
print("DIAGNOSTIC: Hypersurface Identity Function")
print("=" * 80)

# Create hypersurface: curve (u, 0) in 2D - first coordinate is identity
hypersurface = Hypersurface(
    func=lambda u: np.array([u, 0.0]),
    param_ranges=[(0, 1)],
    ambient_dim=2,
    degree=5,
    verbose=True
)

print(f"\n--- Bernstein Coefficients ---")
print(f"x(u) coefficients: {hypersurface.bernstein_coeffs[0]}")
print(f"y(u) coefficients: {hypersurface.bernstein_coeffs[1]}")

# Expected for x(u) = u: [0, 0.2, 0.4, 0.6, 0.8, 1.0]
expected_x = np.array([i / 5 for i in range(6)])
print(f"Expected x(u) (identity): {expected_x}")

# Expected for y(u) = 0: [0, 0, 0, 0, 0, 0]
expected_y = np.zeros(6)
print(f"Expected y(u) (zero): {expected_y}")

# Test evaluation
print(f"\n--- Evaluation Test ---")
test_points = [0.0, 0.25, 0.5, 0.75, 1.0]

for u in test_points:
    # Direct evaluation
    point = hypersurface.evaluate(u)
    direct_x = point[0]
    direct_y = point[1]

    # Bernstein evaluation
    bernstein_x = evaluate_bernstein_1d(hypersurface.bernstein_coeffs[0], u)
    bernstein_y = evaluate_bernstein_1d(hypersurface.bernstein_coeffs[1], u)

    # Expected
    expected_x_val = u
    expected_y_val = 0.0

    print(f"u={u:.2f}:")
    print(f"  x: direct={direct_x:.6f}, bernstein={bernstein_x:.6f}, expected={expected_x_val:.6f}")
    print(f"     direct==expected: {np.isclose(direct_x, expected_x_val)}, bernstein==expected: {np.isclose(bernstein_x, expected_x_val)}")
    print(f"  y: direct={direct_y:.6f}, bernstein={bernstein_y:.6f}, expected={expected_y_val:.6f}")
    print(f"     direct==expected: {np.isclose(direct_y, expected_y_val)}, bernstein==expected: {np.isclose(bernstein_y, expected_y_val)}")

