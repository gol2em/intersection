"""
Test circle and sphere with varying polynomial degrees
"""

import numpy as np
from src.intersection.geometry import Hypersurface
from scipy.special import comb


def print_bernstein_basis_1d(degree):
    """Print the 1D Bernstein basis functions for given degree"""
    print(f"\n--- 1D Bernstein Basis Functions (degree {degree}) ---")
    print(f"B_i^{degree}(u) = C({degree},i) * u^i * (1-u)^({degree}-i)")
    print()
    for i in range(degree + 1):
        coeff = comb(degree, i, exact=True)
        print(f"B_{i}^{degree}(u) = {coeff} * u^{i} * (1-u)^{degree - i}")
    print()


def print_bernstein_basis_2d(degree):
    """Print the 2D Bernstein basis functions for given degree"""
    print(f"\n--- 2D Bernstein Basis Functions (degree {degree} x {degree}) ---")
    print(f"B_{{i,j}}^{degree}(u,v) = B_i^{degree}(u) * B_j^{degree}(v)")
    print(f"             = C({degree},i) * u^i * (1-u)^({degree}-i) * C({degree},j) * v^j * (1-v)^({degree}-j)")
    print()
    print(f"Total basis functions: {(degree + 1) * (degree + 1)}")
    print(f"\nFirst few basis functions:")
    for i in range(min(3, degree + 1)):
        for j in range(min(3, degree + 1)):
            coeff_i = comb(degree, i, exact=True)
            coeff_j = comb(degree, j, exact=True)
            print(f"B_{{{i},{j}}}^{degree}(u,v) = {coeff_i * coeff_j} * u^{i} * (1-u)^{degree - i} * v^{j} * (1-v)^{degree - j}")
    if degree >= 3:
        print(f"... ({(degree + 1) * (degree + 1) - 9} more basis functions)")
    print()


print("=" * 80)
print("CIRCLE TEST: x = cos(2πu), y = sin(2πu), u ∈ [0, 1]")
print("=" * 80)

for degree in range(2, 11):
    print(f"\n{'=' * 80}")
    print(f"DEGREE {degree}")
    print("=" * 80)

    # Print Bernstein basis functions
    print_bernstein_basis_1d(degree)

    circle = Hypersurface(
        func=lambda u: np.array([np.cos(2*np.pi*u), np.sin(2*np.pi*u)]),
        param_ranges=[(0, 1)],
        ambient_dim=2,
        degree=degree,
        verbose=True
    )
    
    # Test evaluation at key points
    print(f"\n--- Evaluation Tests (Degree {degree}) ---")
    test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    for u in test_points:
        point = circle.evaluate(u)
        expected = np.array([np.cos(2*np.pi*u), np.sin(2*np.pi*u)])
        error = np.linalg.norm(point - expected)
        print(f"u={u:.2f}: point=({point[0]:+.6f}, {point[1]:+.6f}), "
              f"expected=({expected[0]:+.6f}, {expected[1]:+.6f}), error={error:.2e}")
    
    print(f"\n{'=' * 80}\n")


print("\n" + "=" * 80)
print("SPHERE TEST: Spherical coordinates")
print("x = sin(πv)cos(2πu), y = sin(πv)sin(2πu), z = cos(πv)")
print("u, v ∈ [0, 1]")
print("=" * 80)

for degree in range(2, 11):
    print(f"\n{'=' * 80}")
    print(f"DEGREE {degree}")
    print("=" * 80)

    # Print Bernstein basis functions
    print_bernstein_basis_2d(degree)

    sphere = Hypersurface(
        func=lambda u, v: np.array([
            np.sin(np.pi*v) * np.cos(2*np.pi*u),
            np.sin(np.pi*v) * np.sin(2*np.pi*u),
            np.cos(np.pi*v)
        ]),
        param_ranges=[(0, 1), (0, 1)],
        ambient_dim=3,
        degree=degree,
        verbose=True
    )
    
    # Test evaluation at key points
    print(f"\n--- Evaluation Tests (Degree {degree}) ---")
    test_points = [
        (0.0, 0.0),   # North pole
        (0.0, 0.5),   # Equator
        (0.0, 1.0),   # South pole
        (0.25, 0.5),  # Equator, 90°
        (0.5, 0.5),   # Equator, 180°
        (0.75, 0.5),  # Equator, 270°
    ]
    
    for u, v in test_points:
        point = sphere.evaluate(u, v)
        expected = np.array([
            np.sin(np.pi*v) * np.cos(2*np.pi*u),
            np.sin(np.pi*v) * np.sin(2*np.pi*u),
            np.cos(np.pi*v)
        ])
        error = np.linalg.norm(point - expected)
        print(f"(u,v)=({u:.2f},{v:.2f}): "
              f"point=({point[0]:+.6f}, {point[1]:+.6f}, {point[2]:+.6f}), "
              f"expected=({expected[0]:+.6f}, {expected[1]:+.6f}, {expected[2]:+.6f}), "
              f"error={error:.2e}")
    
    print(f"\n{'=' * 80}\n")

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED")
print("=" * 80)

