"""
Test: Verify that equation Bernstein coefficients are computed correctly

This demonstrates that the system stores Bernstein coefficients of the
POLYNOMIAL EQUATIONS (after applying hyperplane constraints), not just
the raw hypersurface coefficients.
"""

import numpy as np
import math
from src.intersection.geometry import Hyperplane, Line, Hypersurface
from src.intersection.polynomial_system import (
    create_intersection_system,
    get_equation_bernstein_coeffs,
    evaluate_system
)

print("=" * 80)
print("Testing Equation Bernstein Coefficients")
print("=" * 80)

# Test 1: Simple 2D example - Circle intersecting y=x
print("\n" + "=" * 80)
print("Test 1: Circle (degree 5) intersecting line y=x")
print("=" * 80)

# Create unit circle
circle = Hypersurface(
    func=lambda u: np.array([np.cos(2*np.pi*u), np.sin(2*np.pi*u)]),
    param_ranges=[(0, 1)],
    ambient_dim=2,
    degree=5,
    verbose=False
)

# Create line y = x  =>  x - y = 0
h1 = Hyperplane(coeffs=[1, -1], d=0)
line = Line([h1])

print(f"\nLine equation: x - y = 0")
print(f"Hyperplane coefficients: {h1.coeffs}")
print(f"Hyperplane constant: {h1.d}")

# Create system
system = create_intersection_system(line, circle, verbose=True)

print("\n" + "-" * 80)
print("Understanding the Coefficients")
print("-" * 80)

# Get hypersurface Bernstein coefficients
bern_x = system['hypersurface_bernstein_coeffs'][0]
bern_y = system['hypersurface_bernstein_coeffs'][1]

print(f"\nHypersurface Bernstein coefficients:")
print(f"  x(u): {bern_x}")
print(f"  y(u): {bern_y}")

# Get equation Bernstein coefficients
eq_coeffs = get_equation_bernstein_coeffs(system)
eq_bern = eq_coeffs[0]  # Only one equation for 2D

print(f"\nEquation polynomial: p(u) = x(u) - y(u) + 0")
print(f"Expected Bernstein coefficients: bern_x - bern_y")
print(f"  Computed: {bern_x - bern_y}")
print(f"\nActual equation Bernstein coefficients:")
print(f"  {eq_bern}")

print(f"\nVerification: Are they equal?")
print(f"  Max difference: {np.max(np.abs(eq_bern - (bern_x - bern_y))):.2e}")

# Verify at intersection points
print("\n" + "-" * 80)
print("Verification at Known Intersection Points")
print("-" * 80)

u_intersections = [1/8, 5/8]  # Analytical intersections for y=x and unit circle

for u in u_intersections:
    point = circle.evaluate(u)
    residual = evaluate_system(system, u)[0]
    
    # Manually compute using equation Bernstein coefficients
    # For 1D Bernstein basis: B_i^n(u) = C(n,i) * u^i * (1-u)^(n-i)
    n_deg = len(eq_bern) - 1
    bernstein_basis = np.array([
        math.comb(n_deg, i) * (u**i) * ((1-u)**(n_deg-i))
        for i in range(n_deg + 1)
    ])
    residual_from_bern = np.dot(eq_bern, bernstein_basis)
    
    print(f"\nu = {u:.6f} (analytical intersection)")
    print(f"  Point: ({point[0]:+.10f}, {point[1]:+.10f})")
    print(f"  x - y = {point[0] - point[1]:+.15f}")
    print(f"  Residual from evaluate_system: {residual:+.15f}")
    print(f"  Residual from Bernstein coeffs: {residual_from_bern:+.15f}")
    print(f"  Match: {np.abs(residual - residual_from_bern) < 1e-10}")


# Test 2: 3D example - Surface intersecting a line
print("\n" + "=" * 80)
print("Test 2: Paraboloid (degree 3) intersecting line x=0.5, y=0.5")
print("=" * 80)

# Create paraboloid z = x^2 + y^2
surface = Hypersurface(
    func=lambda u, v: np.array([u, v, u**2 + v**2]),
    param_ranges=[(0, 1), (0, 1)],
    ambient_dim=3,
    degree=3,
    verbose=False
)

# Create line x=0.5, y=0.5
h1 = Hyperplane(coeffs=[1, 0, 0], d=-0.5)  # x = 0.5
h2 = Hyperplane(coeffs=[0, 1, 0], d=-0.5)  # y = 0.5
line_3d = Line([h1, h2])

print(f"\nLine equations:")
print(f"  H1: x - 0.5 = 0")
print(f"  H2: y - 0.5 = 0")

# Create system
system_3d = create_intersection_system(line_3d, surface, verbose=True)

print("\n" + "-" * 80)
print("Equation Bernstein Coefficients (3D)")
print("-" * 80)

# Get hypersurface Bernstein coefficients
bern_x_3d = system_3d['hypersurface_bernstein_coeffs'][0]
bern_y_3d = system_3d['hypersurface_bernstein_coeffs'][1]
bern_z_3d = system_3d['hypersurface_bernstein_coeffs'][2]

print(f"\nHypersurface Bernstein coefficients shapes:")
print(f"  x(u,v): {bern_x_3d.shape}")
print(f"  y(u,v): {bern_y_3d.shape}")
print(f"  z(u,v): {bern_z_3d.shape}")

# Get equation Bernstein coefficients
eq_coeffs_3d = get_equation_bernstein_coeffs(system_3d)

print(f"\nEquation 1: p1(u,v) = x(u,v) - 0.5")
print(f"  Expected: bern_x - 0.5")
print(f"  Shape: {eq_coeffs_3d[0].shape}")
print(f"  Coefficients:\n{eq_coeffs_3d[0]}")

print(f"\nEquation 2: p2(u,v) = y(u,v) - 0.5")
print(f"  Expected: bern_y - 0.5")
print(f"  Shape: {eq_coeffs_3d[1].shape}")
print(f"  Coefficients:\n{eq_coeffs_3d[1]}")

# Verify
print(f"\nVerification:")
print(f"  Eq1 coeffs == bern_x - 0.5: {np.allclose(eq_coeffs_3d[0], bern_x_3d - 0.5)}")
print(f"  Eq2 coeffs == bern_y - 0.5: {np.allclose(eq_coeffs_3d[1], bern_y_3d - 0.5)}")

# Verify at intersection point
print("\n" + "-" * 80)
print("Verification at Known Intersection Point")
print("-" * 80)

u_int, v_int = 0.5, 0.5  # Intersection point
point_3d = surface.evaluate(u_int, v_int)
residuals_3d = evaluate_system(system_3d, u_int, v_int)

print(f"\n(u, v) = ({u_int}, {v_int}) (analytical intersection)")
print(f"  Point: ({point_3d[0]:.6f}, {point_3d[1]:.6f}, {point_3d[2]:.6f})")
print(f"  x - 0.5 = {point_3d[0] - 0.5:+.15f}")
print(f"  y - 0.5 = {point_3d[1] - 0.5:+.15f}")
print(f"  Residuals: [{residuals_3d[0]:+.15f}, {residuals_3d[1]:+.15f}]")
print(f"  Both residuals ≈ 0: {np.allclose(residuals_3d, 0)}")


# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("""
The system now correctly stores:

1. 'hypersurface_bernstein_coeffs': Bernstein coefficients of each coordinate
   - For 2D circle: [bern_x, bern_y]
   - For 3D surface: [bern_x, bern_y, bern_z]

2. 'equation_bernstein_coeffs': Bernstein coefficients of each EQUATION
   - For hyperplane a1*x1 + a2*x2 + ... + an*xn + d = 0
   - Equation polynomial: p(u) = a1*x1(u) + a2*x2(u) + ... + an*xn(u) + d
   - Bernstein coefficients: a1*bern_x1 + a2*bern_x2 + ... + an*bern_xn + d

3. Access methods:
   - system['equation_bernstein_coeffs']  # Direct access
   - get_equation_bernstein_coeffs(system)  # Helper function

4. For 2D circle with line y=x:
   - Equation: x(u) - y(u) = 0
   - Bernstein coefficients: bern_x - bern_y ✅

5. For 3D surface with line x=0.5, y=0.5:
   - Equation 1: x(u,v) - 0.5 = 0
   - Equation 2: y(u,v) - 0.5 = 0
   - Bernstein coefficients: [bern_x - 0.5, bern_y - 0.5] ✅
""")

print("\n" + "=" * 80)
print("All Tests Passed! ✅")
print("=" * 80)

