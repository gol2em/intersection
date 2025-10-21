"""
Test the updated n-dimensional polynomial_system.py
"""

import numpy as np
from src.intersection.geometry import Hyperplane, Line, Hypersurface
from src.intersection.polynomial_system import create_intersection_system, evaluate_system

print("=" * 80)
print("Testing N-Dimensional Polynomial System Formation")
print("=" * 80)

# Test 1: 2D Line-Curve Intersection
print("\n" + "=" * 80)
print("Test 1: 2D Line-Curve Intersection")
print("=" * 80)

# Create a line in 2D: x - y = 0 (diagonal line)
h1 = Hyperplane(coeffs=[1, -1], d=0)
line_2d = Line([h1])
print(f"\nLine: {line_2d}")

# Create a curve in 2D: parabola y = x^2
curve = Hypersurface(
    func=lambda u: np.array([u, u**2]),
    param_ranges=[(0, 2)],
    ambient_dim=2,
    degree=5,
    verbose=False
)
print(f"Curve: {curve}")

# Create intersection system
system_2d = create_intersection_system(line_2d, curve, verbose=True)

# Test evaluation at some parameter values
print("\n--- Testing System Evaluation ---")
test_params = [0.0, 0.5, 1.0, 1.5, 2.0]
for u in test_params:
    residuals = evaluate_system(system_2d, u)
    point = curve.evaluate(u)
    print(f"u={u:.2f}: point=({point[0]:.4f}, {point[1]:.4f}), residual={residuals[0]:.6f}")

# The intersection should be at u=1 where x=1, y=1
print("\n--- Expected Intersection ---")
u_intersect = 1.0
point_intersect = curve.evaluate(u_intersect)
residual_intersect = evaluate_system(system_2d, u_intersect)
print(f"At u={u_intersect}: point={point_intersect}, residual={residual_intersect[0]:.10f}")
print(f"This should be close to 0 (intersection point)")


# Test 2: 3D Line-Surface Intersection
print("\n" + "=" * 80)
print("Test 2: 3D Line-Surface Intersection")
print("=" * 80)

# Create a line in 3D: x=1, y=0 (line parallel to z-axis)
h1 = Hyperplane(coeffs=[1, 0, 0], d=-1)  # x = 1
h2 = Hyperplane(coeffs=[0, 1, 0], d=0)   # y = 0
line_3d = Line([h1, h2])
print(f"\nLine: {line_3d}")

# Create a surface in 3D: z = x^2 + y^2 (paraboloid)
surface = Hypersurface(
    func=lambda u, v: np.array([u, v, u**2 + v**2]),
    param_ranges=[(0, 2), (-1, 1)],
    ambient_dim=3,
    degree=4,
    verbose=False
)
print(f"Surface: {surface}")

# Create intersection system
system_3d = create_intersection_system(line_3d, surface, verbose=True)

# Test evaluation at some parameter values
print("\n--- Testing System Evaluation ---")
test_params_3d = [
    (0.5, 0.0),
    (1.0, 0.0),
    (1.5, 0.0),
    (1.0, 0.5),
]
for u, v in test_params_3d:
    residuals = evaluate_system(system_3d, u, v)
    point = surface.evaluate(u, v)
    print(f"(u,v)=({u:.2f},{v:.2f}): point=({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}), "
          f"residuals=({residuals[0]:.6f}, {residuals[1]:.6f})")

# The intersection should be at u=1, v=0 where x=1, y=0, z=1
print("\n--- Expected Intersection ---")
u_int, v_int = 1.0, 0.0
point_int = surface.evaluate(u_int, v_int)
residuals_int = evaluate_system(system_3d, u_int, v_int)
print(f"At (u,v)=({u_int},{v_int}): point={point_int}, residuals={residuals_int}")
print(f"Residuals should be close to [0, 0] (intersection point)")
print(f"Residual norm: {np.linalg.norm(residuals_int):.10f}")


# Test 3: Circle-Line Intersection (2D)
print("\n" + "=" * 80)
print("Test 3: Circle-Line Intersection (2D)")
print("=" * 80)

# Create a horizontal line: y = 0.5
h1 = Hyperplane(coeffs=[0, 1], d=-0.5)  # y = 0.5
line_horiz = Line([h1])
print(f"\nLine: {line_horiz}")

# Create a unit circle
circle = Hypersurface(
    func=lambda u: np.array([np.cos(2*np.pi*u), np.sin(2*np.pi*u)]),
    param_ranges=[(0, 1)],
    ambient_dim=2,
    degree=8,
    verbose=False
)
print(f"Circle: {circle}")

# Create intersection system
system_circle = create_intersection_system(line_horiz, circle, verbose=True)

# Test evaluation
print("\n--- Testing System Evaluation ---")
test_u_circle = np.linspace(0, 1, 9)
for u in test_u_circle:
    residuals = evaluate_system(system_circle, u)
    point = circle.evaluate(u)
    print(f"u={u:.3f}: point=({point[0]:+.4f}, {point[1]:+.4f}), residual={residuals[0]:+.6f}")

# Find approximate intersections (where residual is close to 0)
print("\n--- Finding Intersections ---")
residuals_all = [evaluate_system(system_circle, u)[0] for u in test_u_circle]
min_residual_idx = np.argmin(np.abs(residuals_all))
print(f"Minimum residual at u={test_u_circle[min_residual_idx]:.3f}, "
      f"residual={residuals_all[min_residual_idx]:.6f}")

# Analytical solution: y = sin(2πu) = 0.5 => u ≈ 0.0833 or u ≈ 0.4167
print(f"\nAnalytical intersections:")
u_analytical = [np.arcsin(0.5) / (2*np.pi), (np.pi - np.arcsin(0.5)) / (2*np.pi)]
for u in u_analytical:
    point = circle.evaluate(u)
    residual = evaluate_system(system_circle, u)
    print(f"u={u:.4f}: point=({point[0]:+.6f}, {point[1]:+.6f}), residual={residual[0]:.10f}")


print("\n" + "=" * 80)
print("All Tests Completed Successfully!")
print("=" * 80)

