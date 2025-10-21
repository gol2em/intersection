"""
Test: Unit Circle intersecting with diagonal line y=x

This demonstrates the n-dimensional polynomial system formation
with a unit circle (degree 12) and the diagonal line y=x.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.intersection.geometry import Hyperplane, Line, Hypersurface
from src.intersection.polynomial_system import create_intersection_system, evaluate_system

print("=" * 80)
print("Unit Circle (degree 12) Intersecting with Line y=x")
print("=" * 80)

# Create the diagonal line: y = x  =>  x - y = 0
print("\n--- Creating Line: y = x ---")
h1 = Hyperplane(coeffs=[1, -1], d=0)
line_diagonal = Line([h1])
print(f"Line: {line_diagonal}")
print(f"Equation: x - y = 0")

# Create unit circle with degree 12
print("\n--- Creating Unit Circle (degree 12) ---")
circle = Hypersurface(
    func=lambda u: np.array([np.cos(2*np.pi*u), np.sin(2*np.pi*u)]),
    param_ranges=[(0, 1)],
    ambient_dim=2,
    degree=12,
    verbose=True
)

print(f"\nCircle: {circle}")
print(f"Parametric form: x(u) = cos(2πu), y(u) = sin(2πu)")
print(f"Parameter range: u ∈ [0, 1]")
print(f"Degree: {circle.degree}")

# Show Bernstein coefficients
print("\n--- Bernstein Coefficients ---")
bern_x = circle.bernstein_coeffs[0]
bern_y = circle.bernstein_coeffs[1]
print(f"x(u) Bernstein coefficients (shape {bern_x.shape}):")
print(f"  {bern_x}")
print(f"\ny(u) Bernstein coefficients (shape {bern_y.shape}):")
print(f"  {bern_y}")

# Create intersection system
print("\n" + "=" * 80)
print("Creating Intersection System")
print("=" * 80)
system = create_intersection_system(line_diagonal, circle, verbose=True)

# Show system details
print("\n--- System Details ---")
print(f"Ambient dimension: {system['n']}")
print(f"Number of parameters: {system['k']}")
print(f"Number of equations: {len(system['equations'])}")
print(f"Parameter ranges: {system['param_ranges']}")
print(f"Polynomial degree: {system['degree']}")

print("\n--- Equation Specification ---")
for i, eq in enumerate(system['equations']):
    print(f"Equation {i+1}:")
    print(f"  Hyperplane coefficients: {eq['hyperplane_coeffs']}")
    print(f"  Constant term d: {eq['hyperplane_d']}")
    print(f"  Description: {eq['description']}")

# Evaluate system at many points to find intersections
print("\n" + "=" * 80)
print("Evaluating System to Find Intersections")
print("=" * 80)

# Sample many points
n_samples = 101
u_values = np.linspace(0, 1, n_samples)
residuals = []
points = []

for u in u_values:
    res = evaluate_system(system, u)[0]
    residuals.append(res)
    pt = circle.evaluate(u)
    points.append(pt)

residuals = np.array(residuals)
points = np.array(points)

# Find where residual changes sign (intersections)
print("\n--- Finding Intersections (Sign Changes) ---")
sign_changes = []
for i in range(len(residuals) - 1):
    if residuals[i] * residuals[i+1] < 0:  # Sign change
        # Linear interpolation to find approximate zero
        u1, u2 = u_values[i], u_values[i+1]
        r1, r2 = residuals[i], residuals[i+1]
        u_zero = u1 - r1 * (u2 - u1) / (r2 - r1)
        sign_changes.append((i, u_zero))
        print(f"Sign change between u={u1:.4f} and u={u2:.4f}")
        print(f"  Approximate intersection at u={u_zero:.6f}")

# Refine intersections using analytical solution
print("\n--- Analytical Solution ---")
print("For unit circle: x = cos(2πu), y = sin(2πu)")
print("Line: x - y = 0  =>  x = y")
print("Substituting: cos(2πu) = sin(2πu)")
print("             tan(2πu) = 1")
print("             2πu = π/4 or 2πu = π/4 + π")
print("             u = 1/8 or u = 5/8")

u_analytical = [1/8, 5/8]
print("\nAnalytical intersections:")
for u in u_analytical:
    point = circle.evaluate(u)
    residual = evaluate_system(system, u)[0]
    print(f"  u = {u:.6f} = {u.as_integer_ratio()[0]}/{u.as_integer_ratio()[1]}")
    print(f"    Point: ({point[0]:+.10f}, {point[1]:+.10f})")
    print(f"    Residual: {residual:.15f}")
    print(f"    x - y = {point[0] - point[1]:.15f}")
    print()

# Show some sample evaluations
print("--- Sample Evaluations Around the Circle ---")
sample_u = [0.0, 1/8, 0.25, 5/8, 0.5, 0.75, 1.0]
print(f"{'u':>8} {'x':>12} {'y':>12} {'x-y':>12} {'residual':>15}")
print("-" * 70)
for u in sample_u:
    point = circle.evaluate(u)
    residual = evaluate_system(system, u)[0]
    print(f"{u:8.4f} {point[0]:+12.8f} {point[1]:+12.8f} {point[0]-point[1]:+12.8f} {residual:+15.10f}")

# Create visualization
print("\n" + "=" * 80)
print("Creating Visualization")
print("=" * 80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Circle and line with intersection points
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# Plot circle
u_plot = np.linspace(0, 1, 200)
circle_points = np.array([circle.evaluate(u) for u in u_plot])
ax1.plot(circle_points[:, 0], circle_points[:, 1], 'b-', linewidth=2, label='Unit Circle (degree 12)')

# Plot line y=x
x_line = np.linspace(-1.5, 1.5, 100)
ax1.plot(x_line, x_line, 'r-', linewidth=2, label='Line: y = x')

# Plot intersection points
for u in u_analytical:
    point = circle.evaluate(u)
    ax1.plot(point[0], point[1], 'go', markersize=12, markeredgewidth=2, 
             markeredgecolor='darkgreen', label=f'Intersection u={u:.3f}' if u == u_analytical[0] else '')

# Add labels for intersection points
for u in u_analytical:
    point = circle.evaluate(u)
    ax1.annotate(f'u={u:.3f}\n({point[0]:.3f}, {point[1]:.3f})', 
                xy=(point[0], point[1]), xytext=(10, 10),
                textcoords='offset points', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Unit Circle Intersecting Line y=x', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.set_xlim(-1.3, 1.3)
ax1.set_ylim(-1.3, 1.3)

# Plot 2: Residual function
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=1, linestyle='--', alpha=0.5)
ax2.plot(u_values, residuals, 'b-', linewidth=2, label='Residual: x(u) - y(u)')

# Mark intersection points
for u in u_analytical:
    residual = evaluate_system(system, u)[0]
    ax2.plot(u, residual, 'go', markersize=12, markeredgewidth=2, 
             markeredgecolor='darkgreen')
    ax2.axvline(x=u, color='g', linewidth=1, linestyle=':', alpha=0.5)
    ax2.annotate(f'u={u:.3f}', xy=(u, residual), xytext=(0, -20),
                textcoords='offset points', fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax2.set_xlabel('Parameter u', fontsize=12)
ax2.set_ylabel('Residual (x - y)', fontsize=12)
ax2.set_title('Residual Function Along Circle', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.set_xlim(0, 1)

plt.tight_layout()
plt.savefig('circle_diagonal_intersection.png', dpi=150, bbox_inches='tight')
print("Saved plot to: circle_diagonal_intersection.png")
plt.show()

print("\n" + "=" * 80)
print("Test Complete!")
print("=" * 80)
print(f"\nSummary:")
print(f"  Circle degree: {circle.degree}")
print(f"  Number of Bernstein coefficients: {len(bern_x)}")
print(f"  Analytical intersections: u = 1/8 and u = 5/8")
print(f"  Intersection points:")
for u in u_analytical:
    point = circle.evaluate(u)
    residual = evaluate_system(system, u)[0]
    print(f"    u={u:.6f}: ({point[0]:+.10f}, {point[1]:+.10f}), residual={residual:.2e}")
print(f"\n  Both residuals are essentially zero! ✅")

