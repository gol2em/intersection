"""
Example: 2D Line-Curve Intersection

This example demonstrates the n-dimensional framework for 2D line-curve intersections.
Shows how to create lines, curves, and polynomial systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.intersection.geometry import Hyperplane, Line, Hypersurface
from src.intersection.polynomial_system import create_intersection_system, evaluate_system


def example_parabola():
    """Example: Horizontal line intersecting a parabola."""

    print("\n" + "=" * 70)
    print("EXAMPLE 1: Horizontal line intersecting a parabola")
    print("=" * 70)

    # Create parabola: y = x^2, parameterized as (u, u^2) for u in [0, 2]
    print("\n--- Creating Parabola ---")
    parabola = Hypersurface(
        func=lambda u: np.array([u, u**2]),
        param_ranges=[(0, 2)],
        ambient_dim=2,
        degree=5,
        verbose=False
    )
    print(f"Parabola: y = x^2, parameterized as (u, u^2)")
    print(f"Parameter range: u ∈ [0, 2]")
    print(f"Degree: {parabola.degree}")

    # Create horizontal line: y = 0.5
    print("\n--- Creating Line: y = 0.5 ---")
    h1 = Hyperplane(coeffs=[0, 1], d=-0.5)  # y = 0.5
    line = Line([h1])
    print(f"Line: {line}")

    # Create intersection system
    print("\n--- Creating Intersection System ---")
    system = create_intersection_system(line, parabola, verbose=True)

    # Find approximate intersections by sampling
    print("\n--- Finding Intersections ---")
    u_values = np.linspace(0, 2, 201)
    residuals = [evaluate_system(system, u)[0] for u in u_values]

    # Find sign changes
    intersections = []
    for i in range(len(residuals) - 1):
        if residuals[i] * residuals[i+1] < 0:
            # Linear interpolation
            u1, u2 = u_values[i], u_values[i+1]
            r1, r2 = residuals[i], residuals[i+1]
            u_int = u1 - r1 * (u2 - u1) / (r2 - r1)
            point = parabola.evaluate(u_int)
            intersections.append((u_int, point))
            print(f"Intersection at u={u_int:.6f}, point=({point[0]:.6f}, {point[1]:.6f})")

    # Analytical solution: y = x^2 = 0.5 => x = ±sqrt(0.5) ≈ ±0.707
    print("\n--- Analytical Solution ---")
    print(f"Expected: x = sqrt(0.5) ≈ 0.707 (within [0, 2])")
    print(f"Found {len(intersections)} intersection(s)")

    return system, intersections


def example_circle():
    """Example: Diagonal line intersecting a circle."""

    print("\n" + "=" * 70)
    print("EXAMPLE 2: Diagonal line intersecting a circle")
    print("=" * 70)

    # Create unit circle
    print("\n--- Creating Unit Circle ---")
    circle = Hypersurface(
        func=lambda u: np.array([np.cos(2*np.pi*u), np.sin(2*np.pi*u)]),
        param_ranges=[(0, 1)],
        ambient_dim=2,
        degree=8,
        verbose=False
    )
    print(f"Circle: parameterized as (cos(2πu), sin(2πu))")
    print(f"Parameter range: u ∈ [0, 1]")
    print(f"Degree: {circle.degree}")

    # Create diagonal line: y = x
    print("\n--- Creating Line: y = x ---")
    h1 = Hyperplane(coeffs=[1, -1], d=0)  # x - y = 0
    line = Line([h1])
    print(f"Line: {line}")

    # Create intersection system
    print("\n--- Creating Intersection System ---")
    system = create_intersection_system(line, circle, verbose=True)

    # Find approximate intersections
    print("\n--- Finding Intersections ---")
    u_values = np.linspace(0, 1, 201)
    residuals = [evaluate_system(system, u)[0] for u in u_values]

    # Find sign changes
    intersections = []
    for i in range(len(residuals) - 1):
        if residuals[i] * residuals[i+1] < 0:
            u1, u2 = u_values[i], u_values[i+1]
            r1, r2 = residuals[i], residuals[i+1]
            u_int = u1 - r1 * (u2 - u1) / (r2 - r1)
            point = circle.evaluate(u_int)
            intersections.append((u_int, point))
            print(f"Intersection at u={u_int:.6f}, point=({point[0]:.6f}, {point[1]:.6f})")

    # Analytical solution: x = y on unit circle => x = y = ±1/sqrt(2) ≈ ±0.707
    print("\n--- Analytical Solution ---")
    print(f"Expected: (1/sqrt(2), 1/sqrt(2)) and (-1/sqrt(2), -1/sqrt(2))")
    print(f"Found {len(intersections)} intersection(s)")

    return system, intersections


def example_sine_curve():
    """Example: Horizontal line intersecting a sine curve."""

    print("\n" + "=" * 70)
    print("EXAMPLE 3: Horizontal line intersecting a sine curve")
    print("=" * 70)

    # Create sine curve: y = sin(2πx)
    print("\n--- Creating Sine Curve ---")
    sine_curve = Hypersurface(
        func=lambda u: np.array([u, np.sin(2*np.pi*u)]),
        param_ranges=[(0, 2)],
        ambient_dim=2,
        degree=10,
        verbose=False
    )
    print(f"Sine curve: y = sin(2πu), parameterized as (u, sin(2πu))")
    print(f"Parameter range: u ∈ [0, 2]")
    print(f"Degree: {sine_curve.degree}")

    # Create horizontal line: y = 0.5
    print("\n--- Creating Line: y = 0.5 ---")
    h1 = Hyperplane(coeffs=[0, 1], d=-0.5)  # y = 0.5
    line = Line([h1])
    print(f"Line: {line}")

    # Create intersection system
    print("\n--- Creating Intersection System ---")
    system = create_intersection_system(line, sine_curve, verbose=True)

    # Find approximate intersections
    print("\n--- Finding Intersections ---")
    u_values = np.linspace(0, 2, 401)
    residuals = [evaluate_system(system, u)[0] for u in u_values]

    # Find sign changes
    intersections = []
    for i in range(len(residuals) - 1):
        if residuals[i] * residuals[i+1] < 0:
            u1, u2 = u_values[i], u_values[i+1]
            r1, r2 = residuals[i], residuals[i+1]
            u_int = u1 - r1 * (u2 - u1) / (r2 - r1)
            point = sine_curve.evaluate(u_int)
            intersections.append((u_int, point))
            print(f"Intersection at u={u_int:.6f}, point=({point[0]:.6f}, {point[1]:.6f})")

    # Analytical solution: sin(2πx) = 0.5 => x = 1/12, 5/12, 13/12, 17/12 (in [0, 2])
    print("\n--- Analytical Solution ---")
    print(f"Expected: x ≈ 0.0833, 0.4167, 1.0833, 1.4167 (approximately)")
    print(f"Found {len(intersections)} intersection(s)")

    return system, intersections


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("2D LINE-CURVE INTERSECTION EXAMPLES")
    print("=" * 70)

    # Run all examples
    print("\nRunning Example 1: Parabola")
    example_parabola()

    print("\nRunning Example 2: Circle")
    example_circle()

    print("\nRunning Example 3: Sine Curve")
    example_sine_curve()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70)
    print("\nNote: These examples demonstrate the n-dimensional framework.")
    print("The polynomial systems are formed correctly and ready for LP method implementation.")

