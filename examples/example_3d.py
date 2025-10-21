"""
Example: 3D Line-Surface Intersection

This example demonstrates the n-dimensional framework for 3D line-surface intersections.
Shows how to create lines, surfaces, and polynomial systems.
"""

import numpy as np
from src.intersection.geometry import Hyperplane, Line, Hypersurface
from src.intersection.polynomial_system import create_intersection_system, evaluate_system


def example_plane():
    """Example: Vertical line intersecting a tilted plane."""

    print("\n" + "=" * 70)
    print("EXAMPLE 1: Vertical line intersecting a tilted plane")
    print("=" * 70)

    # Create tilted plane: z = 0.5*x + 0.3*y
    print("\n--- Creating Tilted Plane ---")
    plane = Hypersurface(
        func=lambda u, v: np.array([u, v, 0.5*u + 0.3*v]),
        param_ranges=[(0, 2), (0, 2)],
        ambient_dim=3,
        degree=3,
        verbose=False
    )
    print(f"Plane: z = 0.5*x + 0.3*y, parameterized as (u, v, 0.5*u + 0.3*v)")
    print(f"Parameter ranges: u, v ∈ [0, 2]")
    print(f"Degree: {plane.degree}")

    # Create vertical line: x=1, y=1 (parallel to z-axis)
    print("\n--- Creating Line: x=1, y=1 ---")
    h1 = Hyperplane(coeffs=[1, 0, 0], d=-1)  # x = 1
    h2 = Hyperplane(coeffs=[0, 1, 0], d=-1)  # y = 1
    line = Line([h1, h2])
    print(f"Line: {line}")

    # Create intersection system
    print("\n--- Creating Intersection System ---")
    system = create_intersection_system(line, plane, verbose=True)

    # The intersection should be at u=1, v=1 where x=1, y=1, z=0.8
    print("\n--- Expected Intersection ---")
    u_int, v_int = 1.0, 1.0
    point_int = plane.evaluate(u_int, v_int)
    residuals_int = evaluate_system(system, u_int, v_int)
    print(f"At (u,v)=({u_int},{v_int}): point={point_int}")
    print(f"Residuals: {residuals_int}")
    print(f"Residual norm: {np.linalg.norm(residuals_int):.10f}")

    return system


def example_paraboloid():
    """Example: Vertical line intersecting a paraboloid."""

    print("\n" + "=" * 70)
    print("EXAMPLE 2: Vertical line intersecting a paraboloid")
    print("=" * 70)

    # Create paraboloid: z = x^2 + y^2
    print("\n--- Creating Paraboloid ---")
    paraboloid = Hypersurface(
        func=lambda u, v: np.array([u, v, u**2 + v**2]),
        param_ranges=[(0, 1), (0, 1)],
        ambient_dim=3,
        degree=5,
        verbose=False
    )
    print(f"Paraboloid: z = x^2 + y^2, parameterized as (u, v, u^2 + v^2)")
    print(f"Parameter ranges: u, v ∈ [0, 1]")
    print(f"Degree: {paraboloid.degree}")

    # Create vertical line: x=0.5, y=0.5 (parallel to z-axis)
    print("\n--- Creating Line: x=0.5, y=0.5 ---")
    h1 = Hyperplane(coeffs=[1, 0, 0], d=-0.5)  # x = 0.5
    h2 = Hyperplane(coeffs=[0, 1, 0], d=-0.5)  # y = 0.5
    line = Line([h1, h2])
    print(f"Line: {line}")

    # Create intersection system
    print("\n--- Creating Intersection System ---")
    system = create_intersection_system(line, paraboloid, verbose=True)

    # The intersection should be at u=0.5, v=0.5 where x=0.5, y=0.5, z=0.5
    print("\n--- Expected Intersection ---")
    u_int, v_int = 0.5, 0.5
    point_int = paraboloid.evaluate(u_int, v_int)
    residuals_int = evaluate_system(system, u_int, v_int)
    print(f"At (u,v)=({u_int},{v_int}): point={point_int}")
    print(f"Residuals: {residuals_int}")
    print(f"Residual norm: {np.linalg.norm(residuals_int):.10f}")

    return system


def example_sphere():
    """Example: Line through center intersecting a sphere."""

    print("\n" + "=" * 70)
    print("EXAMPLE 3: Line through center intersecting a sphere")
    print("=" * 70)

    # Create unit sphere using spherical coordinates
    print("\n--- Creating Unit Sphere ---")
    sphere = Hypersurface(
        func=lambda u, v: np.array([
            np.sin(np.pi*v) * np.cos(2*np.pi*u),
            np.sin(np.pi*v) * np.sin(2*np.pi*u),
            np.cos(np.pi*v)
        ]),
        param_ranges=[(0, 1), (0, 1)],
        ambient_dim=3,
        degree=6,
        verbose=False
    )
    print(f"Sphere: unit sphere in spherical coordinates")
    print(f"Parameter ranges: u, v ∈ [0, 1]")
    print(f"Degree: {sphere.degree}")

    # Create line through center: x=0, y=0 (z-axis)
    print("\n--- Creating Line: x=0, y=0 (z-axis) ---")
    h1 = Hyperplane(coeffs=[1, 0, 0], d=0)  # x = 0
    h2 = Hyperplane(coeffs=[0, 1, 0], d=0)  # y = 0
    line = Line([h1, h2])
    print(f"Line: {line}")

    # Create intersection system
    print("\n--- Creating Intersection System ---")
    system = create_intersection_system(line, sphere, verbose=True)

    # The intersections should be at north pole (u=any, v=0) and south pole (u=any, v=1)
    print("\n--- Expected Intersections ---")
    print("North pole: v=0 => point=(0, 0, 1)")
    u_north, v_north = 0.0, 0.0
    point_north = sphere.evaluate(u_north, v_north)
    residuals_north = evaluate_system(system, u_north, v_north)
    print(f"At (u,v)=({u_north},{v_north}): point={point_north}")
    print(f"Residuals: {residuals_north}, norm: {np.linalg.norm(residuals_north):.10f}")

    print("\nSouth pole: v=1 => point=(0, 0, -1)")
    u_south, v_south = 0.0, 1.0
    point_south = sphere.evaluate(u_south, v_south)
    residuals_south = evaluate_system(system, u_south, v_south)
    print(f"At (u,v)=({u_south},{v_south}): point={point_south}")
    print(f"Residuals: {residuals_south}, norm: {np.linalg.norm(residuals_south):.10f}")

    return system


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("3D LINE-SURFACE INTERSECTION EXAMPLES")
    print("=" * 70)

    # Run all examples
    print("\nRunning Example 1: Plane")
    example_plane()

    print("\nRunning Example 2: Paraboloid")
    example_paraboloid()

    print("\nRunning Example 3: Sphere")
    example_sphere()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70)
    print("\nNote: These examples demonstrate the n-dimensional framework.")
    print("The polynomial systems are formed correctly and ready for LP method implementation.")

