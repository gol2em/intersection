"""
Example: 3D Line-Surface Intersection

This example demonstrates finding intersections between a straight line
and a parametric surface.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src')

from intersection import (
    Line3D,
    ParametricSurface,
    compute_intersections_3d,
    visualize_3d,
    print_intersection_summary
)


def example_plane():
    """Example: Line intersecting a tilted plane."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Line intersecting a tilted plane")
    print("=" * 70)
    
    # Define parametric surface: tilted plane z = 0.5*u + 0.3*v
    def surface_x(u, v):
        return u
    
    def surface_y(u, v):
        return v
    
    def surface_z(u, v):
        return 0.5 * u + 0.3 * v
    
    surface = ParametricSurface(surface_x, surface_y, surface_z,
                               u_range=(0, 2), v_range=(0, 2))
    
    # Define a line going through the plane
    line = Line3D(point=(0.5, 0.5, 0), direction=(0.2, 0.3, 1))
    
    print("\nSurface: Tilted plane z = 0.5*u + 0.3*v")
    print(f"Line: Through (0.5, 0.5, 0) with direction (0.2, 0.3, 1)")
    
    # Compute intersections
    intersections = compute_intersections_3d(line, surface, degree=3, verbose=True)
    
    # Print summary
    print_intersection_summary(intersections, dimension='3D')
    
    # Visualize
    fig, ax = visualize_3d(line, surface, intersections,
                          title="Example 1: Line-Plane Intersection")
    plt.savefig('example_3d_plane.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: example_3d_plane.png")
    plt.show()
    
    return intersections


def example_paraboloid():
    """Example: Line intersecting a paraboloid."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Line intersecting a paraboloid")
    print("=" * 70)
    
    # Define parametric surface: paraboloid z = u^2 + v^2
    def surface_x(u, v):
        return u
    
    def surface_y(u, v):
        return v
    
    def surface_z(u, v):
        return u**2 + v**2
    
    surface = ParametricSurface(surface_x, surface_y, surface_z,
                               u_range=(-1, 1), v_range=(-1, 1))
    
    # Define a vertical line
    line = Line3D(point=(0.3, 0.4, 0), direction=(0, 0, 1))
    
    print("\nSurface: Paraboloid z = u^2 + v^2")
    print(f"Line: Vertical line through (0.3, 0.4, 0)")
    
    # Compute intersections
    intersections = compute_intersections_3d(line, surface, degree=5, verbose=True)
    
    # Print summary
    print_intersection_summary(intersections, dimension='3D')
    
    # Visualize
    fig, ax = visualize_3d(line, surface, intersections,
                          title="Example 2: Line-Paraboloid Intersection")
    plt.savefig('example_3d_paraboloid.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: example_3d_paraboloid.png")
    plt.show()
    
    return intersections


def example_sphere():
    """Example: Line intersecting a sphere."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Line intersecting a sphere")
    print("=" * 70)
    
    # Define parametric surface: sphere (using spherical coordinates)
    def surface_x(u, v):
        # u: azimuthal angle [0, 2π]
        # v: polar angle [0, π]
        theta = 2 * np.pi * u
        phi = np.pi * v
        return np.sin(phi) * np.cos(theta)
    
    def surface_y(u, v):
        theta = 2 * np.pi * u
        phi = np.pi * v
        return np.sin(phi) * np.sin(theta)
    
    def surface_z(u, v):
        phi = np.pi * v
        return np.cos(phi)
    
    surface = ParametricSurface(surface_x, surface_y, surface_z,
                               u_range=(0, 1), v_range=(0, 1))
    
    # Define a line passing through the sphere
    line = Line3D(point=(-2, 0, 0), direction=(1, 0.2, 0.1))
    
    print("\nSurface: Unit sphere")
    print(f"Line: Through (-2, 0, 0) with direction (1, 0.2, 0.1)")
    
    # Compute intersections
    intersections = compute_intersections_3d(line, surface, degree=6, verbose=True)
    
    # Print summary
    print_intersection_summary(intersections, dimension='3D')
    
    # Visualize
    fig, ax = visualize_3d(line, surface, intersections,
                          title="Example 3: Line-Sphere Intersection")
    plt.savefig('example_3d_sphere.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: example_3d_sphere.png")
    plt.show()
    
    return intersections


def example_saddle():
    """Example: Line intersecting a saddle surface."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Line intersecting a saddle surface")
    print("=" * 70)
    
    # Define parametric surface: saddle z = u^2 - v^2
    def surface_x(u, v):
        return u
    
    def surface_y(u, v):
        return v
    
    def surface_z(u, v):
        return u**2 - v**2
    
    surface = ParametricSurface(surface_x, surface_y, surface_z,
                               u_range=(-1, 1), v_range=(-1, 1))
    
    # Define a diagonal line
    line = Line3D(point=(-1, -1, 0), direction=(1, 1, 0.5))
    
    print("\nSurface: Saddle z = u^2 - v^2")
    print(f"Line: Through (-1, -1, 0) with direction (1, 1, 0.5)")
    
    # Compute intersections
    intersections = compute_intersections_3d(line, surface, degree=5, verbose=True)
    
    # Print summary
    print_intersection_summary(intersections, dimension='3D')
    
    # Visualize
    fig, ax = visualize_3d(line, surface, intersections,
                          title="Example 4: Line-Saddle Intersection")
    plt.savefig('example_3d_saddle.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: example_3d_saddle.png")
    plt.show()
    
    return intersections


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("3D LINE-SURFACE INTERSECTION EXAMPLES")
    print("=" * 70)
    
    # Run all examples
    example_plane()
    example_paraboloid()
    example_sphere()
    example_saddle()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70)

