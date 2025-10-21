"""
Example: 2D Line-Curve Intersection

This example demonstrates finding intersections between a straight line
and a parametric curve (parabola).
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src')

from intersection import (
    Line2D, 
    ParametricCurve, 
    compute_intersections_2d,
    visualize_2d,
    print_intersection_summary
)


def example_parabola():
    """Example: Line intersecting a parabola."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Line intersecting a parabola")
    print("=" * 70)
    
    # Define parametric curve: parabola y = x^2
    def curve_x(t):
        return t
    
    def curve_y(t):
        return t**2
    
    curve = ParametricCurve(curve_x, curve_y, t_range=(0, 2))
    
    # Define a horizontal line at y = 0.5
    line = Line2D(point=(0, 0.5), direction=(1, 0))
    
    print("\nCurve: Parabola y = x^2, parameterized as (t, t^2) for t in [0, 2]")
    print(f"Line: Horizontal line at y = 0.5")
    
    # Compute intersections with verbose output
    intersections = compute_intersections_2d(line, curve, degree=5, verbose=True)
    
    # Print summary
    print_intersection_summary(intersections, dimension='2D')
    
    # Visualize
    fig, ax = visualize_2d(line, curve, intersections, 
                          title="Example 1: Line-Parabola Intersection")
    plt.savefig('example_2d_parabola.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: example_2d_parabola.png")
    plt.show()
    
    return intersections


def example_circle():
    """Example: Line intersecting a circle."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Line intersecting a circle")
    print("=" * 70)
    
    # Define parametric curve: circle
    def curve_x(t):
        return np.cos(2 * np.pi * t)
    
    def curve_y(t):
        return np.sin(2 * np.pi * t)
    
    curve = ParametricCurve(curve_x, curve_y, t_range=(0, 1))
    
    # Define a diagonal line
    line = Line2D(point=(-1.5, 0), direction=(1, 0.5))
    
    print("\nCurve: Circle parameterized as (cos(2πt), sin(2πt)) for t in [0, 1]")
    print(f"Line: Diagonal line through (-1.5, 0) with direction (1, 0.5)")
    
    # Compute intersections
    intersections = compute_intersections_2d(line, curve, degree=8, verbose=True)
    
    # Print summary
    print_intersection_summary(intersections, dimension='2D')
    
    # Visualize
    fig, ax = visualize_2d(line, curve, intersections,
                          title="Example 2: Line-Circle Intersection")
    plt.savefig('example_2d_circle.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: example_2d_circle.png")
    plt.show()
    
    return intersections


def example_sine_curve():
    """Example: Line intersecting a sine curve."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Line intersecting a sine curve")
    print("=" * 70)
    
    # Define parametric curve: sine wave
    def curve_x(t):
        return t
    
    def curve_y(t):
        return np.sin(2 * np.pi * t)
    
    curve = ParametricCurve(curve_x, curve_y, t_range=(0, 2))
    
    # Define a slightly tilted line
    line = Line2D(point=(0, 0), direction=(1, 0.1))
    
    print("\nCurve: Sine wave y = sin(2πt), parameterized as (t, sin(2πt)) for t in [0, 2]")
    print(f"Line: Slightly tilted line through origin with direction (1, 0.1)")
    
    # Compute intersections with higher degree for better accuracy
    intersections = compute_intersections_2d(line, curve, degree=10, verbose=True)
    
    # Print summary
    print_intersection_summary(intersections, dimension='2D')
    
    # Visualize
    fig, ax = visualize_2d(line, curve, intersections,
                          title="Example 3: Line-Sine Curve Intersection")
    plt.savefig('example_2d_sine.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: example_2d_sine.png")
    plt.show()
    
    return intersections


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("2D LINE-CURVE INTERSECTION EXAMPLES")
    print("=" * 70)
    
    # Run all examples
    example_parabola()
    example_circle()
    example_sine_curve()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70)

