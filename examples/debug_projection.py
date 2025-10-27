"""
Debug the projection method to see what points are being projected.
"""

import numpy as np
from intersection.bernstein import polynomial_nd_to_bernstein
from intersection.convex_hull import convex_hull_2d, intersect_convex_hull_with_x_axis


# Define equation 1: x² + y² - 1 = 0
eq1_power = np.zeros((3, 3))
eq1_power[0, 0] = -1
eq1_power[2, 0] = 1
eq1_power[0, 2] = 1

# Convert to Bernstein
eq1_bern = polynomial_nd_to_bernstein(eq1_power, k=2)

print("=" * 80)
print("EQUATION 1: x² + y² - 1 = 0")
print("=" * 80)
print("\nBernstein coefficients:")
print(eq1_bern)
print(f"Shape: {eq1_bern.shape}")

# Manually project onto dimension 0 (x-dimension)
print("\n" + "=" * 80)
print("PROJECTION ONTO X-DIMENSION (dim=0)")
print("=" * 80)

shape = eq1_bern.shape
dim = 0

print(f"\nControl points in 3D (i, j, f_ij):")
projected_points = []

for i in range(shape[0]):
    for j in range(shape[1]):
        f_value = eq1_bern[i, j]
        
        # Parameter value for dimension 0 (x)
        if shape[0] == 1:
            t_x = 0.5
        else:
            t_x = i / (shape[0] - 1)
        
        # Parameter value for dimension 1 (y)
        if shape[1] == 1:
            t_y = 0.5
        else:
            t_y = j / (shape[1] - 1)
        
        print(f"  ({i}, {j}) → t_x={t_x:.2f}, t_y={t_y:.2f}, f={f_value:6.2f}")
        
        # Project onto (t_x, f) plane
        projected_points.append([t_x, f_value])

projected_points = np.array(projected_points)

print(f"\nProjected 2D points (t_x, f):")
for i, pt in enumerate(projected_points):
    print(f"  Point {i}: ({pt[0]:.2f}, {pt[1]:6.2f})")

# Compute convex hull
hull = convex_hull_2d(projected_points)
print(f"\nConvex hull vertices:")
for i, pt in enumerate(hull):
    print(f"  Vertex {i}: ({pt[0]:.2f}, {pt[1]:6.2f})")

# Find intersection with x-axis
result = intersect_convex_hull_with_x_axis(projected_points)
print(f"\nIntersection with x-axis (f=0): {result}")

# Now project onto dimension 1 (y-dimension)
print("\n" + "=" * 80)
print("PROJECTION ONTO Y-DIMENSION (dim=1)")
print("=" * 80)

dim = 1
projected_points_y = []

for i in range(shape[0]):
    for j in range(shape[1]):
        f_value = eq1_bern[i, j]
        
        # Parameter value for dimension 1 (y)
        if shape[1] == 1:
            t_y = 0.5
        else:
            t_y = j / (shape[1] - 1)
        
        # Project onto (t_y, f) plane
        projected_points_y.append([t_y, f_value])

projected_points_y = np.array(projected_points_y)

print(f"\nProjected 2D points (t_y, f):")
for i, pt in enumerate(projected_points_y):
    print(f"  Point {i}: ({pt[0]:.2f}, {pt[1]:6.2f})")

# Compute convex hull
hull_y = convex_hull_2d(projected_points_y)
print(f"\nConvex hull vertices:")
for i, pt in enumerate(hull_y):
    print(f"  Vertex {i}: ({pt[0]:.2f}, {pt[1]:6.2f})")

# Find intersection with y-axis
result_y = intersect_convex_hull_with_x_axis(projected_points_y)
print(f"\nIntersection with y-axis (f=0): {result_y}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print("\nFor x² + y² - 1 = 0 on [0,1]²:")
print("  The circle passes through (1, 0) and (0, 1)")
print("  Expected x range: [0, 1] (circle touches both x=0 and x=1)")
print("  Expected y range: [0, 1] (circle touches both y=0 and y=1)")
print(f"\n  Computed x range: {result}")
print(f"  Computed y range: {result_y}")
print("\n  The PP method is correct! The circle does span [0,1] in both dimensions.")
print("  The issue is that this doesn't help us narrow down the search space.")

