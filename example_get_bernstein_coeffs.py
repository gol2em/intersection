"""
Example: How to get Bernstein coefficients from the intersection system
"""

import numpy as np
from src.intersection.geometry import Hyperplane, Line, Hypersurface
from src.intersection.polynomial_system import create_intersection_system

print("=" * 80)
print("How to Access Bernstein Coefficients from Intersection System")
print("=" * 80)

# Create a simple example: unit circle with degree 5
print("\n--- Creating Unit Circle (degree 5) ---")
circle = Hypersurface(
    func=lambda u: np.array([np.cos(2*np.pi*u), np.sin(2*np.pi*u)]),
    param_ranges=[(0, 1)],
    ambient_dim=2,
    degree=5,
    verbose=False
)

# Create a line: y = x
print("\n--- Creating Line: y = x ---")
h1 = Hyperplane(coeffs=[1, -1], d=0)
line = Line([h1])

# Create intersection system
print("\n--- Creating Intersection System ---")
system = create_intersection_system(line, circle, verbose=False)

print("\n" + "=" * 80)
print("METHOD 1: Get Bernstein Coefficients from System Dictionary")
print("=" * 80)

# The system dictionary contains 'bernstein_coeffs' key
bernstein_coeffs = system['bernstein_coeffs']

print(f"\nNumber of coordinates: {len(bernstein_coeffs)}")
print(f"Type: {type(bernstein_coeffs)}")

for i, bern in enumerate(bernstein_coeffs):
    print(f"\nCoordinate {i+1} (x{i+1}):")
    print(f"  Shape: {bern.shape}")
    print(f"  Dtype: {bern.dtype}")
    print(f"  Values: {bern}")

print("\n" + "=" * 80)
print("METHOD 2: Get Bernstein Coefficients from Hypersurface Object")
print("=" * 80)

# You can also get them directly from the hypersurface
hypersurface = system['hypersurface']
bern_from_hypersurface = hypersurface.bernstein_coeffs

print(f"\nNumber of coordinates: {len(bern_from_hypersurface)}")

for i, bern in enumerate(bern_from_hypersurface):
    print(f"\nCoordinate {i+1}:")
    print(f"  Shape: {bern.shape}")
    print(f"  Values: {bern}")

print("\n" + "=" * 80)
print("METHOD 3: Access Individual Coordinate Coefficients")
print("=" * 80)

# For 2D case (circle)
bern_x = system['bernstein_coeffs'][0]  # x-coordinate
bern_y = system['bernstein_coeffs'][1]  # y-coordinate

print(f"\nx(u) Bernstein coefficients:")
print(f"  {bern_x}")

print(f"\ny(u) Bernstein coefficients:")
print(f"  {bern_y}")

print("\n" + "=" * 80)
print("EXAMPLE: 3D Surface")
print("=" * 80)

# Create a 3D surface: paraboloid z = x^2 + y^2
print("\n--- Creating Paraboloid Surface (degree 4) ---")
surface = Hypersurface(
    func=lambda u, v: np.array([u, v, u**2 + v**2]),
    param_ranges=[(0, 1), (0, 1)],
    ambient_dim=3,
    degree=4,
    verbose=False
)

# Create a line: x=0.5, y=0.5 (parallel to z-axis)
print("\n--- Creating Line: x=0.5, y=0.5 ---")
h1 = Hyperplane(coeffs=[1, 0, 0], d=-0.5)  # x = 0.5
h2 = Hyperplane(coeffs=[0, 1, 0], d=-0.5)  # y = 0.5
line_3d = Line([h1, h2])

# Create intersection system
print("\n--- Creating 3D Intersection System ---")
system_3d = create_intersection_system(line_3d, surface, verbose=False)

# Get Bernstein coefficients
bernstein_coeffs_3d = system_3d['bernstein_coeffs']

print(f"\nNumber of coordinates: {len(bernstein_coeffs_3d)}")

for i, bern in enumerate(bernstein_coeffs_3d):
    print(f"\nCoordinate {i+1} (x{i+1}):")
    print(f"  Shape: {bern.shape}")
    print(f"  Number of coefficients: {bern.size}")
    if bern.ndim == 1:
        print(f"  Values: {bern}")
    else:
        print(f"  Values (2D array):")
        print(f"{bern}")

print("\n" + "=" * 80)
print("UNDERSTANDING THE STRUCTURE")
print("=" * 80)

print("""
For a k-dimensional hypersurface in n-dimensional space:

1. system['bernstein_coeffs'] is a LIST of n arrays
   - One array for each coordinate (x1, x2, ..., xn)

2. Each array has shape depending on k (number of parameters):
   - k=1 (curve):    shape = (degree+1,)           [1D array]
   - k=2 (surface):  shape = (degree+1, degree+1)  [2D array]
   - k=3:            shape = (degree+1, degree+1, degree+1)  [3D array]
   - etc.

3. For the 2D circle example (k=1, n=2):
   - bernstein_coeffs[0] = x(u) coefficients, shape (6,) for degree 5
   - bernstein_coeffs[1] = y(u) coefficients, shape (6,) for degree 5

4. For the 3D surface example (k=2, n=3):
   - bernstein_coeffs[0] = x(u,v) coefficients, shape (5,5) for degree 4
   - bernstein_coeffs[1] = y(u,v) coefficients, shape (5,5) for degree 4
   - bernstein_coeffs[2] = z(u,v) coefficients, shape (5,5) for degree 4
""")

print("\n" + "=" * 80)
print("QUICK REFERENCE")
print("=" * 80)

print("""
# Get all Bernstein coefficients:
bernstein_coeffs = system['bernstein_coeffs']

# For 2D (curve in 2D space):
bern_x = system['bernstein_coeffs'][0]  # shape: (degree+1,)
bern_y = system['bernstein_coeffs'][1]  # shape: (degree+1,)

# For 3D (surface in 3D space):
bern_x = system['bernstein_coeffs'][0]  # shape: (degree+1, degree+1)
bern_y = system['bernstein_coeffs'][1]  # shape: (degree+1, degree+1)
bern_z = system['bernstein_coeffs'][2]  # shape: (degree+1, degree+1)

# Alternative: Get from hypersurface directly
bern_coeffs = system['hypersurface'].bernstein_coeffs

# Get other useful information:
n = system['n']                    # Ambient dimension
k = system['k']                    # Number of parameters
degree = system['degree']          # Polynomial degree
param_ranges = system['param_ranges']  # Parameter ranges
""")

print("\n" + "=" * 80)
print("Example Complete!")
print("=" * 80)

