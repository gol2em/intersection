"""
Example: How to get Bernstein coefficients from the intersection system

This example demonstrates the difference between:
1. Hypersurface Bernstein coefficients (coordinate functions)
2. Equation Bernstein coefficients (polynomial equations after applying hyperplane constraints)
"""

import numpy as np
from src.intersection.geometry import Hyperplane, Line, Hypersurface
from src.intersection.polynomial_system import create_intersection_system, get_equation_bernstein_coeffs

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
print("METHOD 1: Get Hypersurface Bernstein Coefficients (Coordinate Functions)")
print("=" * 80)

# The system dictionary contains 'hypersurface_bernstein_coeffs' key
hypersurface_coeffs = system['hypersurface_bernstein_coeffs']

print(f"\nNumber of coordinates: {len(hypersurface_coeffs)}")
print(f"Type: {type(hypersurface_coeffs)}")
print("\nThese are the Bernstein coefficients of the coordinate functions x(u) and y(u):")

for i, bern in enumerate(hypersurface_coeffs):
    print(f"\nCoordinate {i+1} (x{i+1}):")
    print(f"  Shape: {bern.shape}")
    print(f"  Dtype: {bern.dtype}")
    print(f"  Values: {bern}")

print("\n" + "=" * 80)
print("METHOD 2: Get Equation Bernstein Coefficients (Polynomial Equations)")
print("=" * 80)

# The system dictionary contains 'equation_bernstein_coeffs' key
# These are the Bernstein coefficients of the polynomial equations AFTER applying hyperplane constraints
equation_coeffs = system['equation_bernstein_coeffs']

print(f"\nNumber of equations: {len(equation_coeffs)}")
print(f"Type: {type(equation_coeffs)}")
print("\nThese are the Bernstein coefficients of the equation polynomial p(u) = x(u) - y(u):")

for i, eq_bern in enumerate(equation_coeffs):
    print(f"\nEquation {i+1}:")
    print(f"  Shape: {eq_bern.shape}")
    print(f"  Dtype: {eq_bern.dtype}")
    print(f"  Values: {eq_bern}")

# Verify the relationship: eq_bern = bern_x - bern_y
print("\n--- Verification: eq_bern = bern_x - bern_y ---")
bern_x = hypersurface_coeffs[0]
bern_y = hypersurface_coeffs[1]
eq_bern_computed = bern_x - bern_y
eq_bern_from_system = equation_coeffs[0]
difference = np.max(np.abs(eq_bern_computed - eq_bern_from_system))
print(f"Max difference: {difference:.2e}")
print(f"Match: {np.allclose(eq_bern_computed, eq_bern_from_system)}")

print("\n" + "=" * 80)
print("METHOD 3: Get Equation Coefficients Using Helper Function")
print("=" * 80)

# Alternative: use the helper function
eq_coeffs_alt = get_equation_bernstein_coeffs(system)

print(f"\nUsing get_equation_bernstein_coeffs():")
print(f"Number of equations: {len(eq_coeffs_alt)}")
print(f"Same as direct access: {all(np.array_equal(a, b) for a, b in zip(equation_coeffs, eq_coeffs_alt))}")

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

# Get hypersurface Bernstein coefficients
print("\n--- Hypersurface Bernstein Coefficients ---")
hypersurface_coeffs_3d = system_3d['hypersurface_bernstein_coeffs']

print(f"Number of coordinates: {len(hypersurface_coeffs_3d)}")

for i, bern in enumerate(hypersurface_coeffs_3d):
    print(f"\nCoordinate {i+1} (x{i+1}):")
    print(f"  Shape: {bern.shape}")
    print(f"  Number of coefficients: {bern.size}")
    if bern.ndim == 1:
        print(f"  Values: {bern}")
    else:
        print(f"  Values (2D array):")
        print(f"{bern}")

# Get equation Bernstein coefficients
print("\n--- Equation Bernstein Coefficients ---")
equation_coeffs_3d = system_3d['equation_bernstein_coeffs']

print(f"Number of equations: {len(equation_coeffs_3d)}")

for i, eq_bern in enumerate(equation_coeffs_3d):
    print(f"\nEquation {i+1}:")
    print(f"  Shape: {eq_bern.shape}")
    print(f"  Number of coefficients: {eq_bern.size}")
    print(f"  Values (2D array):")
    print(f"{eq_bern}")

# Verify the relationships
print("\n--- Verification ---")
bern_x = hypersurface_coeffs_3d[0]
bern_y = hypersurface_coeffs_3d[1]
eq1_computed = bern_x - 0.5
eq2_computed = bern_y - 0.5
eq1_from_system = equation_coeffs_3d[0]
eq2_from_system = equation_coeffs_3d[1]

diff1 = np.max(np.abs(eq1_computed - eq1_from_system))
diff2 = np.max(np.abs(eq2_computed - eq2_from_system))

print(f"Equation 1: eq_bern = bern_x - 0.5")
print(f"  Max difference: {diff1:.2e}")
print(f"  Match: {np.allclose(eq1_computed, eq1_from_system)}")

print(f"\nEquation 2: eq_bern = bern_y - 0.5")
print(f"  Max difference: {diff2:.2e}")
print(f"  Match: {np.allclose(eq2_computed, eq2_from_system)}")

print("\n" + "=" * 80)
print("UNDERSTANDING THE STRUCTURE")
print("=" * 80)

print("""
For a k-dimensional hypersurface in n-dimensional space:

TWO TYPES OF BERNSTEIN COEFFICIENTS:

1. HYPERSURFACE BERNSTEIN COEFFICIENTS (Coordinate Functions)
   - system['hypersurface_bernstein_coeffs'] is a LIST of n arrays
   - One array for each coordinate function (x1(u), x2(u), ..., xn(u))
   - Each array has shape depending on k (number of parameters):
     * k=1 (curve):    shape = (degree+1,)           [1D array]
     * k=2 (surface):  shape = (degree+1, degree+1)  [2D array]
     * k=3:            shape = (degree+1, degree+1, degree+1)  [3D array]

2. EQUATION BERNSTEIN COEFFICIENTS (Polynomial Equations)
   - system['equation_bernstein_coeffs'] is a LIST of k arrays
   - One array for each equation (after applying hyperplane constraints)
   - Each array has the SAME shape as hypersurface coefficients
   - Computed as linear combinations: sum(a_ij * bern_xj) + d_i

EXAMPLES:

For the 2D circle example (k=1, n=2):
  - hypersurface_bernstein_coeffs[0] = x(u) coefficients, shape (6,) for degree 5
  - hypersurface_bernstein_coeffs[1] = y(u) coefficients, shape (6,) for degree 5
  - equation_bernstein_coeffs[0] = (x(u) - y(u)) coefficients, shape (6,)

For the 3D surface example (k=2, n=3):
  - hypersurface_bernstein_coeffs[0] = x(u,v) coefficients, shape (5,5) for degree 4
  - hypersurface_bernstein_coeffs[1] = y(u,v) coefficients, shape (5,5) for degree 4
  - hypersurface_bernstein_coeffs[2] = z(u,v) coefficients, shape (5,5) for degree 4
  - equation_bernstein_coeffs[0] = (x(u,v) - 0.5) coefficients, shape (5,5)
  - equation_bernstein_coeffs[1] = (y(u,v) - 0.5) coefficients, shape (5,5)
""")

print("\n" + "=" * 80)
print("QUICK REFERENCE")
print("=" * 80)

print("""
# Get hypersurface Bernstein coefficients (coordinate functions):
hypersurface_coeffs = system['hypersurface_bernstein_coeffs']

# Get equation Bernstein coefficients (polynomial equations):
equation_coeffs = system['equation_bernstein_coeffs']

# Or use helper function:
from src.intersection.polynomial_system import get_equation_bernstein_coeffs
equation_coeffs = get_equation_bernstein_coeffs(system)

# For 2D (curve in 2D space):
bern_x = system['hypersurface_bernstein_coeffs'][0]  # shape: (degree+1,)
bern_y = system['hypersurface_bernstein_coeffs'][1]  # shape: (degree+1,)
eq_bern = system['equation_bernstein_coeffs'][0]     # shape: (degree+1,)

# For 3D (surface in 3D space):
bern_x = system['hypersurface_bernstein_coeffs'][0]  # shape: (degree+1, degree+1)
bern_y = system['hypersurface_bernstein_coeffs'][1]  # shape: (degree+1, degree+1)
bern_z = system['hypersurface_bernstein_coeffs'][2]  # shape: (degree+1, degree+1)
eq1_bern = system['equation_bernstein_coeffs'][0]    # shape: (degree+1, degree+1)
eq2_bern = system['equation_bernstein_coeffs'][1]    # shape: (degree+1, degree+1)

# Alternative: Get from hypersurface directly
hypersurface_coeffs = system['hypersurface'].bernstein_coeffs

# Get other useful information:
n = system['n']                    # Ambient dimension
k = system['k']                    # Number of parameters
degree = system['degree']          # Polynomial degree
param_ranges = system['param_ranges']  # Parameter ranges
equations = system['equations']    # List of equation specifications
""")

print("\n" + "=" * 80)
print("Example Complete!")
print("=" * 80)

