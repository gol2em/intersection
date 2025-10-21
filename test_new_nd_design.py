"""
Test the new n-dimensional design
"""

import numpy as np
from src.intersection.geometry import Hyperplane, Line, Hypersurface

print("=" * 70)
print("Testing n-Dimensional Intersection Framework")
print("=" * 70)

# Test 1: Hyperplane in 2D
print("\n### Test 1: Hyperplane in 2D ###")
h1 = Hyperplane(coeffs=[1, -1], d=0)  # x - y = 0
print(h1)

# Test 2: Line in 2D (1 hyperplane)
print("\n### Test 2: Line in 2D (1 hyperplane) ###")
line_2d = Line([h1])
print(line_2d)

# Test 3: Line in 3D (2 hyperplanes)
print("\n### Test 3: Line in 3D (2 hyperplanes) ###")
h2 = Hyperplane(coeffs=[1, 0, 0], d=-1)  # x = 1
h3 = Hyperplane(coeffs=[0, 1, 0], d=-2)  # y = 2
line_3d = Line([h2, h3])
print(line_3d)

# Test 4: Hypersurface - 2D Curve (1 param → 2D)
print("\n### Test 4: Hypersurface - 2D Curve (1→2D) ###")
curve = Hypersurface(
    func=lambda u: np.array([u, u**2]),
    param_ranges=[(0, 1)],
    ambient_dim=2,
    degree=3,
    verbose=True
)
print(f"\n{curve}")
print(f"Evaluate at u=0.5: {curve.evaluate(0.5)}")

# Test 5: Hypersurface - 3D Surface (2 params → 3D)
print("\n### Test 5: Hypersurface - 3D Surface (2→3D) ###")
surface = Hypersurface(
    func=lambda u, v: np.array([u, v, u**2 + v**2]),
    param_ranges=[(0, 1), (0, 1)],
    ambient_dim=3,
    degree=3,
    verbose=True
)
print(f"\n{surface}")
print(f"Evaluate at (u,v)=(0.5,0.5): {surface.evaluate(0.5, 0.5)}")

# Test 6: Hypersurface - 4D "Surface" (3 params → 4D)
print("\n### Test 6: Hypersurface - 4D Hypersurface (3→4D) ###")
hypersurface_4d = Hypersurface(
    func=lambda u, v, w: np.array([u, v, w, u*v*w]),
    param_ranges=[(0, 1), (0, 1), (0, 1)],
    ambient_dim=4,
    degree=2,
    verbose=True
)
print(f"\n{hypersurface_4d}")
print(f"Evaluate at (u,v,w)=(0.5,0.5,0.5): {hypersurface_4d.evaluate(0.5, 0.5, 0.5)}")

# Test 7: Line in 4D (3 hyperplanes)
print("\n### Test 7: Line in 4D (3 hyperplanes) ###")
h4 = Hyperplane(coeffs=[1, 0, 0, 0], d=0)  # x1 = 0
h5 = Hyperplane(coeffs=[0, 1, 0, 0], d=0)  # x2 = 0
h6 = Hyperplane(coeffs=[0, 0, 1, 0], d=0)  # x3 = 0
line_4d = Line([h4, h5, h6])
print(line_4d)

print("\n" + "=" * 70)
print("All tests completed successfully!")
print("=" * 70)

