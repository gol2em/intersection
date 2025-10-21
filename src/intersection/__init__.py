"""
Intersection Computation Library

Compute intersections of straight lines with parametric hypersurfaces in n-dimensional space.
"""

from .geometry import Hyperplane, Line, Hypersurface
from .interpolation import interpolate_hypersurface
from .bernstein import polynomial_nd_to_bernstein
from .polynomial_system import (
    create_intersection_system,
    evaluate_system,
    get_equation_bernstein_coeffs,
)

__version__ = "0.2.0"

__all__ = [
    "Hyperplane",
    "Line",
    "Hypersurface",
    "interpolate_hypersurface",
    "polynomial_nd_to_bernstein",
    "create_intersection_system",
    "evaluate_system",
    "get_equation_bernstein_coeffs",
]

