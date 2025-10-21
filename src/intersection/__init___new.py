"""
Intersection Computation Library

Compute intersections of straight lines with parametric hypersurfaces in n-dimensional space.
"""

from .geometry import Hyperplane, Line, Hypersurface
from .interpolation import interpolate_hypersurface
from .bernstein import polynomial_nd_to_bernstein

__version__ = "0.2.0"

__all__ = [
    "Hyperplane",
    "Line",
    "Hypersurface",
    "interpolate_hypersurface",
    "polynomial_nd_to_bernstein",
]

