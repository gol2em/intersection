"""
Convex hull utilities for PP method.

This module provides functions to compute intersections of convex hulls
with coordinate axes, which is essential for the PP (Projected Polyhedron) method.
"""

import numpy as np
from typing import List, Tuple, Optional


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """
    Compute the convex hull of a set of 2D points using Graham scan.
    
    Args:
        points: Array of shape (n, 2) containing 2D points
        
    Returns:
        Array of shape (m, 2) containing vertices of convex hull in counter-clockwise order
        
    Example:
        >>> points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])
        >>> hull = convex_hull_2d(points)
        >>> hull.shape
        (4, 2)
    """
    if len(points) < 3:
        return points
    
    # Remove duplicate points
    points = np.unique(points, axis=0)
    
    if len(points) < 3:
        return points
    
    def cross_product(o, a, b):
        """Compute cross product of vectors OA and OB."""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Sort points lexicographically (first by x, then by y)
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    
    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    # Remove last point of each half because it's repeated
    hull = np.array(lower[:-1] + upper[:-1])
    
    return hull


def intersect_convex_hull_with_x_axis(points: np.ndarray, tolerance: float = 1e-10) -> Optional[Tuple[float, float]]:
    """
    Find the intersection of the convex hull of 2D points with the x-axis (y=0).
    
    This is a key operation for the PP method: given Bernstein coefficients
    as 2D points (parameter_value, function_value), find the range of parameter
    values where the function could be zero.
    
    Args:
        points: Array of shape (n, 2) where points[:, 0] are x-coordinates (parameters)
                and points[:, 1] are y-coordinates (function values)
        tolerance: Numerical tolerance for considering a point on the x-axis
        
    Returns:
        Tuple (x_min, x_max) representing the x-range where the convex hull
        intersects the x-axis, or None if no intersection exists.
        
    Algorithm:
        1. Compute convex hull of points
        2. For each edge of the hull, check if it crosses y=0
        3. If it crosses, compute the x-coordinate of the intersection
        4. Return the min and max x-coordinates of all intersections
        
    Example:
        >>> # Points forming a triangle that crosses x-axis
        >>> points = np.array([[0, 1], [0.5, -1], [1, 1]])
        >>> x_min, x_max = intersect_convex_hull_with_x_axis(points)
        >>> print(f"Intersection range: [{x_min:.3f}, {x_max:.3f}]")
        Intersection range: [0.250, 0.750]
        
    Use case in PP method:
        >>> # Bernstein coefficients for 1D polynomial
        >>> # Points are (t_i, b_i) where t_i = i/n and b_i are coefficients
        >>> n = 3
        >>> coeffs = np.array([1.0, -0.5, -0.5, 1.0])
        >>> t_values = np.linspace(0, 1, len(coeffs))
        >>> points = np.column_stack([t_values, coeffs])
        >>> result = intersect_convex_hull_with_x_axis(points)
        >>> if result:
        >>>     t_min, t_max = result
        >>>     print(f"Roots must be in [{t_min}, {t_max}]")
    """
    if len(points) < 2:
        return None
    
    # Compute convex hull
    hull = convex_hull_2d(points)
    
    if len(hull) < 2:
        return None
    
    # Check if all points are on one side of x-axis
    y_values = hull[:, 1]
    if np.all(y_values > tolerance) or np.all(y_values < -tolerance):
        # No intersection with x-axis
        return None
    
    # Find all intersections with x-axis
    x_intersections = []
    
    # Check each edge of the hull
    n_vertices = len(hull)
    for i in range(n_vertices):
        p1 = hull[i]
        p2 = hull[(i + 1) % n_vertices]
        
        x1, y1 = p1
        x2, y2 = p2
        
        # Check if point is on x-axis
        if abs(y1) <= tolerance:
            x_intersections.append(x1)
        
        # Check if edge crosses x-axis
        if (y1 > tolerance and y2 < -tolerance) or (y1 < -tolerance and y2 > tolerance):
            # Edge crosses x-axis, compute intersection point
            # Line equation: y = y1 + (y2 - y1) * t, where t = (x - x1) / (x2 - x1)
            # Set y = 0 and solve for x:
            # 0 = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
            # x = x1 - y1 * (x2 - x1) / (y2 - y1)
            
            if abs(y2 - y1) > tolerance:
                x_intersection = x1 - y1 * (x2 - x1) / (y2 - y1)
                x_intersections.append(x_intersection)
    
    if not x_intersections:
        # Check if entire hull is on x-axis
        if np.all(np.abs(y_values) <= tolerance):
            # All points on x-axis, return full x-range
            return (np.min(hull[:, 0]), np.max(hull[:, 0]))
        return None
    
    # Return the range of x-intersections
    x_min = np.min(x_intersections)
    x_max = np.max(x_intersections)
    
    return (x_min, x_max)


def intersect_convex_hull_with_axis_nd(points: np.ndarray, axis: int = 0, 
                                        tolerance: float = 1e-10) -> Optional[Tuple[float, float]]:
    """
    Find the intersection of the convex hull with a coordinate axis in n-D.
    
    For n-dimensional case, this projects the convex hull onto a 2D plane
    (parameter_axis, function_value) and finds the intersection.
    
    Args:
        points: Array of shape (n, d) where d >= 2
                points[:, :-1] are parameter coordinates
                points[:, -1] are function values
        axis: Which parameter axis to project onto (0 to d-2)
        tolerance: Numerical tolerance
        
    Returns:
        Tuple (param_min, param_max) for the given axis, or None if no intersection
        
    Example:
        >>> # 2D parameters (u, v) with function values
        >>> # Points are (u, v, f(u,v))
        >>> points = np.array([
        ...     [0, 0, 1],
        ...     [0.5, 0.5, -1],
        ...     [1, 1, 1]
        ... ])
        >>> # Project onto u-axis
        >>> result = intersect_convex_hull_with_axis_nd(points, axis=0)
    """
    if points.shape[1] < 2:
        raise ValueError("Points must have at least 2 dimensions")
    
    # Extract the parameter axis and function values
    param_values = points[:, axis]
    function_values = points[:, -1]
    
    # Create 2D points for convex hull computation
    points_2d = np.column_stack([param_values, function_values])
    
    # Use 2D intersection function
    return intersect_convex_hull_with_x_axis(points_2d, tolerance)


def find_root_box_pp_1d(coeffs: np.ndarray, tolerance: float = 1e-10) -> Optional[Tuple[float, float]]:
    """
    Find the tightest box containing all roots using PP method for 1D polynomial.

    This uses the convex hull of Bernstein control points to find a tighter
    bound than just checking if 0 is in [min(coeffs), max(coeffs)].

    Args:
        coeffs: 1D array of Bernstein coefficients
        tolerance: Numerical tolerance

    Returns:
        Tuple (t_min, t_max) where roots must lie, or None if no roots exist

    Example:
        >>> # Polynomial with roots around 0.3 and 0.7
        >>> coeffs = np.array([0.21, -0.1, -0.1, 0.21])
        >>> result = find_root_box_pp_1d(coeffs)
        >>> if result:
        ...     print(f"Roots in [{result[0]:.3f}, {result[1]:.3f}]")
    """
    # Quick check: if all coefficients have same sign, no root
    if np.all(coeffs > tolerance) or np.all(coeffs < -tolerance):
        return None

    # Create control points: (t_i, b_i) where t_i = i / (n-1)
    n = len(coeffs)
    if n == 1:
        # Single coefficient
        if abs(coeffs[0]) <= tolerance:
            return (0.0, 1.0)
        else:
            return None

    t_values = np.linspace(0, 1, n)
    points = np.column_stack([t_values, coeffs])

    # Find intersection with x-axis
    return intersect_convex_hull_with_x_axis(points, tolerance)


def find_root_box_pp_nd(equation_coeffs_list: List[np.ndarray],
                        k: int,
                        tolerance: float = 1e-10) -> Optional[List[Tuple[float, float]]]:
    """
    Find the tightest bounding box containing all roots using PP method for k-D system.

    For each dimension j (j=0, 1, ..., k-1):
    1. Extract the j-th component from each equation (univariate polynomial in dimension j)
    2. Compute convex hull intersection with x-axis for each equation
    3. Intersect all ranges to get the tightest bound for dimension j
    4. Construct the k-dimensional bounding box

    Args:
        equation_coeffs_list: List of Bernstein coefficient arrays, one per equation
                             Each array has shape (n0+1, n1+1, ..., n_{k-1}+1)
        k: Number of dimensions (parameters)
        tolerance: Numerical tolerance

    Returns:
        List of k tuples [(t0_min, t0_max), (t1_min, t1_max), ...] representing
        the bounding box, or None if no roots exist

    Example:
        >>> # 2D system with 2 equations
        >>> # Each equation has shape (5, 5) for degree 4 in both dimensions
        >>> eq1_coeffs = np.random.randn(5, 5)
        >>> eq2_coeffs = np.random.randn(5, 5)
        >>> result = find_root_box_pp_nd([eq1_coeffs, eq2_coeffs], k=2)
        >>> if result:
        ...     print(f"Roots in [{result[0][0]:.3f}, {result[0][1]:.3f}] x "
        ...           f"[{result[1][0]:.3f}, {result[1][1]:.3f}]")
    """
    if not equation_coeffs_list:
        return None

    # Initialize bounding box as full [0,1]^k
    bounding_box = [(0.0, 1.0) for _ in range(k)]

    # For each dimension
    for dim in range(k):
        # Collect all ranges from all equations for this dimension
        dim_ranges = []

        # For each equation
        for eq_coeffs in equation_coeffs_list:
            # Extract univariate polynomial for this dimension
            # by fixing all other dimensions and extracting slices
            univariate_range = _extract_dimension_range(eq_coeffs, dim, k, tolerance)

            if univariate_range is None:
                # This equation has no roots in this dimension
                return None

            dim_ranges.append(univariate_range)

        # Intersect all ranges for this dimension
        if not dim_ranges:
            return None

        # Find intersection of all ranges
        t_min = max(r[0] for r in dim_ranges)
        t_max = min(r[1] for r in dim_ranges)

        if t_min > t_max + tolerance:
            # No intersection - no roots
            return None

        bounding_box[dim] = (t_min, t_max)

    return bounding_box


def _extract_dimension_range(coeffs: np.ndarray,
                             dim: int,
                             k: int,
                             tolerance: float = 1e-10) -> Optional[Tuple[float, float]]:
    """
    Extract the range for a specific dimension from k-D Bernstein coefficients.

    This implements the correct projection method from the paper:
    For dimension j, we project all control points onto the 2D plane (t_j, f) where:
    - t_j is the j-th parameter coordinate
    - f is the function value

    Then we compute the 2D convex hull and intersect with the t_j-axis (f=0).

    Args:
        coeffs: k-dimensional array of Bernstein coefficients
        dim: Dimension to extract (0 to k-1)
        k: Total number of dimensions
        tolerance: Numerical tolerance

    Returns:
        Tuple (t_min, t_max) for this dimension, or None if no roots possible
    """
    if k == 1:
        # 1D case - direct computation
        return find_root_box_pp_1d(coeffs, tolerance)

    # Quick check: if all coefficients have same sign, no root
    overall_min = np.min(coeffs)
    overall_max = np.max(coeffs)

    if overall_min > tolerance or overall_max < -tolerance:
        # Polynomial cannot be zero anywhere
        return None

    # Multi-dimensional case: project all control points onto (t_dim, f) plane
    # Get the shape of the coefficient array
    shape = coeffs.shape

    # Generate all multi-indices for the control points
    # For a 2D array of shape (n0, n1), we get indices like (0,0), (0,1), ..., (n0-1, n1-1)
    indices = np.ndindex(*shape)

    # Collect projected 2D points (t_dim, f_value)
    projected_points = []

    for multi_idx in indices:
        # Get the function value at this control point
        f_value = coeffs[multi_idx]

        # Get the parameter value for dimension 'dim'
        # For Bernstein polynomials, the i-th control point in dimension j
        # corresponds to parameter value i / (n_j - 1)
        i_dim = multi_idx[dim]
        n_dim = shape[dim]

        if n_dim == 1:
            t_dim = 0.5  # Single point, use midpoint
        else:
            t_dim = i_dim / (n_dim - 1)

        projected_points.append([t_dim, f_value])

    # Convert to numpy array
    projected_points = np.array(projected_points)

    # Compute 2D convex hull and intersect with t-axis (f=0)
    return intersect_convex_hull_with_x_axis(projected_points, tolerance)

