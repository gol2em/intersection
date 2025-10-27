"""
de Casteljau Algorithm for Bernstein Polynomial Subdivision

Implements the de Casteljau algorithm for subdividing Bernstein polynomials.
The algorithm automatically renormalizes coefficients to [0,1]^k, which is
perfect for LP/PP methods.

Key Properties:
--------------
1. Subdivision preserves the polynomial (exact, no approximation)
2. Subdivided coefficients are automatically on [0,1]^k
3. Numerically stable (only convex combinations)
4. Works in any dimension (1D, 2D, kD)

Domain Tracking:
---------------
The Box class tracks the domain transformations:
- Parent box: [a, b] in normalized space
- Subdivide at t ∈ [0, 1] (relative to parent box)
- Left child: [a, a + t*(b-a)]
- Right child: [a + t*(b-a), b]
- de Casteljau gives coefficients on [0, 1] for each child
"""

import numpy as np
from typing import Tuple, List
from .box import Box


def de_casteljau_eval_1d(coeffs: np.ndarray, t: float) -> float:
    """
    Evaluate 1D Bernstein polynomial at parameter t using de Casteljau algorithm.
    
    Parameters
    ----------
    coeffs : np.ndarray
        Bernstein coefficients, shape (n+1,)
    t : float
        Parameter value in [0, 1]
        
    Returns
    -------
    float
        Polynomial value at t
        
    Examples
    --------
    >>> # p(t) = t with degree 1: coeffs = [0, 1]
    >>> de_casteljau_eval_1d(np.array([0, 1]), 0.5)
    0.5
    
    >>> # p(t) = t^2 with degree 2: coeffs = [0, 0, 1]
    >>> de_casteljau_eval_1d(np.array([0, 0, 1]), 0.5)
    0.25
    """
    n = len(coeffs) - 1
    
    # Build de Casteljau pyramid
    b = coeffs.copy()
    
    for j in range(1, n + 1):
        for i in range(n - j + 1):
            b[i] = (1 - t) * b[i] + t * b[i + 1]
    
    return b[0]


def de_casteljau_subdivide_1d(coeffs: np.ndarray, t: float = 0.5, 
                               verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subdivide 1D Bernstein polynomial at parameter t using de Casteljau algorithm.
    
    The algorithm computes Bernstein coefficients for the two sub-polynomials:
    - Left:  p(s) for s ∈ [0, t], renormalized to [0, 1]
    - Right: p(s) for s ∈ [t, 1], renormalized to [0, 1]
    
    Parameters
    ----------
    coeffs : np.ndarray
        Bernstein coefficients of parent polynomial, shape (n+1,)
    t : float
        Subdivision point in [0, 1] (default: 0.5 for midpoint)
    verbose : bool
        If True, print subdivision details
        
    Returns
    -------
    left_coeffs : np.ndarray
        Bernstein coefficients for left sub-polynomial on [0, 1]
    right_coeffs : np.ndarray
        Bernstein coefficients for right sub-polynomial on [0, 1]
        
    Notes
    -----
    The de Casteljau pyramid gives us both sets of coefficients:
    - Left coefficients: First column of pyramid
    - Right coefficients: Diagonal of pyramid
    
    Examples
    --------
    >>> # Subdivide p(t) = t at midpoint
    >>> coeffs = np.array([0, 1])
    >>> left, right = de_casteljau_subdivide_1d(coeffs, 0.5)
    >>> left   # [0, 0.5] renormalized to [0, 1] → [0, 1]
    array([0., 0.5])
    >>> right  # [0.5, 1] renormalized to [0, 1] → [0, 1]
    array([0.5, 1.])
    """
    n = len(coeffs) - 1
    
    # Build de Casteljau pyramid
    # pyramid[j][i] = b_i^j
    pyramid = [coeffs.copy()]
    
    for j in range(1, n + 1):
        level = np.zeros(n - j + 1)
        for i in range(n - j + 1):
            level[i] = (1 - t) * pyramid[j-1][i] + t * pyramid[j-1][i + 1]
        pyramid.append(level)
    
    # Extract coefficients
    left_coeffs = np.array([pyramid[j][0] for j in range(n + 1)])
    right_coeffs = np.array([pyramid[n - j][j] for j in range(n + 1)])
    
    if verbose:
        print(f"\n=== de Casteljau Subdivision (1D) ===")
        print(f"Degree: {n}")
        print(f"Subdivision point: t = {t}")
        print(f"Parent coefficients: {coeffs}")
        print(f"Left coefficients:   {left_coeffs}")
        print(f"Right coefficients:  {right_coeffs}")
        
        # Verify by evaluation
        left_val = de_casteljau_eval_1d(left_coeffs, 1.0)
        right_val = de_casteljau_eval_1d(right_coeffs, 0.0)
        parent_val = de_casteljau_eval_1d(coeffs, t)
        print(f"\nVerification at subdivision point:")
        print(f"  Parent at t={t}: {parent_val}")
        print(f"  Left at 1.0:     {left_val}")
        print(f"  Right at 0.0:    {right_val}")
        print(f"  Match: {np.allclose([left_val, right_val], parent_val)}")
    
    return left_coeffs, right_coeffs


def de_casteljau_subdivide_2d(coeffs: np.ndarray, axis: int, t: float = 0.5,
                               verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subdivide 2D tensor product Bernstein polynomial along one axis.
    
    For a 2D polynomial p(u, v) with coefficients C[i, j], subdividing along
    axis 0 (u-direction) gives two polynomials with coefficients on [0,1]^2.
    
    Parameters
    ----------
    coeffs : np.ndarray
        2D array of Bernstein coefficients, shape (n+1, m+1)
    axis : int
        Axis to subdivide (0 for u, 1 for v)
    t : float
        Subdivision point in [0, 1]
    verbose : bool
        If True, print subdivision details
        
    Returns
    -------
    left_coeffs : np.ndarray
        Bernstein coefficients for left sub-polynomial
    right_coeffs : np.ndarray
        Bernstein coefficients for right sub-polynomial
        
    Examples
    --------
    >>> # Subdivide p(u,v) = u along u-axis
    >>> coeffs = np.array([[0, 0], [1, 1]])  # degree (1, 1)
    >>> left, right = de_casteljau_subdivide_2d(coeffs, axis=0, t=0.5)
    >>> left   # u ∈ [0, 0.5] → coeffs for [0, 1]
    array([[0., 0.], [0.5, 0.5]])
    >>> right  # u ∈ [0.5, 1] → coeffs for [0, 1]
    array([[0.5, 0.5], [1., 1.]])
    """
    if coeffs.ndim != 2:
        raise ValueError(f"Expected 2D array, got {coeffs.ndim}D")
    
    n_u, n_v = coeffs.shape
    
    if axis == 0:
        # Subdivide along u-direction
        # Apply 1D subdivision to each v-slice
        left_coeffs = np.zeros_like(coeffs)
        right_coeffs = np.zeros_like(coeffs)
        
        for j in range(n_v):
            left_coeffs[:, j], right_coeffs[:, j] = de_casteljau_subdivide_1d(
                coeffs[:, j], t, verbose=False
            )
    else:
        # Subdivide along v-direction
        # Apply 1D subdivision to each u-slice
        left_coeffs = np.zeros_like(coeffs)
        right_coeffs = np.zeros_like(coeffs)
        
        for i in range(n_u):
            left_coeffs[i, :], right_coeffs[i, :] = de_casteljau_subdivide_1d(
                coeffs[i, :], t, verbose=False
            )
    
    if verbose:
        print(f"\n=== de Casteljau Subdivision (2D) ===")
        print(f"Coefficient shape: {coeffs.shape}")
        print(f"Subdivision axis: {axis} ({'u' if axis == 0 else 'v'})")
        print(f"Subdivision point: t = {t}")
        print(f"Parent coefficients:\n{coeffs}")
        print(f"Left coefficients:\n{left_coeffs}")
        print(f"Right coefficients:\n{right_coeffs}")
    
    return left_coeffs, right_coeffs


def de_casteljau_subdivide_kd(coeffs: np.ndarray, axis: int, t: float = 0.5,
                               verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subdivide k-dimensional tensor product Bernstein polynomial along one axis.
    
    Parameters
    ----------
    coeffs : np.ndarray
        k-D array of Bernstein coefficients
    axis : int
        Axis to subdivide (0 to k-1)
    t : float
        Subdivision point in [0, 1]
    verbose : bool
        If True, print subdivision details
        
    Returns
    -------
    left_coeffs : np.ndarray
        Bernstein coefficients for left sub-polynomial
    right_coeffs : np.ndarray
        Bernstein coefficients for right sub-polynomial
    """
    k = coeffs.ndim
    
    if axis < 0 or axis >= k:
        raise ValueError(f"Invalid axis {axis} for {k}D array")
    
    # Move the subdivision axis to the last position
    coeffs_moved = np.moveaxis(coeffs, axis, -1)
    shape = coeffs_moved.shape
    n_last = shape[-1]
    n_rest = np.prod(shape[:-1])
    
    # Reshape to 2D for easier processing
    coeffs_2d = coeffs_moved.reshape(n_rest, n_last)
    
    # Apply 1D subdivision to each slice
    left_2d = np.zeros_like(coeffs_2d)
    right_2d = np.zeros_like(coeffs_2d)
    
    for i in range(n_rest):
        left_2d[i, :], right_2d[i, :] = de_casteljau_subdivide_1d(
            coeffs_2d[i, :], t, verbose=False
        )
    
    # Reshape back to original shape
    left_moved = left_2d.reshape(shape)
    right_moved = right_2d.reshape(shape)
    
    # Move axis back to original position
    left_coeffs = np.moveaxis(left_moved, -1, axis)
    right_coeffs = np.moveaxis(right_moved, -1, axis)
    
    if verbose:
        print(f"\n=== de Casteljau Subdivision ({k}D) ===")
        print(f"Coefficient shape: {coeffs.shape}")
        print(f"Subdivision axis: {axis}")
        print(f"Subdivision point: t = {t}")
    
    return left_coeffs, right_coeffs


def extract_subbox_1d(coeffs: np.ndarray, t_min: float, t_max: float,
                      tolerance: float = 1e-10, verbose: bool = False) -> np.ndarray:
    """
    Extract Bernstein coefficients for a sub-interval [t_min, t_max] ⊂ [0, 1].

    Uses de Casteljau subdivision twice:
    1. Split at t_min to get polynomial on [t_min, 1]
    2. Split that at (t_max - t_min) / (1 - t_min) to get [t_min, t_max]

    The result is renormalized to [0, 1].

    Parameters
    ----------
    coeffs : np.ndarray
        Bernstein coefficients on [0, 1], shape (n+1,)
    t_min : float
        Start of sub-interval in [0, 1]
    t_max : float
        End of sub-interval in [0, 1]
    tolerance : float
        Tolerance for boundary comparison (default: 1e-10)
        If |t_min - 0| < tolerance and |t_max - 1| < tolerance, return original coefficients
    verbose : bool
        If True, print extraction details

    Returns
    -------
    np.ndarray
        Bernstein coefficients for sub-interval, renormalized to [0, 1]

    Examples
    --------
    >>> # Extract p(t) = t on [0.25, 0.75]
    >>> coeffs = np.array([0.0, 1.0])
    >>> sub_coeffs = extract_subbox_1d(coeffs, 0.25, 0.75)
    >>> # Result: [0.25, 0.75] renormalized to [0, 1]
    >>> sub_coeffs
    array([0.25, 0.75])

    >>> # Near-boundary case: [1e-12, 1.0 - 1e-12] ≈ [0, 1]
    >>> sub_coeffs = extract_subbox_1d(coeffs, 1e-12, 1.0 - 1e-12)
    >>> # Returns original coefficients (no subdivision)
    """
    if t_min >= t_max:
        raise ValueError(f"Invalid interval: t_min={t_min} >= t_max={t_max}")

    if t_min < -tolerance or t_max > 1 + tolerance:
        raise ValueError(f"Interval [{t_min}, {t_max}] must be in [0, 1]")

    # Clamp to [0, 1]
    t_min = max(0.0, min(1.0, t_min))
    t_max = max(0.0, min(1.0, t_max))

    # Special case: full interval (within tolerance)
    if abs(t_min) < tolerance and abs(t_max - 1.0) < tolerance:
        if verbose:
            print(f"\n=== Extract Sub-box (1D) ===")
            print(f"Sub-interval [{t_min}, {t_max}] is within tolerance of [0, 1]")
            print(f"Returning original coefficients (no subdivision)")
        return coeffs.copy()

    # Step 1: Split at t_min to get [t_min, 1]
    if abs(t_min) < tolerance:
        right_coeffs = coeffs.copy()
    else:
        _, right_coeffs = de_casteljau_subdivide_1d(coeffs, t_min, verbose=False)

    # Step 2: Split at relative position to get [t_min, t_max]
    if abs(t_max - 1.0) < tolerance:
        result = right_coeffs
    else:
        # Map t_max to position in [t_min, 1]
        t_relative = (t_max - t_min) / (1.0 - t_min)
        result, _ = de_casteljau_subdivide_1d(right_coeffs, t_relative, verbose=False)

    if verbose:
        print(f"\n=== Extract Sub-box (1D) ===")
        print(f"Original interval: [0, 1]")
        print(f"Sub-interval: [{t_min}, {t_max}]")
        print(f"Original coefficients: {coeffs}")
        print(f"Sub-box coefficients:  {result}")

    return result


def extract_subbox_2d(coeffs: np.ndarray, ranges: List[Tuple[float, float]],
                      tolerance: float = 1e-10, verbose: bool = False) -> np.ndarray:
    """
    Extract Bernstein coefficients for a sub-box in [0, 1]^2.

    Parameters
    ----------
    coeffs : np.ndarray
        2D Bernstein coefficients on [0, 1]^2, shape (n+1, m+1)
    ranges : List[Tuple[float, float]]
        Sub-box ranges: [(u_min, u_max), (v_min, v_max)]
    tolerance : float
        Tolerance for boundary comparison (default: 1e-10)
    verbose : bool
        If True, print extraction details

    Returns
    -------
    np.ndarray
        Bernstein coefficients for sub-box, renormalized to [0, 1]^2

    Examples
    --------
    >>> # Extract p(u,v) = u on [0.25, 0.75] × [0, 1]
    >>> coeffs = np.array([[0, 0], [1, 1]])
    >>> sub_coeffs = extract_subbox_2d(coeffs, [(0.25, 0.75), (0, 1)])
    """
    if coeffs.ndim != 2:
        raise ValueError(f"Expected 2D array, got {coeffs.ndim}D")

    if len(ranges) != 2:
        raise ValueError(f"Expected 2 ranges for 2D, got {len(ranges)}")

    n_u, n_v = coeffs.shape

    # Extract along u-direction
    u_min, u_max = ranges[0]
    result = np.zeros_like(coeffs)

    for j in range(n_v):
        result[:, j] = extract_subbox_1d(coeffs[:, j], u_min, u_max, tolerance, verbose=False)

    # Extract along v-direction
    v_min, v_max = ranges[1]
    for i in range(n_u):
        result[i, :] = extract_subbox_1d(result[i, :], v_min, v_max, tolerance, verbose=False)

    if verbose:
        print(f"\n=== Extract Sub-box (2D) ===")
        print(f"Original box: [0, 1]^2")
        print(f"Sub-box: [{u_min}, {u_max}] × [{v_min}, {v_max}]")
        print(f"Original coefficients:\n{coeffs}")
        print(f"Sub-box coefficients:\n{result}")

    return result


def extract_subbox_kd(coeffs: np.ndarray, ranges: List[Tuple[float, float]],
                      tolerance: float = 1e-10, verbose: bool = False) -> np.ndarray:
    """
    Extract Bernstein coefficients for a sub-box in [0, 1]^k.

    Parameters
    ----------
    coeffs : np.ndarray
        k-D Bernstein coefficients on [0, 1]^k
    ranges : List[Tuple[float, float]]
        Sub-box ranges: [(t1_min, t1_max), ..., (tk_min, tk_max)]
    tolerance : float
        Tolerance for boundary comparison (default: 1e-10)
    verbose : bool
        If True, print extraction details

    Returns
    -------
    np.ndarray
        Bernstein coefficients for sub-box, renormalized to [0, 1]^k
    """
    k = coeffs.ndim

    if len(ranges) != k:
        raise ValueError(f"Expected {k} ranges for {k}D, got {len(ranges)}")

    result = coeffs.copy()

    # Extract along each dimension
    for axis in range(k):
        t_min, t_max = ranges[axis]

        # Move axis to last position
        result = np.moveaxis(result, axis, -1)
        shape = result.shape
        n_last = shape[-1]
        n_rest = np.prod(shape[:-1])

        # Reshape to 2D
        result_2d = result.reshape(n_rest, n_last)

        # Extract along this dimension
        for i in range(n_rest):
            result_2d[i, :] = extract_subbox_1d(result_2d[i, :], t_min, t_max, tolerance, verbose=False)

        # Reshape back and move axis back
        result = result_2d.reshape(shape)
        result = np.moveaxis(result, -1, axis)

    if verbose:
        print(f"\n=== Extract Sub-box ({k}D) ===")
        print(f"Original box: [0, 1]^{k}")
        print(f"Sub-box ranges: {ranges}")
        print(f"Coefficient shape: {coeffs.shape}")

    return result


def extract_subbox_with_box(coeffs: np.ndarray, parent_box: Box,
                            sub_ranges: List[Tuple[float, float]],
                            tolerance: float = 1e-10, verbose: bool = False) -> Tuple[np.ndarray, Box]:
    """
    Extract Bernstein coefficients for a sub-box and create corresponding Box.

    This is the main utility for getting coefficients on an arbitrary sub-box.

    Parameters
    ----------
    coeffs : np.ndarray
        Bernstein coefficients on parent box (1D, 2D, or kD)
    parent_box : Box
        Parent box (should have ranges [0, 1]^k in normalized space)
    sub_ranges : List[Tuple[float, float]]
        Sub-box ranges in [0, 1]^k (relative to parent box)
        Example: [(0.25, 0.75), (0.0, 0.5)] for 2D
    tolerance : float
        Tolerance for boundary comparison (default: 1e-10)
        If sub_ranges are within tolerance of parent box, return original coefficients
    verbose : bool
        If True, print extraction details

    Returns
    -------
    sub_coeffs : np.ndarray
        Bernstein coefficients for sub-box, renormalized to [0, 1]^k
    sub_box : Box
        Box for sub-domain with correct domain tracking

    Examples
    --------
    >>> # Extract p(t) = t on [0.25, 0.75]
    >>> coeffs = np.array([0, 1])
    >>> box = Box(k=1, ranges=[(0.0, 1.0)])
    >>> sub_coeffs, sub_box = extract_subbox_with_box(coeffs, box, [(0.25, 0.75)])
    >>> sub_coeffs
    array([0.25, 0.75])
    >>> sub_box.ranges
    [(0.25, 0.75)]

    >>> # Near-boundary case: no subdivision needed
    >>> sub_coeffs, sub_box = extract_subbox_with_box(coeffs, box, [(1e-12, 1.0 - 1e-12)])
    >>> # Returns original coefficients
    """
    k = coeffs.ndim

    if len(sub_ranges) != k:
        raise ValueError(f"Expected {k} ranges for {k}D coefficients, got {len(sub_ranges)}")

    # Check if sub_ranges are essentially the full box
    is_full_box = all(
        abs(t_min) < tolerance and abs(t_max - 1.0) < tolerance
        for t_min, t_max in sub_ranges
    )

    if is_full_box:
        if verbose:
            print(f"\n=== Extract Sub-box ({k}D) ===")
            print(f"Sub-ranges {sub_ranges} are within tolerance of [0, 1]^{k}")
            print(f"Returning original coefficients and box (no subdivision)")
        return coeffs.copy(), parent_box

    # Extract coefficients
    if k == 1:
        t_min, t_max = sub_ranges[0]
        sub_coeffs = extract_subbox_1d(coeffs, t_min, t_max, tolerance, verbose)
    elif k == 2:
        sub_coeffs = extract_subbox_2d(coeffs, sub_ranges, tolerance, verbose)
    else:
        sub_coeffs = extract_subbox_kd(coeffs, sub_ranges, tolerance, verbose)

    # Create sub-box
    # Map sub_ranges from [0,1]^k to actual ranges in parent box
    actual_ranges = []
    for i, (t_min, t_max) in enumerate(sub_ranges):
        parent_min, parent_max = parent_box.ranges[i]
        actual_min = parent_min + t_min * (parent_max - parent_min)
        actual_max = parent_min + t_max * (parent_max - parent_min)
        actual_ranges.append((actual_min, actual_max))

    sub_box = Box(
        k=k,
        ranges=actual_ranges,
        normalization_transform=parent_box.normalization_transform,
        parent_box=parent_box,
        depth=parent_box.depth + 1
    )

    if verbose:
        print(f"\n=== Box Extraction ===")
        print(f"Parent box: {parent_box}")
        print(f"Sub-box:    {sub_box}")

    return sub_coeffs, sub_box


def subdivide_with_box(coeffs: np.ndarray, box: Box, axis: int, t: float = 0.5,
                       verbose: bool = False) -> Tuple[np.ndarray, Box, np.ndarray, Box]:
    """
    Subdivide Bernstein polynomial and create corresponding sub-boxes.

    This is the main utility function for LP/PP methods. It:
    1. Subdivides the Bernstein coefficients using de Casteljau
    2. Creates child boxes with correct domain tracking
    3. Returns both coefficients and boxes for further processing

    Parameters
    ----------
    coeffs : np.ndarray
        Bernstein coefficients (1D, 2D, or kD array)
    box : Box
        Current box with domain tracking
    axis : int
        Axis to subdivide
    t : float
        Subdivision point in [0, 1] (default: 0.5)
    verbose : bool
        If True, print subdivision details

    Returns
    -------
    left_coeffs : np.ndarray
        Bernstein coefficients for left sub-polynomial
    left_box : Box
        Box for left sub-domain
    right_coeffs : np.ndarray
        Bernstein coefficients for right sub-polynomial
    right_box : Box
        Box for right sub-domain

    Examples
    --------
    >>> # 1D example: subdivide p(t) = t
    >>> coeffs = np.array([0, 1])
    >>> box = Box(k=1, ranges=[(0.0, 1.0)])
    >>> left_c, left_b, right_c, right_b = subdivide_with_box(coeffs, box, axis=0)
    >>> left_c   # [0, 0.5] on [0, 1]
    array([0., 0.5])
    >>> left_b.ranges
    [(0.0, 0.5)]
    >>> right_c  # [0.5, 1] on [0, 1]
    array([0.5, 1.])
    >>> right_b.ranges
    [(0.5, 1.0)]
    """
    k = coeffs.ndim

    # Subdivide coefficients using de Casteljau
    if k == 1:
        left_coeffs, right_coeffs = de_casteljau_subdivide_1d(coeffs, t, verbose)
    elif k == 2:
        left_coeffs, right_coeffs = de_casteljau_subdivide_2d(coeffs, axis, t, verbose)
    else:
        left_coeffs, right_coeffs = de_casteljau_subdivide_kd(coeffs, axis, t, verbose)

    # Create child boxes
    left_box, right_box = box.subdivide(axis, t)

    if verbose:
        print(f"\n=== Box Subdivision ===")
        print(f"Parent box: {box}")
        print(f"Left box:   {left_box}")
        print(f"Right box:  {right_box}")

    return left_coeffs, left_box, right_coeffs, right_box

