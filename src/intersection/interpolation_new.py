"""
Step 1: Polynomial Interpolation for n-dimensional hypersurfaces

Interpolate parametric hypersurfaces as polynomials using Chebyshev nodes.
"""

import numpy as np
from numpy.polynomial import Polynomial
from typing import List, Tuple
from itertools import product


def interpolate_hypersurface(hypersurface, degree: int = 5, verbose: bool = False) -> List:
    """
    Interpolate a parametric hypersurface as polynomial functions.
    
    For k parameters, we use tensor product polynomials.
    
    Parameters
    ----------
    hypersurface : Hypersurface
        The parametric hypersurface to interpolate
    degree : int
        Degree of the interpolating polynomial in each direction
    verbose : bool
        If True, print interpolation details
        
    Returns
    -------
    polynomials : List
        List of n polynomial representations (one for each coordinate)
        - For k=1: List of Polynomial objects
        - For k>=2: List of coefficient tensors
    """
    k = hypersurface.k  # Number of parameters
    n = hypersurface.n  # Ambient dimension
    
    if k == 1:
        # 1D case: curve
        return _interpolate_curve(hypersurface, degree, verbose)
    elif k == 2:
        # 2D case: surface
        return _interpolate_surface(hypersurface, degree, verbose)
    else:
        # General k-dimensional case
        return _interpolate_general(hypersurface, degree, verbose)


def _interpolate_curve(hypersurface, degree: int, verbose: bool) -> List[Polynomial]:
    """
    Interpolate 1-parameter hypersurface (curve) as polynomials.
    
    Returns:
        List of n Polynomial objects
    """
    u_min, u_max = hypersurface.param_ranges[0]
    n_points = degree + 1
    n = hypersurface.n
    
    # Chebyshev nodes mapped to [u_min, u_max]
    k_vals = np.arange(n_points)
    u_cheb = np.cos((2 * k_vals + 1) * np.pi / (2 * n_points))
    u_values = 0.5 * (u_max - u_min) * (u_cheb + 1) + u_min
    
    # Evaluate hypersurface at sample points
    points = np.array([hypersurface.evaluate(u) for u in u_values])
    
    # Fit polynomials for each coordinate
    polynomials = []
    for i in range(n):
        coord_values = points[:, i]
        poly = Polynomial.fit(u_values, coord_values, degree)
        polynomials.append(poly)
    
    if verbose:
        print(f"Interpolation using {n_points} Chebyshev nodes")
        print(f"Parameter range: [{u_min:.4f}, {u_max:.4f}]")
        print(f"Polynomial degree: {degree}")
        
        for i in range(n):
            print(f"\nPolynomial x{i+1}(u) coefficients (ascending powers):")
            print(f"  {polynomials[i].coef}")
        
        # Compute interpolation error
        u_test = np.linspace(u_min, u_max, 100)
        points_true = np.array([hypersurface.evaluate(u) for u in u_test])
        points_interp = np.column_stack([poly(u_test) for poly in polynomials])
        error = np.linalg.norm(points_true - points_interp, axis=1)
        max_error = np.max(error)
        mean_error = np.mean(error)
        
        print(f"\nInterpolation error:")
        print(f"  Max error:  {max_error:.6e}")
        print(f"  Mean error: {mean_error:.6e}")
    
    return polynomials


def _interpolate_surface(hypersurface, degree: int, verbose: bool) -> List[np.ndarray]:
    """
    Interpolate 2-parameter hypersurface (surface) as tensor product polynomials.
    
    Returns:
        List of n coefficient matrices (each of shape (degree+1, degree+1))
    """
    u_min, u_max = hypersurface.param_ranges[0]
    v_min, v_max = hypersurface.param_ranges[1]
    n_points = degree + 1
    n = hypersurface.n
    
    # Chebyshev nodes for both parameters
    k_vals = np.arange(n_points)
    u_cheb = np.cos((2 * k_vals + 1) * np.pi / (2 * n_points))
    v_cheb = np.cos((2 * k_vals + 1) * np.pi / (2 * n_points))
    
    u_values = 0.5 * (u_max - u_min) * (u_cheb + 1) + u_min
    v_values = 0.5 * (v_max - v_min) * (v_cheb + 1) + v_min
    
    # Create grid
    U, V = np.meshgrid(u_values, v_values, indexing='ij')
    
    # Evaluate hypersurface at grid points
    coord_grids = []
    for i in range(n):
        coord_grid = np.zeros((n_points, n_points))
        for i_u in range(n_points):
            for i_v in range(n_points):
                point = hypersurface.evaluate(U[i_u, i_v], V[i_u, i_v])
                coord_grid[i_u, i_v] = point[i]
        coord_grids.append(coord_grid)
    
    # Fit 2D polynomials (tensor product)
    polynomials = []
    for i in range(n):
        poly_coeffs = _fit_2d_polynomial(u_values, v_values, coord_grids[i], degree)
        polynomials.append(poly_coeffs)
    
    if verbose:
        print(f"Interpolation using {n_points}x{n_points} Chebyshev grid")
        print(f"Parameter ranges: u=[{u_min:.4f}, {u_max:.4f}], v=[{v_min:.4f}, {v_max:.4f}]")
        print(f"Polynomial degree: {degree} in each direction")
        print(f"\nPolynomial coefficient matrices shape: {polynomials[0].shape}")
        print(f"Total coefficients per coordinate: {polynomials[0].size}")
        
        # Compute interpolation error
        u_test = np.linspace(u_min, u_max, 20)
        v_test = np.linspace(v_min, v_max, 20)
        U_test, V_test = np.meshgrid(u_test, v_test, indexing='ij')
        
        errors = []
        for i_u in range(len(u_test)):
            for i_v in range(len(v_test)):
                u, v = U_test[i_u, i_v], V_test[i_u, i_v]
                point_true = hypersurface.evaluate(u, v)
                point_interp = np.array([_evaluate_2d_polynomial(poly, u, v) for poly in polynomials])
                errors.append(np.linalg.norm(point_true - point_interp))
        
        max_error = np.max(errors)
        mean_error = np.mean(errors)
        
        print(f"\nInterpolation error:")
        print(f"  Max error:  {max_error:.6e}")
        print(f"  Mean error: {mean_error:.6e}")
    
    return polynomials


def _interpolate_general(hypersurface, degree: int, verbose: bool) -> List[np.ndarray]:
    """
    Interpolate k-parameter hypersurface (k >= 3) as tensor product polynomials.
    
    Returns:
        List of n coefficient tensors (each of shape (degree+1)^k)
    """
    k = hypersurface.k
    n = hypersurface.n
    n_points = degree + 1
    
    # Chebyshev nodes for each parameter
    param_values = []
    for u_min, u_max in hypersurface.param_ranges:
        k_vals = np.arange(n_points)
        u_cheb = np.cos((2 * k_vals + 1) * np.pi / (2 * n_points))
        u_vals = 0.5 * (u_max - u_min) * (u_cheb + 1) + u_min
        param_values.append(u_vals)
    
    # Create grid of all parameter combinations
    param_grid = list(product(*param_values))
    
    # Evaluate hypersurface at all grid points
    points = np.array([hypersurface.evaluate(*params) for params in param_grid])
    
    # Fit k-dimensional tensor product polynomials
    polynomials = []
    for i in range(n):
        coord_values = points[:, i]
        poly_coeffs = _fit_kd_polynomial(param_values, coord_values, degree, k)
        polynomials.append(poly_coeffs)
    
    if verbose:
        grid_size = " x ".join([str(n_points)] * k)
        print(f"Interpolation using {grid_size} Chebyshev grid")
        print(f"Parameter ranges: {hypersurface.param_ranges}")
        print(f"Polynomial degree: {degree} in each direction")
        print(f"\nPolynomial coefficient tensor shape: {polynomials[0].shape}")
        print(f"Total coefficients per coordinate: {polynomials[0].size}")
    
    return polynomials


def _fit_2d_polynomial(u_values: np.ndarray, v_values: np.ndarray, 
                       Z: np.ndarray, degree: int) -> np.ndarray:
    """
    Fit a 2D polynomial to gridded data.
    
    Returns coefficient matrix C where P(u,v) = sum_{i,j} C[i,j] * u^i * v^j
    """
    n = degree + 1
    
    # Build Vandermonde-like matrix for 2D polynomial
    A = []
    b = []
    
    for i_u, ui in enumerate(u_values):
        for i_v, vj in enumerate(v_values):
            row = []
            for i in range(n):
                for j in range(n):
                    row.append(ui**i * vj**j)
            A.append(row)
            b.append(Z[i_u, i_v])
    
    A = np.array(A)
    b = np.array(b)
    
    # Solve least squares
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # Reshape to matrix form
    C = coeffs.reshape(n, n)
    
    return C


def _evaluate_2d_polynomial(C: np.ndarray, u: float, v: float) -> float:
    """
    Evaluate 2D polynomial with coefficient matrix C at point (u, v).
    """
    n = C.shape[0]
    result = 0.0
    
    for i in range(n):
        for j in range(n):
            result += C[i, j] * (u**i) * (v**j)
    
    return result


def _fit_kd_polynomial(param_values: List[np.ndarray], coord_values: np.ndarray, 
                       degree: int, k: int) -> np.ndarray:
    """
    Fit a k-dimensional tensor product polynomial.
    
    Returns coefficient tensor C of shape (degree+1)^k
    """
    n = degree + 1
    
    # Build Vandermonde-like matrix for k-D polynomial
    param_grid = list(product(*param_values))
    
    A = []
    for params in param_grid:
        row = []
        # Generate all monomial combinations
        for powers in product(range(n), repeat=k):
            monomial = np.prod([params[i]**powers[i] for i in range(k)])
            row.append(monomial)
        A.append(row)
    
    A = np.array(A)
    b = coord_values
    
    # Solve least squares
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # Reshape to tensor form
    C = coeffs.reshape([n] * k)
    
    return C

