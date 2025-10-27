"""
Step 2: Bernstein Basis Conversion for n-dimensional polynomials

Convert polynomials from power basis to Bernstein basis.
"""

import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import comb
from typing import Union
from itertools import product


def polynomial_nd_to_bernstein(poly: Union[Polynomial, np.ndarray], k: int, verbose: bool = False) -> np.ndarray:
    """
    Convert k-dimensional polynomial from power basis to Bernstein basis.
    
    Parameters
    ----------
    poly : Polynomial or np.ndarray
        - For k=1: Polynomial object or 1D array of coefficients
        - For k>=2: k-dimensional array of coefficients
    k : int
        Number of parameters (dimensions)
    verbose : bool
        If True, print conversion details
        
    Returns
    -------
    np.ndarray
        Bernstein coefficients (same shape as input)
    """
    if k == 1:
        return _polynomial_1d_to_bernstein(poly, verbose)
    elif k == 2:
        return _polynomial_2d_to_bernstein(poly, verbose)
    else:
        return _polynomial_kd_to_bernstein(poly, k, verbose)


def _polynomial_1d_to_bernstein(poly: Union[Polynomial, np.ndarray], verbose: bool = False) -> np.ndarray:
    """
    Convert 1D polynomial from power basis to Bernstein basis.

    Power basis: p(t) = sum_{i=0}^n a_i * t^i
    Bernstein basis: p(t) = sum_{i=0}^n b_i * B_i^n(t)
    where B_i^n(t) = C(n,i) * t^i * (1-t)^(n-i)

    Uses the formula: For a polynomial p(x), the Bernstein coefficients are
    b_ν = p(ν/n) for the Bernstein polynomial approximation.

    For exact conversion (when p is already a polynomial of degree ≤ n):
    b_j = Σ_{i=j}^{n} a_i * C(i,j) * C(n-j, n-i) / C(n,j)

    References
    ----------
    https://en.wikipedia.org/wiki/Bernstein_polynomial
    Farouki, R. T. (2012). The Bernstein polynomial basis: A centennial retrospective.
    """
    # Extract coefficients
    if isinstance(poly, Polynomial):
        power_coeffs = poly.coef
    else:
        power_coeffs = np.array(poly)

    n = len(power_coeffs) - 1  # degree

    # Bernstein coefficients using the correct conversion formula
    # The monomial t^k can be expressed as: t^k = Σ_{j=k}^{n} [C(j,k)/C(n,k)] * B_j^n(t)
    # Therefore: b_j = Σ_{k=0}^{j} a_k * C(j,k) / C(n,k)
    bernstein_coeffs = np.zeros(n + 1)

    for j in range(n + 1):
        sum_val = 0.0
        for k in range(j + 1):
            sum_val += power_coeffs[k] * comb(j, k, exact=True) / comb(n, k, exact=True)
        bernstein_coeffs[j] = sum_val

    if verbose:
        print(f"1D Polynomial degree: {n}")
        print(f"Power basis coefficients: {power_coeffs}")
        print(f"Bernstein basis coefficients: {bernstein_coeffs}")

    return bernstein_coeffs


def _polynomial_2d_to_bernstein(poly_coeffs: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Convert 2D tensor product polynomial to Bernstein basis.
    
    Power basis: p(u,v) = sum_{i,j} C[i,j] * u^i * v^j
    Bernstein basis: p(u,v) = sum_{i,j} B[i,j] * B_i^n(u) * B_j^m(v)
    """
    if poly_coeffs.ndim != 2:
        raise ValueError("Expected 2D coefficient array")
    
    n_u, n_v = poly_coeffs.shape
    degree_u = n_u - 1
    degree_v = n_v - 1
    
    # Convert along u direction first
    temp = np.zeros_like(poly_coeffs)
    for j in range(n_v):
        temp[:, j] = _polynomial_1d_to_bernstein(poly_coeffs[:, j], verbose=False)
    
    # Convert along v direction
    bernstein_coeffs = np.zeros_like(poly_coeffs)
    for i in range(n_u):
        bernstein_coeffs[i, :] = _polynomial_1d_to_bernstein(temp[i, :], verbose=False)
    
    if verbose:
        print(f"2D Polynomial degrees: ({degree_u}, {degree_v})")
        print(f"Coefficient matrix shape: {poly_coeffs.shape}")
        print(f"Bernstein coefficient matrix shape: {bernstein_coeffs.shape}")
    
    return bernstein_coeffs


def _polynomial_kd_to_bernstein(poly_coeffs: np.ndarray, k: int, verbose: bool = False) -> np.ndarray:
    """
    Convert k-dimensional tensor product polynomial to Bernstein basis.
    
    Uses separable conversion along each dimension.
    """
    if poly_coeffs.ndim != k:
        raise ValueError(f"Expected {k}D coefficient array, got {poly_coeffs.ndim}D")
    
    degrees = [s - 1 for s in poly_coeffs.shape]
    
    # Convert along each dimension sequentially
    result = poly_coeffs.copy()
    
    for dim in range(k):
        # Convert along dimension 'dim'
        # Move dimension to last position
        result = np.moveaxis(result, dim, -1)
        
        # Get shape
        shape = result.shape
        n_last = shape[-1]
        n_rest = np.prod(shape[:-1])
        
        # Reshape to 2D for easier processing
        result_2d = result.reshape(n_rest, n_last)
        
        # Convert each 1D slice
        for i in range(n_rest):
            result_2d[i, :] = _polynomial_1d_to_bernstein(result_2d[i, :], verbose=False)
        
        # Reshape back
        result = result_2d.reshape(shape)
        
        # Move dimension back
        result = np.moveaxis(result, -1, dim)
    
    if verbose:
        print(f"{k}D Polynomial degrees: {degrees}")
        print(f"Coefficient tensor shape: {poly_coeffs.shape}")
        print(f"Bernstein coefficient tensor shape: {result.shape}")
    
    return result


def evaluate_bernstein_1d(bernstein_coeffs: np.ndarray, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Evaluate 1D Bernstein polynomial at parameter value(s).
    
    Parameters
    ----------
    bernstein_coeffs : np.ndarray
        Bernstein coefficients
    t : float or np.ndarray
        Parameter value(s) in [0, 1]
        
    Returns
    -------
    float or np.ndarray
        Polynomial value(s)
    """
    # Direct evaluation using Bernstein basis functions
    t = np.asarray(t, dtype=float)
    scalar_input = t.ndim == 0
    t = np.atleast_1d(t)
    
    result = np.zeros_like(t, dtype=float)
    n = len(bernstein_coeffs) - 1
    
    for i, b_i in enumerate(bernstein_coeffs):
        # B_i^n(t) = C(n,i) * t^i * (1-t)^(n-i)
        basis = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        result += b_i * basis
    
    if scalar_input:
        return result[0]
    return result


def evaluate_bernstein_2d(bernstein_coeffs: np.ndarray, u: float, v: float) -> float:
    """
    Evaluate 2D Bernstein polynomial at parameter values.
    
    Parameters
    ----------
    bernstein_coeffs : np.ndarray
        2D array of Bernstein coefficients
    u, v : float
        Parameter values in [0, 1]
        
    Returns
    -------
    float
        Polynomial value
    """
    n_u, n_v = bernstein_coeffs.shape
    degree_u = n_u - 1
    degree_v = n_v - 1
    
    result = 0.0
    for i in range(n_u):
        for j in range(n_v):
            # B_i^n(u) * B_j^m(v)
            basis_u = comb(degree_u, i) * (u ** i) * ((1 - u) ** (degree_u - i))
            basis_v = comb(degree_v, j) * (v ** j) * ((1 - v) ** (degree_v - j))
            result += bernstein_coeffs[i, j] * basis_u * basis_v
    
    return result


def evaluate_bernstein_kd(bernstein_coeffs: np.ndarray, *params) -> float:
    """
    Evaluate k-dimensional Bernstein polynomial at parameter values.
    
    Parameters
    ----------
    bernstein_coeffs : np.ndarray
        k-dimensional array of Bernstein coefficients
    *params : float
        k parameter values in [0, 1]
        
    Returns
    -------
    float
        Polynomial value
    """
    k = bernstein_coeffs.ndim
    if len(params) != k:
        raise ValueError(f"Expected {k} parameters, got {len(params)}")
    
    degrees = [s - 1 for s in bernstein_coeffs.shape]
    
    result = 0.0
    
    # Iterate over all multi-indices
    for multi_index in product(*[range(d + 1) for d in degrees]):
        # Compute tensor product of Bernstein basis functions
        basis = 1.0
        for dim, (i, deg, param) in enumerate(zip(multi_index, degrees, params)):
            basis *= comb(deg, i) * (param ** i) * ((1 - param) ** (deg - i))
        
        result += bernstein_coeffs[multi_index] * basis
    
    return result

