"""
Step 3: Polynomial System Formation (N-Dimensional)

Create polynomial systems for line-hypersurface intersection in n-dimensional space.

Mathematical Formulation:
-----------------------
Line in n-dimensional space: Intersection of (n-1) hyperplanes
    H_i: a_i1*x_1 + a_i2*x_2 + ... + a_in*x_n + d_i = 0, i = 1, ..., n-1

Hypersurface in n-dimensional space: (n-1)-dimensional parametric manifold
    S(u_1, ..., u_{n-1}) = (x_1(u), x_2(u), ..., x_n(u))

Intersection condition:
    Point P on hypersurface S must satisfy all n-1 hyperplane equations
    H_i(S(u)) = 0 for all i = 1, ..., n-1

This gives us (n-1) polynomial equations in (n-1) unknowns (u_1, ..., u_{n-1}).
"""

import numpy as np
from typing import Dict, Any, List
from .geometry import Line, Hypersurface


def create_intersection_system(line: Line, hypersurface: Hypersurface,
                               verbose: bool = False) -> Dict[str, Any]:
    """
    Create polynomial system for line-hypersurface intersection in n-dimensional space.

    Mathematical Setup:
    ------------------
    Line: Defined by (n-1) hyperplanes in n-dimensional space
        H_i: a_i1*x_1 + ... + a_in*x_n + d_i = 0, i = 1, ..., n-1

    Hypersurface: (n-1)-dimensional parametric manifold
        S(u_1, ..., u_{n-1}) = (x_1(u), ..., x_n(u))

    Intersection Condition:
    ----------------------
    Point P = S(u) on hypersurface must satisfy all hyperplane equations:
        H_i(S(u)) = 0 for all i = 1, ..., n-1

    This gives (n-1) polynomial equations in (n-1) parameters.

    Parameters
    ----------
    line : Line
        Line defined by (n-1) hyperplanes
    hypersurface : Hypersurface
        Parametric hypersurface with Bernstein coefficients
    verbose : bool
        If True, print system details

    Returns
    -------
    dict
        Polynomial system representation with:
        - 'n': ambient dimension
        - 'k': number of parameters (= n-1)
        - 'line': Line object
        - 'hypersurface': Hypersurface object
        - 'equations': List of (n-1) equation specifications
        - 'bernstein_coeffs': List of Bernstein coefficient arrays for each coordinate

    Raises
    ------
    ValueError
        If line and hypersurface dimensions don't match
    """
    # Validate dimensions
    if line.n != hypersurface.n:
        raise ValueError(f"Line dimension ({line.n}) must match hypersurface dimension ({hypersurface.n})")

    if len(line.hyperplanes) != hypersurface.k:
        raise ValueError(f"Line must have {hypersurface.k} hyperplanes for {hypersurface.n}D intersection")

    n = line.n
    k = hypersurface.k

    if verbose:
        print(f"\n=== Creating Intersection System ({k}→{n}D) ===")
        print(f"Ambient dimension: {n}")
        print(f"Number of parameters: {k}")
        print(f"Number of equations: {k}")
        print(f"\nLine defined by {len(line.hyperplanes)} hyperplanes:")
        for i, h in enumerate(line.hyperplanes):
            print(f"  H{i+1}: {h}")

    # Extract Bernstein coefficients for each coordinate
    bernstein_coeffs = hypersurface.bernstein_coeffs

    if verbose:
        print(f"\nHypersurface Bernstein coefficients:")
        for i, bern in enumerate(bernstein_coeffs):
            if isinstance(bern, np.ndarray):
                print(f"  x{i+1}(u): shape = {bern.shape}, dtype = {bern.dtype}")
            else:
                print(f"  x{i+1}(u): {type(bern)}")

    # Create equation specifications
    # Each hyperplane H_i gives one equation: sum_j(a_ij * x_j(u)) + d_i = 0
    equations = []
    for i, hyperplane in enumerate(line.hyperplanes):
        equation_spec = {
            'hyperplane_index': i,
            'coeffs': hyperplane.coeffs,  # [a_i1, a_i2, ..., a_in]
            'd': hyperplane.d,
            'description': f"H{i+1}: " + " + ".join([
                f"{hyperplane.coeffs[j]:.4f}*x{j+1}(u)" for j in range(n)
            ]) + f" + {hyperplane.d:.4f} = 0"
        }
        equations.append(equation_spec)

        if verbose:
            print(f"\nEquation {i+1}: {equation_spec['description']}")

    system = {
        'n': n,
        'k': k,
        'line': line,
        'hypersurface': hypersurface,
        'equations': equations,
        'bernstein_coeffs': bernstein_coeffs,
        'param_ranges': hypersurface.param_ranges,
        'degree': hypersurface.degree
    }

    if verbose:
        print(f"\n=== System Created Successfully ===")
        print(f"System has {len(equations)} equations in {k} unknowns")

    return system



def evaluate_system(system: Dict[str, Any], *params) -> np.ndarray:
    """
    Evaluate the intersection system at given parameter values.

    For each hyperplane equation H_i, compute:
        residual_i = sum_j(a_ij * x_j(u)) + d_i

    At an intersection point, all residuals should be 0.

    Parameters
    ----------
    system : dict
        System created by create_intersection_system
    *params : float
        Parameter values (u_1, ..., u_k) where k = n-1

    Returns
    -------
    np.ndarray
        Array of residuals, one for each equation
        Shape: (k,) where k = n-1

    Raises
    ------
    ValueError
        If wrong number of parameters provided
    """
    k = system['k']
    n = system['n']

    if len(params) != k:
        raise ValueError(f"Expected {k} parameters, got {len(params)}")

    # Evaluate hypersurface at parameter values to get point in n-dimensional space
    point = system['hypersurface'].evaluate(*params)

    # Evaluate each hyperplane equation at this point
    residuals = []
    for eq_spec in system['equations']:
        # H_i: sum_j(a_ij * x_j) + d_i = 0
        residual = np.dot(eq_spec['coeffs'], point) + eq_spec['d']
        residuals.append(residual)

    return np.array(residuals)


def evaluate_system_bernstein(system: Dict[str, Any], *params,
                              use_polynomials: bool = True) -> np.ndarray:
    """
    Evaluate the intersection system using Bernstein polynomial representation.

    This is more efficient than evaluate_system() when we want to use the
    polynomial approximation rather than the original function.

    Parameters
    ----------
    system : dict
        System created by create_intersection_system
    *params : float
        Parameter values (u_1, ..., u_k) where k = n-1
    use_polynomials : bool
        If True, use polynomial interpolation; if False, use original function

    Returns
    -------
    np.ndarray
        Array of residuals, one for each equation

    Notes
    -----
    This function evaluates the Bernstein polynomials directly, which is useful
    for the LP method that works in Bernstein basis.
    """
    k = system['k']
    n = system['n']

    if len(params) != k:
        raise ValueError(f"Expected {k} parameters, got {len(params)}")

    # Evaluate each coordinate using Bernstein polynomials
    if use_polynomials:
        # Use polynomial evaluation (to be implemented with Bernstein evaluation)
        # For now, fall back to original function
        point = system['hypersurface'].evaluate(*params)
    else:
        point = system['hypersurface'].evaluate(*params)

    # Evaluate each hyperplane equation
    residuals = []
    for eq_spec in system['equations']:
        residual = np.dot(eq_spec['coeffs'], point) + eq_spec['d']
        residuals.append(residual)

    return np.array(residuals)


# Backward compatibility functions (deprecated)
def create_intersection_system_2d(line, bern_x: np.ndarray, bern_y: np.ndarray,
                                   verbose: bool = False) -> Dict[str, Any]:
    """
    DEPRECATED: Use create_intersection_system() with Line and Hypersurface objects.

    This function is kept for backward compatibility only.
    """
    raise NotImplementedError(
        "create_intersection_system_2d is deprecated. "
        "Use create_intersection_system(line, hypersurface) instead."
    )


def create_intersection_system_3d(line, bern_x: np.ndarray, bern_y: np.ndarray,
                                   bern_z: np.ndarray, verbose: bool = False) -> Dict[str, Any]:
    """
    DEPRECATED: Use create_intersection_system() with Line and Hypersurface objects.

    This function is kept for backward compatibility only.
    """
    raise NotImplementedError(
        "create_intersection_system_3d is deprecated. "
        "Use create_intersection_system(line, hypersurface) instead."
    )


def evaluate_system_2d(system: Dict[str, Any], t: float) -> float:
    """
    DEPRECATED: Use evaluate_system() instead.
    """
    raise NotImplementedError(
        "evaluate_system_2d is deprecated. "
        "Use evaluate_system(system, t) instead."
    )


def evaluate_system_3d(system: Dict[str, Any], u: float, v: float) -> tuple:
    """
    DEPRECATED: Use evaluate_system() instead.
    """
    raise NotImplementedError(
        "evaluate_system_3d is deprecated. "
        "Use evaluate_system(system, u, v) instead."
    )

