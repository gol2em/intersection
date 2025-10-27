"""
Standalone Polynomial System Solver

Solve systems of polynomial equations using PP/LP/Hybrid methods with subdivision.

This module provides a standalone solver that works with arbitrary polynomial systems,
not limited to line-hypersurface intersections.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class PolynomialSystem:
    """
    Represents a system of polynomial equations.
    
    Attributes
    ----------
    equation_coeffs : List[np.ndarray]
        Bernstein coefficients for each equation.
        For k parameters, each array has shape (degree+1,)^k
    param_ranges : List[Tuple[float, float]]
        Parameter ranges [(min1, max1), (min2, max2), ...]
    k : int
        Number of parameters (variables)
    degree : int
        Polynomial degree
    param_names : List[str], optional
        Names for parameters (default: ['t', 'u', 'v', 'w', 's'])
    metadata : Dict[str, Any], optional
        Additional metadata about the system
    """
    equation_coeffs: List[np.ndarray]
    param_ranges: List[Tuple[float, float]]
    k: int
    degree: int
    param_names: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and set defaults."""
        if self.param_names is None:
            default_names = ['t', 'u', 'v', 'w', 's']
            self.param_names = default_names[:self.k]
        
        if self.metadata is None:
            self.metadata = {}
        
        # Validate
        if len(self.equation_coeffs) == 0:
            raise ValueError("Must have at least one equation")
        
        if len(self.param_ranges) != self.k:
            raise ValueError(f"param_ranges length ({len(self.param_ranges)}) must equal k ({self.k})")
        
        if len(self.param_names) != self.k:
            raise ValueError(f"param_names length ({len(self.param_names)}) must equal k ({self.k})")
        
        # Validate equation shapes
        for i, coeffs in enumerate(self.equation_coeffs):
            if coeffs.ndim != self.k:
                raise ValueError(
                    f"Equation {i} has {coeffs.ndim} dimensions, expected {self.k}"
                )


def create_polynomial_system(
    equation_coeffs: List[np.ndarray],
    param_ranges: List[Tuple[float, float]],
    param_names: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> PolynomialSystem:
    """
    Create a polynomial system from Bernstein coefficients.
    
    Parameters
    ----------
    equation_coeffs : List[np.ndarray]
        Bernstein coefficients for each equation.
        For 1D (k=1): each array has shape (degree+1,)
        For 2D (k=2): each array has shape (degree+1, degree+1)
        For kD: each array has shape (degree+1,)^k
    param_ranges : List[Tuple[float, float]]
        Parameter ranges [(min1, max1), (min2, max2), ...]
    param_names : List[str], optional
        Names for parameters (default: ['t', 'u', 'v', 'w', 's'])
    metadata : Dict[str, Any], optional
        Additional metadata
        
    Returns
    -------
    PolynomialSystem
        The polynomial system
        
    Examples
    --------
    >>> # 1D system: f(t) = t^2 - 0.5 = 0
    >>> # Bernstein coefficients for degree 2
    >>> coeffs = [np.array([-0.5, 0.0, 0.5])]
    >>> system = create_polynomial_system(
    ...     equation_coeffs=coeffs,
    ...     param_ranges=[(0.0, 1.0)]
    ... )
    
    >>> # 2D system: f1(u,v) = u - 0.5 = 0, f2(u,v) = v - 0.5 = 0
    >>> coeffs1 = np.array([[-0.5, -0.5], [0.5, 0.5]])
    >>> coeffs2 = np.array([[-0.5, 0.5], [-0.5, 0.5]])
    >>> system = create_polynomial_system(
    ...     equation_coeffs=[coeffs1, coeffs2],
    ...     param_ranges=[(0.0, 1.0), (0.0, 1.0)],
    ...     param_names=['u', 'v']
    ... )
    """
    # Infer k from first equation
    k = equation_coeffs[0].ndim
    
    # Infer degree from first equation
    degree = equation_coeffs[0].shape[0] - 1
    
    return PolynomialSystem(
        equation_coeffs=equation_coeffs,
        param_ranges=param_ranges,
        k=k,
        degree=degree,
        param_names=param_names,
        metadata=metadata
    )


def calculate_optimal_max_depth(equation_coeffs: List[np.ndarray], k: int) -> int:
    """
    Calculate optimal max depth based on theoretical maximum number of roots.

    For a polynomial system with maximum degree m in dimension d:
    - Maximum number of roots: m^d (by Bezout's theorem)
    - Each bisection along one dimension adds 1 to depth
    - To find all m^d roots, we need at most log_2(m^d) = d * log_2(m) subdivisions

    Parameters
    ----------
    equation_coeffs : List[np.ndarray]
        List of Bernstein coefficient arrays
    k : int
        Number of parameters (dimension)

    Returns
    -------
    int
        Optimal maximum depth
    """
    import math

    # Find maximum degree across all equations and all dimensions
    max_degree = 0
    for coeffs in equation_coeffs:
        if k == 1:
            degree = len(coeffs) - 1
        else:
            # For k-dimensional, shape is (n1+1, n2+1, ..., nk+1)
            # Maximum degree is max of all dimensions
            degree = max(coeffs.shape) - 1
        max_degree = max(max_degree, degree)

    if max_degree <= 1:
        # Linear system, very shallow depth needed
        return max(5, k)

    # Theoretical max roots: m^d
    # Max depth needed: log_2(m^d) = d * log_2(m)
    optimal_depth = int(math.ceil(k * math.log2(max_degree)))

    # Add some buffer for numerical issues and non-simple roots
    # But cap at reasonable maximum
    buffered_depth = optimal_depth + 5
    capped_depth = min(buffered_depth, 100)

    return capped_depth


def solve_polynomial_system(
    system: PolynomialSystem,
    method: str = 'pp',
    tolerance: float = 1e-6,
    crit: float = 0.8,
    max_depth: Optional[int] = None,
    subdivision_tolerance: float = 1e-10,
    refine: bool = True,
    verbose: bool = False,
    return_stats: bool = False
) -> tuple:
    """
    Solve a polynomial system.

    Complete workflow:
    1. System is already in Bernstein basis with specified parameter ranges
    2. Use PP/LP/Hybrid method to find all possible roots
    3. (Optional) Use Newton iteration to refine each root
    4. Return solutions in original parameter domain

    Parameters
    ----------
    system : PolynomialSystem
        The polynomial system to solve
    method : str
        Solving method: 'pp', 'lp', or 'hybrid' (default: 'pp')
    tolerance : float
        Size threshold for claiming a root in parameter space (default: 1e-6)
    crit : float
        Critical ratio for subdivision (default: 0.8)
    max_depth : Optional[int]
        Maximum subdivision depth. If None, automatically calculated as ceil(d * log_2(m)) + 5
        where m is the maximum degree and d is the dimension (default: None)
    subdivision_tolerance : float
        Numerical tolerance for zero detection in function value space (default: 1e-10)
        Increase this if coefficients are very small (e.g., use 1e-7 for coeffs ~1e-6)
    refine : bool
        Whether to refine solutions using Newton iteration (default: True)
    verbose : bool
        Print progress (default: False)
    return_stats : bool
        If True, return (solutions, stats). If False, return only solutions for backward compatibility (default: False)

    Returns
    -------
    tuple or List[Dict[str, float]]
        If return_stats=True: (solutions, stats) where:
            - solutions: List[Dict[str, float]] - Solutions in original parameter space
            - stats: dict - Statistics with keys 'boxes_processed', 'boxes_pruned', 'subdivisions', 'solutions_found'
        If return_stats=False: List[Dict[str, float]] - Solutions only (backward compatible)

    Examples
    --------
    >>> # Solve f(t) = t^2 - 0.5 = 0
    >>> coeffs = [np.array([-0.5, 0.0, 0.5])]
    >>> system = create_polynomial_system(coeffs, [(0.0, 1.0)])
    >>> solutions = solve_polynomial_system(system, verbose=True)
    >>> # Returns: [{'t': 0.707...}]

    >>> # Solve 2D system
    >>> coeffs1 = np.array([[-0.5, -0.5], [0.5, 0.5]])
    >>> coeffs2 = np.array([[-0.5, 0.5], [-0.5, 0.5]])
    >>> system = create_polynomial_system(
    ...     [coeffs1, coeffs2],
    ...     [(0.0, 1.0), (0.0, 1.0)],
    ...     param_names=['u', 'v']
    ... )
    >>> solutions = solve_polynomial_system(system)
    >>> # Returns: [{'u': 0.5, 'v': 0.5}]
    """
    from .subdivision_solver import solve_with_subdivision, BoundingMethod
    from .solver import _remove_duplicate_solutions

    # Calculate optimal max_depth if not provided
    if max_depth is None:
        max_depth = calculate_optimal_max_depth(system.equation_coeffs, system.k)
        if verbose:
            print(f"Auto-calculated max_depth: {max_depth}")

    if verbose:
        print("\n" + "=" * 80)
        print("SOLVING POLYNOMIAL SYSTEM")
        print("=" * 80)
        print(f"Number of equations: {len(system.equation_coeffs)}")
        print(f"Number of parameters: {system.k}")
        print(f"Parameter names: {system.param_names}")
        print(f"Parameter ranges: {system.param_ranges}")
        print(f"Method: {method.upper()}")
        print(f"Tolerance: {tolerance}")
        print(f"Max depth: {max_depth}")
        print(f"Refine: {refine}")
    
    # Step 1: Normalize to [0, 1]^k
    if verbose:
        print(f"\nStep 1: Normalizing to [0, 1]^{system.k}")
    
    normalized_coeffs = _normalize_coefficients(
        system.equation_coeffs,
        system.param_ranges,
        system.k,
        verbose
    )
    
    # Create normalization transform
    normalization_transform = {
        'original_ranges': system.param_ranges,
        'normalized_ranges': [(0.0, 1.0) for _ in range(system.k)]
    }
    
    # Step 2: Solve using subdivision method
    if verbose:
        print(f"\nStep 2: Finding roots using {method.upper()} subdivision method")

    method_enum = BoundingMethod[method.upper()]

    solutions_normalized, solver_stats = solve_with_subdivision(
        normalized_coeffs,
        k=system.k,
        method=method,
        tolerance=tolerance,
        crit=crit,
        max_depth=max_depth,
        subdivision_tolerance=subdivision_tolerance,
        normalization_transform=normalization_transform,
        verbose=verbose
    )

    if verbose:
        print(f"\n  Found {len(solutions_normalized)} candidate solutions")
    
    # Step 3: (Optional) Refine using Newton iteration
    solutions_refined = solutions_normalized

    if refine and len(solutions_normalized) > 0:
        if verbose:
            print(f"\nStep 3: Refining solutions using Newton iteration")

        solutions_refined = []
        for i, sol_norm in enumerate(solutions_normalized):
            sol_refined_norm = _refine_solution_newton_standalone(
                normalized_coeffs, sol_norm, system.k, max_iter=10, tol=1e-10, verbose=False
            )

            if sol_refined_norm is not None:
                solutions_refined.append(sol_refined_norm)
                if verbose:
                    print(f"  Solution {i+1}: {sol_norm} -> {sol_refined_norm}")
            else:
                solutions_refined.append(sol_norm)
                if verbose:
                    print(f"  Solution {i+1}: {sol_norm} (Newton failed, keeping unrefined)")
    
    # Step 4: Convert back to original domain
    if verbose:
        print(f"\nStep 4: Converting to original parameter domain")
    
    solutions_original = []
    for sol_norm in solutions_refined:
        sol_orig = _denormalize_solution(
            sol_norm,
            system.param_ranges,
            system.param_names
        )
        solutions_original.append(sol_orig)
        
        if verbose:
            print(f"  Normalized: {sol_norm} -> Original: {sol_orig}")
    
    # Remove duplicates
    solutions_original = _remove_duplicate_solutions(solutions_original, tolerance=tolerance)

    if verbose:
        print(f"\n" + "=" * 80)
        print(f"TOTAL SOLUTIONS FOUND: {len(solutions_original)}")
        print("=" * 80)

    # Return solutions with or without stats based on return_stats flag
    if return_stats:
        return solutions_original, solver_stats
    else:
        return solutions_original


def _normalize_coefficients(
    equation_coeffs: List[np.ndarray],
    param_ranges: List[Tuple[float, float]],
    k: int,
    verbose: bool = False
) -> List[np.ndarray]:
    """
    Normalize Bernstein coefficients from arbitrary parameter ranges to [0, 1]^k.

    IMPORTANT: The input Bernstein coefficients must already represent the polynomial
    in the [0,1]^k domain. If your original polynomial is defined on a custom domain
    [a,b]^k, you must transform it to [0,1]^k BEFORE converting to Bernstein basis.

    Example for custom domain:
    -------------------------
    Original: f(x) = x - 5 on domain x ∈ [2, 8]

    Step 1: Transform to [0,1] domain
        Let s ∈ [0,1] where x = 2 + 6*s
        Then f(s) = (2 + 6*s) - 5 = 6*s - 3

    Step 2: Convert f(s) to Bernstein basis
        power_coeffs = [-3, 6]
        bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)

    Step 3: Create system with original param_ranges
        system = create_polynomial_system(
            equation_coeffs=[bern_coeffs],
            param_ranges=[(2.0, 8.0)],  # Original domain
            param_names=['x']
        )

    The param_ranges are used only for denormalization of solutions back to
    the original domain. The actual solving happens in [0,1]^k space.

    Parameters
    ----------
    equation_coeffs : List[np.ndarray]
        Bernstein coefficients for each equation (already in [0,1]^k domain)
    param_ranges : List[Tuple[float, float]]
        Original parameter ranges (used for denormalization only)
    k : int
        Number of parameters
    verbose : bool
        Print progress

    Returns
    -------
    List[np.ndarray]
        Normalized coefficients (same as input for Bernstein basis)

    Notes
    -----
    For Bernstein polynomials, the coefficients are already in the correct form
    for [0,1]^k domain. The parameter transformation is handled by the Box class
    during denormalization.
    """
    if verbose:
        print(f"\nNormalizing coefficients:")
        print(f"  Parameter dimension: {k}")
        print(f"  Parameter ranges: {param_ranges}")
        print(f"  Note: Bernstein coefficients should already be for [0,1]^k domain")

    # For Bernstein basis, coefficients are already normalized to [0,1]^k
    # The param_ranges are used only for denormalization of solutions
    return equation_coeffs


def _denormalize_solution(
    solution: np.ndarray,
    param_ranges: List[Tuple[float, float]],
    param_names: List[str]
) -> Dict[str, float]:
    """
    Convert solution from normalized [0,1]^k space to original parameter space.

    Parameters
    ----------
    solution : np.ndarray
        Solution in normalized space
    param_ranges : List[Tuple[float, float]]
        Original parameter ranges
    param_names : List[str]
        Parameter names

    Returns
    -------
    Dict[str, float]
        Solution in original space
    """
    result = {}
    for i, name in enumerate(param_names):
        min_val, max_val = param_ranges[i]
        # Linear interpolation: x_orig = min + (max - min) * x_norm
        result[name] = min_val + (max_val - min_val) * solution[i]

    return result


def _evaluate_polynomial_system(
    equation_coeffs: List[np.ndarray],
    params: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Evaluate polynomial system at given parameter values.

    Parameters
    ----------
    equation_coeffs : List[np.ndarray]
        Bernstein coefficients for each equation
    params : np.ndarray
        Parameter values (shape: (k,))
    k : int
        Number of parameters

    Returns
    -------
    np.ndarray
        Residuals for each equation
    """
    from .bernstein import evaluate_bernstein_kd

    residuals = np.zeros(len(equation_coeffs))

    for i, coeffs in enumerate(equation_coeffs):
        residuals[i] = evaluate_bernstein_kd(coeffs, *params)

    return residuals


def _refine_solution_newton_standalone(
    equation_coeffs: List[np.ndarray],
    solution: np.ndarray,
    k: int,
    max_iter: int = 10,
    tol: float = 1e-10,
    verbose: bool = False
) -> Optional[np.ndarray]:
    """
    Refine a solution using Newton iteration with numerical Jacobian.

    Parameters
    ----------
    equation_coeffs : List[np.ndarray]
        Bernstein coefficients for each equation
    solution : np.ndarray
        Initial solution guess
    k : int
        Number of parameters
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print progress

    Returns
    -------
    Optional[np.ndarray]
        Refined solution, or None if Newton iteration failed
    """
    x = solution.copy()

    for iteration in range(max_iter):
        # Evaluate residuals
        residuals = _evaluate_polynomial_system(equation_coeffs, x, k)
        residual_norm = np.linalg.norm(residuals)

        if verbose:
            print(f"  Newton iteration {iteration}: residual = {residual_norm:.2e}")

        if residual_norm < tol:
            if verbose:
                print(f"  Converged in {iteration} iterations")
            return x

        # Compute Jacobian numerically
        J = np.zeros((len(residuals), k))
        h = 1e-8

        for j in range(k):
            x_plus = x.copy()
            x_plus[j] += h
            x_plus = np.clip(x_plus, 0.0, 1.0)
            residuals_plus = _evaluate_polynomial_system(equation_coeffs, x_plus, k)
            J[:, j] = (residuals_plus - residuals) / h

        # Solve J * delta = -residuals
        try:
            delta = np.linalg.solve(J, -residuals)
        except np.linalg.LinAlgError:
            if verbose:
                print(f"  Singular Jacobian at iteration {iteration}")
            return None

        # Update solution
        x = x + delta
        x = np.clip(x, 0.0, 1.0)

    # Check final residual
    residuals = _evaluate_polynomial_system(equation_coeffs, x, k)
    residual_norm = np.linalg.norm(residuals)

    if verbose:
        print(f"  Max iterations reached. Final residual: {residual_norm:.2e}")

    if residual_norm < tol * 10:
        return x
    else:
        return None

