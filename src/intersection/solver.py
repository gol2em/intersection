"""
Step 4: Polynomial System Solver

Solve polynomial systems to find intersection parameters.
Supports multiple methods: LP, PP, numerical (default).
"""

import numpy as np
from scipy.optimize import fsolve, brentq
from typing import List, Dict, Any, Optional


def solve_polynomial_system(
    system: Dict[str, Any],
    method: str = 'auto',
    tolerance: float = 1e-6,
    max_depth: int = 20,
    verbose: bool = False
) -> List[Dict[str, float]]:
    """
    Solve the polynomial system to find intersection parameters.

    Parameters
    ----------
    system : dict
        Polynomial system from create_intersection_system()
    method : str
        Solving method:
        - 'auto': Choose based on system type (default)
        - 'lp': Linear Programming method (Sherbrooke & Patrikalakis 1993)
        - 'pp': Projected Polyhedron method (Sherbrooke & Patrikalakis 1993)
        - 'numerical': Numerical methods (numpy roots, fsolve)
        - 'subdivision': Simple Bernstein subdivision
    tolerance : float
        Convergence tolerance for subdivision methods
    max_depth : int
        Maximum subdivision depth for subdivision methods
    verbose : bool
        If True, print solving details

    Returns
    -------
    list of dict
        List of solutions with parameters

    Examples
    --------
    >>> system = create_intersection_system(line, hypersurface)
    >>> solutions = solve_polynomial_system(system, method='lp', verbose=True)
    >>> solutions = solve_polynomial_system(system, method='pp')
    >>> solutions = solve_polynomial_system(system)  # auto-select method
    """
    # Auto-select method based on system
    if method == 'auto':
        method = _auto_select_method(system)
        if verbose:
            print(f"Auto-selected method: {method}")

    # Dispatch to appropriate solver
    if method == 'lp':
        return _solve_lp(system, tolerance, max_depth, verbose)
    elif method == 'pp':
        return _solve_pp(system, tolerance, max_depth, verbose)
    elif method == 'numerical':
        return _solve_numerical(system, verbose)
    elif method == 'subdivision':
        return _solve_subdivision(system, tolerance, max_depth, verbose)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: 'auto', 'lp', 'pp', 'numerical', 'subdivision'")


def _auto_select_method(system: Dict[str, Any]) -> str:
    """
    Automatically select solving method based on system properties.

    Strategy:
    - Use PP method if Bernstein coefficients are available
    - Otherwise use numerical methods
    """
    # Check if we have equation Bernstein coefficients
    if 'equation_bernstein_coeffs' in system:
        # Use PP method for systems with Bernstein coefficients
        return 'pp'
    else:
        return 'numerical'


def _solve_lp(system: Dict[str, Any], tolerance: float, max_depth: int, verbose: bool) -> List[Dict[str, float]]:
    """
    Solve using Linear Programming method.

    TODO: Implement LP method from Sherbrooke & Patrikalakis 1993
    """
    if verbose:
        print("LP method not yet implemented, falling back to numerical method")
    return _solve_numerical(system, verbose)


def _solve_pp(system: Dict[str, Any], tolerance: float, max_depth: int, verbose: bool) -> List[Dict[str, float]]:
    """
    Solve using Projected Polyhedron method.

    Complete workflow:
    1. Convert to Bernstein basis and normalize (already done in system)
    2. Use PP method to find all possible roots
    3. (Optional) Use Newton iteration to refine each root
    4. Return to original domain

    Parameters
    ----------
    system : dict
        Polynomial system with equation_bernstein_coeffs
    tolerance : float
        Convergence tolerance
    max_depth : int
        Maximum subdivision depth
    verbose : bool
        Print progress

    Returns
    -------
    list of dict
        Solutions in original parameter space
    """
    from .subdivision_solver import solve_with_subdivision
    from .box import Box

    if verbose:
        print("\n" + "=" * 80)
        print("SOLVING WITH PP METHOD")
        print("=" * 80)

    # Step 1: Get Bernstein coefficients (already normalized to [0,1]^k)
    equation_coeffs = system['equation_bernstein_coeffs']
    k = system['k']
    param_ranges = system['param_ranges']

    if verbose:
        print(f"\nStep 1: System already in Bernstein basis")
        print(f"  Number of equations: {len(equation_coeffs)}")
        print(f"  Number of parameters: {k}")
        print(f"  Original domain: {param_ranges}")
        print(f"  Normalized domain: [0, 1]^{k}")

    # Create normalization transform for converting back to original domain
    normalization_transform = {
        'original_ranges': param_ranges,
        'normalized_ranges': [(0.0, 1.0) for _ in range(k)]
    }

    # Step 2: Use PP method to find all possible roots
    if verbose:
        print(f"\nStep 2: Finding roots using PP subdivision method")
        print(f"  Tolerance: {tolerance}")
        print(f"  Max depth: {max_depth}")

    solutions_normalized = solve_with_subdivision(
        equation_coeffs,
        k=k,
        method='pp',
        tolerance=tolerance,
        max_depth=max_depth,
        normalization_transform=normalization_transform,
        verbose=verbose
    )

    if verbose:
        print(f"\n  Found {len(solutions_normalized)} candidate solutions")

    # Step 3: (Optional) Refine using Newton iteration
    if verbose:
        print(f"\nStep 3: Refining solutions using Newton iteration")

    solutions_refined = []
    for i, sol_norm in enumerate(solutions_normalized):
        # Refine in normalized space
        sol_refined_norm = _refine_solution_newton(
            system, sol_norm, max_iter=10, tol=1e-10, verbose=verbose
        )

        if sol_refined_norm is not None:
            solutions_refined.append(sol_refined_norm)
            if verbose:
                print(f"  Solution {i+1}: {sol_norm} -> {sol_refined_norm}")
        else:
            # Keep unrefined if Newton fails
            solutions_refined.append(sol_norm)
            if verbose:
                print(f"  Solution {i+1}: {sol_norm} (Newton failed, keeping unrefined)")

    # Step 4: Convert back to original domain
    if verbose:
        print(f"\nStep 4: Converting to original parameter domain")

    solutions_original = []
    for sol_norm in solutions_refined:
        # Convert from [0,1]^k to original parameter ranges
        sol_orig = _denormalize_solution(sol_norm, param_ranges, k)
        solutions_original.append(sol_orig)

        if verbose:
            print(f"  Normalized: {sol_norm} -> Original: {sol_orig}")

    # Remove duplicates
    solutions_original = _remove_duplicate_solutions(solutions_original, tolerance=tolerance)

    if verbose:
        print(f"\n" + "=" * 80)
        print(f"TOTAL SOLUTIONS FOUND: {len(solutions_original)}")
        print("=" * 80)

    return solutions_original


def _solve_subdivision(system: Dict[str, Any], tolerance: float, max_depth: int, verbose: bool) -> List[Dict[str, float]]:
    """
    Solve using simple Bernstein subdivision.

    TODO: Implement subdivision method
    """
    if verbose:
        print("Subdivision method not yet implemented, falling back to numerical method")
    return _solve_numerical(system, verbose)


def _refine_solution_newton(
    system: Dict[str, Any],
    solution: np.ndarray,
    max_iter: int = 10,
    tol: float = 1e-10,
    verbose: bool = False
) -> Optional[np.ndarray]:
    """
    Refine a solution using Newton iteration.

    For a system of equations F(x) = 0, Newton iteration is:
        x_{n+1} = x_n - J(x_n)^{-1} * F(x_n)

    where J is the Jacobian matrix.

    Parameters
    ----------
    system : dict
        Polynomial system
    solution : np.ndarray
        Initial guess (in normalized [0,1]^k space)
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print progress

    Returns
    -------
    np.ndarray or None
        Refined solution, or None if Newton fails
    """
    from .polynomial_system import evaluate_system

    x = solution.copy()
    k = len(x)

    for iteration in range(max_iter):
        # Evaluate system at current point
        residuals = evaluate_system(system, *x)
        residual_norm = np.linalg.norm(residuals)

        if residual_norm < tol:
            # Converged
            return x

        # Compute Jacobian numerically
        J = np.zeros((len(residuals), k))
        h = 1e-8

        for j in range(k):
            x_plus = x.copy()
            x_plus[j] += h

            # Clamp to [0, 1]
            x_plus = np.clip(x_plus, 0.0, 1.0)

            residuals_plus = evaluate_system(system, *x_plus)
            J[:, j] = (residuals_plus - residuals) / h

        # Solve J * delta = -residuals
        try:
            delta = np.linalg.solve(J, -residuals)
        except np.linalg.LinAlgError:
            # Singular Jacobian
            return None

        # Update
        x = x + delta

        # Clamp to [0, 1]
        x = np.clip(x, 0.0, 1.0)

    # Check final residual
    residuals = evaluate_system(system, *x)
    residual_norm = np.linalg.norm(residuals)

    if residual_norm < tol * 10:  # Relaxed tolerance
        return x
    else:
        return None


def _denormalize_solution(
    solution: np.ndarray,
    param_ranges: List[tuple],
    k: int
) -> Dict[str, float]:
    """
    Convert solution from normalized [0,1]^k space to original parameter space.

    Parameters
    ----------
    solution : np.ndarray
        Solution in normalized space
    param_ranges : list of tuples
        Original parameter ranges [(min1, max1), (min2, max2), ...]
    k : int
        Number of parameters

    Returns
    -------
    dict
        Solution in original space with keys 't', 'u', 'v', etc.
    """
    param_names = ['t', 'u', 'v', 'w', 's'][:k]

    result = {}
    for i, name in enumerate(param_names):
        min_val, max_val = param_ranges[i]
        # Linear interpolation: x_orig = min + (max - min) * x_norm
        result[name] = min_val + (max_val - min_val) * solution[i]

    return result


def _remove_duplicate_solutions(
    solutions: List[Dict[str, float]],
    tolerance: float = 1e-6
) -> List[Dict[str, float]]:
    """
    Remove duplicate solutions.

    Parameters
    ----------
    solutions : list of dict
        Solutions to filter
    tolerance : float
        Distance threshold for considering solutions as duplicates

    Returns
    -------
    list of dict
        Unique solutions
    """
    if not solutions:
        return []

    unique = [solutions[0]]

    for sol in solutions[1:]:
        is_duplicate = False

        for unique_sol in unique:
            # Check if all parameters are close
            all_close = True
            for key in sol.keys():
                if abs(sol[key] - unique_sol[key]) > tolerance:
                    all_close = False
                    break

            if all_close:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(sol)

    return unique


def _solve_numerical(system: Dict[str, Any], verbose: bool) -> List[Dict[str, float]]:
    """
    Solve using numerical methods (existing implementation).

    Dispatches to appropriate solver based on system type.
    """
    if system['type'] == '2D':
        return solve_2d_system(system, verbose)
    elif system['type'] == '3D':
        return solve_3d_system(system, verbose)
    else:
        # For n-dimensional systems, try to use the general approach
        k = system.get('k', 1)
        if k == 1:
            return solve_2d_system(system, verbose)
        elif k == 2:
            return solve_3d_system(system, verbose)
        else:
            raise ValueError(f"Numerical solver not implemented for k={k} parameters")



# ============================================================================
# Numerical Solvers (Existing Implementation)
# ============================================================================

def solve_2d_system(system: Dict[str, Any], verbose: bool = False) -> List[Dict[str, float]]:
    """
    Solve 2D polynomial equation to find curve parameter t.
    
    Uses multiple methods:
    1. Find roots of polynomial equation
    2. Refine using Newton's method
    3. Filter to valid parameter range
    """
    equation = system['equation']
    
    if verbose:
        print(f"Solving polynomial equation of degree {equation.degree()}")
        print(f"Coefficients: {equation.coef}")
    
    # Find roots using numpy
    roots = equation.roots()
    
    if verbose:
        print(f"\nFound {len(roots)} roots (including complex):")
        for i, root in enumerate(roots):
            print(f"  Root {i+1}: {root}")
    
    # Filter to real roots in valid range [0, 1]
    # (assuming curve is parameterized on [0, 1] after normalization)
    real_roots = []
    tolerance = 1e-6
    
    for root in roots:
        if np.abs(root.imag) < tolerance:  # Real root
            t = root.real
            # Check if in valid range (with small tolerance)
            if -tolerance <= t <= 1 + tolerance:
                # Clamp to [0, 1]
                t = np.clip(t, 0, 1)
                
                # Verify it's actually a root
                residual = abs(equation(t))
                
                if residual < 1e-4:  # Tolerance for root
                    real_roots.append(t)
                    
                    if verbose:
                        print(f"  Valid root: t = {t:.8f}, residual = {residual:.2e}")
    
    # Remove duplicates
    if len(real_roots) > 1:
        real_roots = remove_duplicate_roots(real_roots, tolerance=1e-6)
    
    # Refine roots using Newton's method
    refined_roots = []
    for t in real_roots:
        t_refined = refine_root_newton(equation, t, max_iter=10)
        refined_roots.append(t_refined)
        
        if verbose:
            print(f"  Refined: t = {t:.8f} -> {t_refined:.8f}")
    
    solutions = [{'t': t} for t in refined_roots]
    
    if verbose:
        print(f"\nTotal solutions found: {len(solutions)}")
    
    return solutions


def solve_3d_system(system: Dict[str, Any], verbose: bool = False) -> List[Dict[str, float]]:
    """
    Solve 3D polynomial system to find surface parameters (u, v).
    
    This is more complex as we have 2 equations in 2 unknowns.
    We use numerical methods with multiple initial guesses.
    """
    from .polynomial_system import evaluate_system_3d
    
    if verbose:
        print("Solving 3D system: 2 equations in 2 unknowns (u, v)")
    
    # Define residual function
    def residuals(params):
        u, v = params
        r1, r2 = evaluate_system_3d(system, u, v)
        return [r1, r2]
    
    # Try multiple initial guesses on a grid
    n_grid = 5
    u_guesses = np.linspace(0, 1, n_grid)
    v_guesses = np.linspace(0, 1, n_grid)
    
    solutions = []
    tolerance = 1e-6
    
    if verbose:
        print(f"Trying {n_grid}x{n_grid} = {n_grid**2} initial guesses")
    
    for u0 in u_guesses:
        for v0 in v_guesses:
            try:
                # Solve using fsolve
                sol, info, ier, msg = fsolve(residuals, [u0, v0], full_output=True)
                u, v = sol
                
                # Check if solution is valid
                if ier == 1:  # Solution found
                    # Check if in valid range
                    if -tolerance <= u <= 1 + tolerance and -tolerance <= v <= 1 + tolerance:
                        u = np.clip(u, 0, 1)
                        v = np.clip(v, 0, 1)
                        
                        # Check residual
                        r1, r2 = evaluate_system_3d(system, u, v)
                        residual = np.sqrt(r1**2 + r2**2)
                        
                        if residual < 1e-4:
                            # Check if this is a duplicate
                            is_duplicate = False
                            for existing in solutions:
                                du = abs(u - existing['u'])
                                dv = abs(v - existing['v'])
                                if du < tolerance and dv < tolerance:
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate:
                                solutions.append({'u': u, 'v': v})
                                
                                if verbose:
                                    print(f"  Found solution: u={u:.6f}, v={v:.6f}, residual={residual:.2e}")
            except:
                pass  # fsolve failed, try next initial guess
    
    if verbose:
        print(f"\nTotal solutions found: {len(solutions)}")
    
    return solutions


def refine_root_newton(poly, x0: float, max_iter: int = 10, tol: float = 1e-10) -> float:
    """
    Refine a polynomial root using Newton's method.
    """
    x = x0
    poly_deriv = poly.deriv()
    
    for _ in range(max_iter):
        fx = poly(x)
        if abs(fx) < tol:
            break
        
        fpx = poly_deriv(x)
        if abs(fpx) < 1e-14:
            break  # Derivative too small
        
        x = x - fx / fpx
    
    return x


def remove_duplicate_roots(roots: List[float], tolerance: float = 1e-6) -> List[float]:
    """
    Remove duplicate roots from a list.
    """
    if not roots:
        return []
    
    roots = sorted(roots)
    unique = [roots[0]]
    
    for root in roots[1:]:
        if abs(root - unique[-1]) > tolerance:
            unique.append(root)
    
    return unique


def subdivide_and_solve(bernstein_coeffs: np.ndarray, depth: int = 0, max_depth: int = 10,
                       tolerance: float = 1e-6) -> List[float]:
    """
    Solve polynomial using subdivision (for Bernstein form).
    
    This is an alternative solver that uses the convex hull property
    of Bernstein polynomials for robust root finding.
    
    Parameters
    ----------
    bernstein_coeffs : np.ndarray
        Bernstein coefficients
    depth : int
        Current recursion depth
    max_depth : int
        Maximum recursion depth
    tolerance : float
        Tolerance for root finding
        
    Returns
    -------
    list of float
        Roots in [0, 1]
    """
    from .bernstein import subdivide_bernstein, evaluate_bernstein
    
    # Check if all coefficients have the same sign (no root)
    if np.all(bernstein_coeffs > 0) or np.all(bernstein_coeffs < 0):
        return []
    
    # Check if interval is small enough
    interval_size = 0.5 ** depth
    if interval_size < tolerance or depth >= max_depth:
        # Return midpoint as approximate root
        t_mid = 0.5
        # Map to actual parameter range based on subdivision history
        return [t_mid]
    
    # Subdivide at t = 0.5
    left, right = subdivide_bernstein(bernstein_coeffs, 0.5)
    
    # Recursively solve in both halves
    roots = []
    
    left_roots = subdivide_and_solve(left, depth + 1, max_depth, tolerance)
    for t in left_roots:
        roots.append(t * 0.5)  # Map from [0,1] to [0, 0.5]
    
    right_roots = subdivide_and_solve(right, depth + 1, max_depth, tolerance)
    for t in right_roots:
        roots.append(0.5 + t * 0.5)  # Map from [0,1] to [0.5, 1]
    
    return roots

