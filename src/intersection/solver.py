"""
Step 4: Polynomial System Solver

Solve polynomial systems to find intersection parameters.
"""

import numpy as np
from scipy.optimize import fsolve, brentq
from typing import List, Dict, Any


def solve_polynomial_system(system: Dict[str, Any], verbose: bool = False) -> List[Dict[str, float]]:
    """
    Solve the polynomial system to find intersection parameters.
    
    Parameters
    ----------
    system : dict
        Polynomial system from create_intersection_system_*
    verbose : bool
        If True, print solving details
        
    Returns
    -------
    list of dict
        List of solutions with parameters
    """
    if system['type'] == '2D':
        return solve_2d_system(system, verbose)
    elif system['type'] == '3D':
        return solve_3d_system(system, verbose)
    else:
        raise ValueError(f"Unknown system type: {system['type']}")


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

