"""
Step 3: Polynomial System Formation

Create polynomial systems for line-curve/surface intersection.
"""

import numpy as np
from typing import Dict, Any


def create_intersection_system_2d(line, bern_x: np.ndarray, bern_y: np.ndarray, 
                                   verbose: bool = False) -> Dict[str, Any]:
    """
    Create polynomial system for 2D line-curve intersection.
    
    The intersection condition is:
    Line(s) = Curve(t)
    point + s * direction = (x(t), y(t))
    
    This gives us 2 equations:
    x0 + s * dx = x(t)
    y0 + s * dy = y(t)
    
    We can eliminate s to get a single polynomial equation in t.
    
    Parameters
    ----------
    line : Line2D
        The straight line
    bern_x : np.ndarray
        Bernstein coefficients for x(t)
    bern_y : np.ndarray
        Bernstein coefficients for y(t)
    verbose : bool
        If True, print system details
        
    Returns
    -------
    dict
        Polynomial system representation
    """
    x0, y0 = line.point
    dx, dy = line.direction
    
    if verbose:
        print(f"Line: point=({x0:.4f}, {y0:.4f}), direction=({dx:.4f}, {dy:.4f})")
        print(f"Curve x(t) Bernstein coefficients: {bern_x}")
        print(f"Curve y(t) Bernstein coefficients: {bern_y}")
    
    # From the two equations, eliminate s:
    # s = (x(t) - x0) / dx = (y(t) - y0) / dy
    # Cross multiply: dy * (x(t) - x0) = dx * (y(t) - y0)
    # dy * x(t) - dy * x0 = dx * y(t) - dx * y0
    # dy * x(t) - dx * y(t) = dy * x0 - dx * y0
    
    # This is a polynomial equation in t
    # We need to convert Bernstein polynomials to power basis for manipulation
    from .bernstein import bernstein_to_polynomial
    
    poly_x = bernstein_to_polynomial(bern_x)
    poly_y = bernstein_to_polynomial(bern_y)
    
    # Form the equation: dy * x(t) - dx * y(t) - (dy * x0 - dx * y0) = 0
    equation = dy * poly_x - dx * poly_y
    equation = equation - (dy * x0 - dx * y0)
    
    if verbose:
        print(f"\nIntersection equation (power basis):")
        print(f"  {dy:.4f} * x(t) - {dx:.4f} * y(t) - {dy * x0 - dx * y0:.4f} = 0")
        print(f"  Polynomial coefficients: {equation.coef}")
        print(f"  Degree: {equation.degree()}")
    
    system = {
        'type': '2D',
        'equation': equation,
        'line': line,
        'bern_x': bern_x,
        'bern_y': bern_y,
        'poly_x': poly_x,
        'poly_y': poly_y
    }
    
    return system


def create_intersection_system_3d(line, bern_x: np.ndarray, bern_y: np.ndarray, 
                                   bern_z: np.ndarray, verbose: bool = False) -> Dict[str, Any]:
    """
    Create polynomial system for 3D line-surface intersection.
    
    The intersection condition is:
    Line(s) = Surface(u, v)
    point + s * direction = (x(u,v), y(u,v), z(u,v))
    
    This gives us 3 equations in 3 unknowns (s, u, v):
    x0 + s * dx = x(u, v)
    y0 + s * dy = y(u, v)
    z0 + s * dz = z(u, v)
    
    We can eliminate s to get 2 polynomial equations in (u, v).
    
    Parameters
    ----------
    line : Line3D
        The straight line
    bern_x : np.ndarray
        Bernstein coefficients for x(u,v)
    bern_y : np.ndarray
        Bernstein coefficients for y(u,v)
    bern_z : np.ndarray
        Bernstein coefficients for z(u,v)
    verbose : bool
        If True, print system details
        
    Returns
    -------
    dict
        Polynomial system representation
    """
    x0, y0, z0 = line.point
    dx, dy, dz = line.direction
    
    if verbose:
        print(f"Line: point=({x0:.4f}, {y0:.4f}, {z0:.4f}), direction=({dx:.4f}, {dy:.4f}, {dz:.4f})")
        print(f"Surface x(u,v) Bernstein coefficient matrix shape: {bern_x.shape}")
        print(f"Surface y(u,v) Bernstein coefficient matrix shape: {bern_y.shape}")
        print(f"Surface z(u,v) Bernstein coefficient matrix shape: {bern_z.shape}")
    
    # Eliminate s from the three equations:
    # From equations 1 and 2: dy * (x(u,v) - x0) = dx * (y(u,v) - y0)
    # From equations 1 and 3: dz * (x(u,v) - x0) = dx * (z(u,v) - z0)
    
    # This gives us two polynomial equations in (u, v)
    # Equation 1: dy * x(u,v) - dx * y(u,v) = dy * x0 - dx * y0
    # Equation 2: dz * x(u,v) - dx * z(u,v) = dz * x0 - dx * z0
    
    if verbose:
        print(f"\nIntersection system (2 equations in u, v):")
        print(f"  Eq1: {dy:.4f} * x(u,v) - {dx:.4f} * y(u,v) = {dy * x0 - dx * y0:.4f}")
        print(f"  Eq2: {dz:.4f} * x(u,v) - {dx:.4f} * z(u,v) = {dz * x0 - dx * z0:.4f}")
    
    system = {
        'type': '3D',
        'line': line,
        'bern_x': bern_x,
        'bern_y': bern_y,
        'bern_z': bern_z,
        'equation1_coeffs': (dy, -dx, dy * x0 - dx * y0),  # dy*x - dx*y = const
        'equation2_coeffs': (dz, -dx, dz * x0 - dx * z0),  # dz*x - dx*z = const
    }
    
    return system


def evaluate_system_2d(system: Dict[str, Any], t: float) -> float:
    """
    Evaluate the 2D intersection equation at parameter t.
    
    Returns the residual (should be 0 at intersection).
    """
    return system['equation'](t)


def evaluate_system_3d(system: Dict[str, Any], u: float, v: float) -> tuple:
    """
    Evaluate the 3D intersection system at parameters (u, v).
    
    Returns tuple of residuals (should be (0, 0) at intersection).
    """
    from .interpolation import evaluate_2d_polynomial
    
    x_val = evaluate_2d_polynomial(system['bern_x'], np.array([[u]]), np.array([[v]]))[0, 0]
    y_val = evaluate_2d_polynomial(system['bern_y'], np.array([[u]]), np.array([[v]]))[0, 0]
    z_val = evaluate_2d_polynomial(system['bern_z'], np.array([[u]]), np.array([[v]]))[0, 0]
    
    c1_x, c1_y, const1 = system['equation1_coeffs']
    c2_x, c2_z, const2 = system['equation2_coeffs']
    
    residual1 = c1_x * x_val + c1_y * y_val - const1
    residual2 = c2_x * x_val + c2_z * z_val - const2
    
    return residual1, residual2

