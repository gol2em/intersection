"""
Parameter Domain Normalization

Utilities for normalizing parameter domains to [0,1]^k using affine transformations.
This is required for LP and PP methods which assume normalized parameter ranges.
"""

import numpy as np
from typing import Callable, List, Tuple, Dict, Any
from .geometry import Hypersurface


def normalize_hypersurface(hypersurface: Hypersurface, verbose: bool = False) -> Tuple[Hypersurface, Dict[str, Any]]:
    """
    Create a normalized version of a hypersurface with parameter domain [0,1]^k.
    
    Uses affine transformation to map original parameter ranges to [0,1]^k.
    
    Parameters
    ----------
    hypersurface : Hypersurface
        Original hypersurface with arbitrary parameter ranges
    verbose : bool
        If True, print normalization details
        
    Returns
    -------
    normalized_hypersurface : Hypersurface
        New hypersurface with parameter ranges [(0,1), (0,1), ..., (0,1)]
    transform_info : dict
        Information about the transformation:
        - 'original_ranges': Original parameter ranges
        - 'scale': Scale factors for each parameter
        - 'offset': Offset for each parameter
        - 'forward': Function to map normalized params to original params
        - 'inverse': Function to map original params to normalized params
        
    Notes
    -----
    The affine transformation is:
        u_original = scale * u_normalized + offset
        u_normalized = (u_original - offset) / scale
        
    For parameter i with range [a_i, b_i]:
        scale_i = b_i - a_i
        offset_i = a_i
        
    Examples
    --------
    >>> # Circle with parameter in [-π, π]
    >>> circle = Hypersurface(
    ...     func=lambda u: np.array([np.cos(u), np.sin(u)]),
    ...     param_ranges=[(-np.pi, np.pi)],
    ...     ambient_dim=2,
    ...     degree=8
    ... )
    >>> normalized_circle, transform = normalize_hypersurface(circle)
    >>> # normalized_circle has param_ranges = [(0, 1)]
    >>> # transform['forward'](0.5) returns 0.0 (middle of [-π, π])
    """
    k = hypersurface.k
    original_ranges = hypersurface.param_ranges
    
    # Compute scale and offset for each parameter
    scales = np.array([b - a for a, b in original_ranges])
    offsets = np.array([a for a, b in original_ranges])
    
    if verbose:
        print(f"\n=== Normalizing Hypersurface Parameter Domain ===")
        print(f"Number of parameters: {k}")
        print(f"Original ranges: {original_ranges}")
        print(f"Scales: {scales}")
        print(f"Offsets: {offsets}")
    
    # Create transformation functions
    def forward_transform(*normalized_params):
        """Map normalized params [0,1]^k to original params."""
        if len(normalized_params) != k:
            raise ValueError(f"Expected {k} parameters, got {len(normalized_params)}")
        normalized = np.array(normalized_params)
        original = scales * normalized + offsets
        return original
    
    def inverse_transform(*original_params):
        """Map original params to normalized params [0,1]^k."""
        if len(original_params) != k:
            raise ValueError(f"Expected {k} parameters, got {len(original_params)}")
        original = np.array(original_params)
        normalized = (original - offsets) / scales
        return normalized
    
    # Create normalized function
    def normalized_func(*normalized_params):
        """Evaluate hypersurface at normalized parameters."""
        original_params = forward_transform(*normalized_params)
        return hypersurface.func(*original_params)
    
    # Create normalized parameter ranges
    normalized_ranges = [(0.0, 1.0) for _ in range(k)]
    
    # Create normalized hypersurface
    normalized_hypersurface = Hypersurface(
        func=normalized_func,
        param_ranges=normalized_ranges,
        ambient_dim=hypersurface.n,
        degree=hypersurface.degree,
        verbose=verbose
    )
    
    # Store transformation info
    transform_info = {
        'original_ranges': original_ranges,
        'normalized_ranges': normalized_ranges,
        'scales': scales,
        'offsets': offsets,
        'forward': forward_transform,
        'inverse': inverse_transform,
    }
    
    if verbose:
        print(f"Normalized ranges: {normalized_ranges}")
        print(f"Transformation: u_original = {scales} * u_normalized + {offsets}")
        print(f"=== Normalization Complete ===\n")
    
    return normalized_hypersurface, transform_info


def denormalize_solutions(solutions: List[Dict[str, float]], 
                         transform_info: Dict[str, Any],
                         verbose: bool = False) -> List[Dict[str, float]]:
    """
    Convert solutions from normalized parameter space back to original space.
    
    Parameters
    ----------
    solutions : list of dict
        Solutions in normalized parameter space [0,1]^k
        Each dict has keys like 'u0', 'u1', ... or 't', 'u', 'v', etc.
    transform_info : dict
        Transformation info from normalize_hypersurface()
    verbose : bool
        If True, print denormalization details
        
    Returns
    -------
    list of dict
        Solutions in original parameter space
        
    Examples
    --------
    >>> solutions_normalized = [{'t': 0.5}, {'t': 0.75}]
    >>> solutions_original = denormalize_solutions(solutions_normalized, transform_info)
    """
    scales = transform_info['scales']
    offsets = transform_info['offsets']
    k = len(scales)
    
    if verbose:
        print(f"\n=== Denormalizing Solutions ===")
        print(f"Number of solutions: {len(solutions)}")
        print(f"Transformation: u_original = {scales} * u_normalized + {offsets}")
    
    denormalized_solutions = []
    
    for sol in solutions:
        # Extract parameter values (handle different naming conventions)
        if k == 1:
            # Try common 1D parameter names
            if 't' in sol:
                normalized_params = [sol['t']]
            elif 'u' in sol:
                normalized_params = [sol['u']]
            elif 'u0' in sol:
                normalized_params = [sol['u0']]
            else:
                raise ValueError(f"Cannot find parameter in solution: {sol}")
        elif k == 2:
            # Try common 2D parameter names
            if 'u' in sol and 'v' in sol:
                normalized_params = [sol['u'], sol['v']]
            elif 'u0' in sol and 'u1' in sol:
                normalized_params = [sol['u0'], sol['u1']]
            else:
                raise ValueError(f"Cannot find parameters in solution: {sol}")
        else:
            # General case: use u0, u1, u2, ...
            normalized_params = [sol[f'u{i}'] for i in range(k)]
        
        # Apply transformation
        normalized_params = np.array(normalized_params)
        original_params = scales * normalized_params + offsets
        
        # Create denormalized solution dict
        denormalized_sol = {}
        if k == 1:
            # Use same key as input
            key = 't' if 't' in sol else ('u' if 'u' in sol else 'u0')
            denormalized_sol[key] = original_params[0]
        elif k == 2:
            # Use same keys as input
            if 'u' in sol and 'v' in sol:
                denormalized_sol['u'] = original_params[0]
                denormalized_sol['v'] = original_params[1]
            else:
                denormalized_sol['u0'] = original_params[0]
                denormalized_sol['u1'] = original_params[1]
        else:
            for i in range(k):
                denormalized_sol[f'u{i}'] = original_params[i]
        
        denormalized_solutions.append(denormalized_sol)
        
        if verbose:
            print(f"  Normalized: {sol} -> Original: {denormalized_sol}")
    
    if verbose:
        print(f"=== Denormalization Complete ===\n")
    
    return denormalized_solutions


def verify_normalization(hypersurface: Hypersurface, 
                        normalized_hypersurface: Hypersurface,
                        transform_info: Dict[str, Any],
                        n_test_points: int = 10,
                        verbose: bool = True) -> bool:
    """
    Verify that normalization preserves the hypersurface geometry.
    
    Tests that evaluating at corresponding points gives the same result.
    
    Parameters
    ----------
    hypersurface : Hypersurface
        Original hypersurface
    normalized_hypersurface : Hypersurface
        Normalized hypersurface
    transform_info : dict
        Transformation info
    n_test_points : int
        Number of test points to check
    verbose : bool
        If True, print verification details
        
    Returns
    -------
    bool
        True if verification passes
    """
    k = hypersurface.k
    forward = transform_info['forward']
    
    if verbose:
        print(f"\n=== Verifying Normalization ===")
        print(f"Testing {n_test_points} random points...")
    
    max_error = 0.0
    
    for i in range(n_test_points):
        # Generate random normalized parameters
        normalized_params = np.random.rand(k)
        
        # Get corresponding original parameters
        original_params = forward(*normalized_params)
        
        # Evaluate both hypersurfaces
        point_normalized = normalized_hypersurface.evaluate(*normalized_params)
        point_original = hypersurface.evaluate(*original_params)
        
        # Check if they match
        error = np.linalg.norm(point_normalized - point_original)
        max_error = max(max_error, error)
        
        if verbose and i < 3:  # Print first 3 test points
            print(f"  Test {i+1}:")
            print(f"    Normalized params: {normalized_params}")
            print(f"    Original params: {original_params}")
            print(f"    Point (normalized): {point_normalized}")
            print(f"    Point (original): {point_original}")
            print(f"    Error: {error:.2e}")
    
    tolerance = 1e-10
    passed = max_error < tolerance
    
    if verbose:
        print(f"\nMax error: {max_error:.2e}")
        print(f"Tolerance: {tolerance:.2e}")
        print(f"Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
        print(f"=== Verification Complete ===\n")
    
    return passed

