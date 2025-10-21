"""
Geometric primitives for n-dimensional intersection computation.

This module defines the generalized geometric objects:
- Hyperplane: Single hyperplane in n-dimensional space
- Line: Line in n-dimensional space (intersection of n-1 hyperplanes)
- Hypersurface: Parametric (n-1)-dimensional manifold in n-dimensional space
"""

import numpy as np
from typing import Callable, Tuple, List, Optional
from .interpolation import interpolate_hypersurface
from .bernstein import polynomial_nd_to_bernstein


class Hyperplane:
    """
    Represents a hyperplane in n-dimensional space.
    
    Equation: a₁x₁ + a₂x₂ + ... + aₙxₙ + d = 0
    
    Attributes:
        coeffs (np.ndarray): Coefficients [a₁, a₂, ..., aₙ]
        d (float): Constant term
        n (int): Dimension of ambient space
    """
    
    def __init__(self, coeffs: np.ndarray, d: float):
        """
        Initialize hyperplane.
        
        Args:
            coeffs: Coefficients [a₁, a₂, ..., aₙ] as array-like
            d: Constant term
            
        Raises:
            ValueError: If all coefficients are zero
        """
        self.coeffs = np.array(coeffs, dtype=float)
        self.d = float(d)
        self.n = len(self.coeffs)
        
        if np.linalg.norm(self.coeffs) < 1e-10:
            raise ValueError("Coefficients cannot all be zero")
    
    def __repr__(self) -> str:
        terms = [f"{self.coeffs[i]:.3f}x{i+1}" for i in range(self.n)]
        return f"Hyperplane({' + '.join(terms)} + {self.d:.3f} = 0)"


class Line:
    """
    Represents a line in n-dimensional space as intersection of (n-1) hyperplanes.
    
    Attributes:
        hyperplanes (List[Hyperplane]): List of n-1 hyperplanes
        n (int): Dimension of ambient space
    """
    
    def __init__(self, hyperplanes: List[Hyperplane]):
        """
        Initialize line from hyperplanes.
        
        Args:
            hyperplanes: List of n-1 hyperplanes defining the line
            
        Raises:
            ValueError: If hyperplanes are invalid or not independent
        """
        if len(hyperplanes) == 0:
            raise ValueError("Need at least one hyperplane")
        
        self.hyperplanes = hyperplanes
        self.n = hyperplanes[0].n
        
        # Verify all hyperplanes have same dimension
        if not all(h.n == self.n for h in hyperplanes):
            raise ValueError("All hyperplanes must have same dimension")
        
        # Verify we have n-1 hyperplanes
        if len(hyperplanes) != self.n - 1:
            raise ValueError(f"Need exactly {self.n - 1} hyperplanes for {self.n}D line, got {len(hyperplanes)}")
        
        # Verify hyperplanes are linearly independent
        normals = np.array([h.coeffs for h in hyperplanes])
        rank = np.linalg.matrix_rank(normals)
        if rank < len(hyperplanes):
            raise ValueError("Hyperplanes must be linearly independent")
    
    def __repr__(self) -> str:
        lines = [f"  {repr(h)}" for h in self.hyperplanes]
        return f"Line({self.n}D):\n" + "\n".join(lines)


class Hypersurface:
    """
    Represents a (n-1)-dimensional parametric hypersurface in n-dimensional space.
    
    The hypersurface is defined by a parametric function:
        f: ℝ^(n-1) → ℝ^n
        (u₁, u₂, ..., u_(n-1)) ↦ (x₁, x₂, ..., xₙ)
    
    Upon initialization:
    1. Automatically interpolates as polynomials
    2. Converts to Bernstein basis
    
    Attributes:
        func (Callable): Parametric function taking (n-1) parameters, returning n-dimensional point
        param_ranges (List[Tuple]): List of (n-1) parameter ranges
        n (int): Ambient space dimension
        k (int): Number of parameters (= n-1)
        degree (int): Polynomial degree for interpolation
        polynomials (List): Interpolated polynomials for each coordinate
        bernstein_coeffs (List): Bernstein coefficients for each coordinate
    """
    
    def __init__(self,
                 func: Callable,
                 param_ranges: List[Tuple[float, float]],
                 ambient_dim: int,
                 degree: int = 5,
                 verbose: bool = False):
        """
        Initialize parametric hypersurface with automatic interpolation.
        
        Args:
            func: Parametric function taking k=(n-1) parameters, returning n-dimensional point
                  Example for 2D curve: lambda u: np.array([u, u**2])
                  Example for 3D surface: lambda u, v: np.array([u, v, u**2 + v**2])
            param_ranges: List of k parameter ranges [(u₁_min, u₁_max), ...]
            ambient_dim: Dimension of ambient space (n)
            degree: Polynomial degree for interpolation
            verbose: Print interpolation details
            
        Raises:
            ValueError: If dimensions are inconsistent
        """
        self.func = func
        self.param_ranges = param_ranges
        self.n = ambient_dim
        self.k = len(param_ranges)
        self.degree = degree
        self.verbose = verbose
        
        # Validate: k must equal n-1 for hypersurface
        if self.k != self.n - 1:
            raise ValueError(f"Hypersurface in {self.n}D space must have {self.n-1} parameters, got {self.k}")
        
        # Step 1: Interpolate as polynomials
        if verbose:
            print(f"\n=== Hypersurface({self.k}→{self.n}D) Initialization ===")
            print(f"Parameters: {self.k}")
            print(f"Ambient dimension: {self.n}")
            print(f"Parameter ranges: {param_ranges}")
            print(f"Polynomial degree: {degree}")
        
        self.polynomials = interpolate_hypersurface(self, degree, verbose)
        
        # Step 2: Convert to Bernstein basis
        if verbose:
            print("\n--- Converting to Bernstein basis ---")
        
        self.bernstein_coeffs = []
        for i, poly in enumerate(self.polynomials):
            if verbose:
                print(f"Converting coordinate {i+1}/{self.n}...")
            bern = polynomial_nd_to_bernstein(poly, self.k, verbose)
            self.bernstein_coeffs.append(bern)
        
        if verbose:
            print(f"\nHypersurface initialization complete!")
    
    def evaluate(self, *params) -> np.ndarray:
        """
        Evaluate hypersurface at parameter values using original function.
        
        Args:
            *params: k=(n-1) parameter values
        
        Returns:
            Point in n-dimensional space
            
        Raises:
            ValueError: If wrong number of parameters
        """
        if len(params) != self.k:
            raise ValueError(f"Expected {self.k} parameters, got {len(params)}")
        
        result = self.func(*params)
        result = np.array(result, dtype=float)
        
        if len(result) != self.n:
            raise ValueError(f"Function must return {self.n}-dimensional point, got {len(result)}")
        
        return result
    
    def sample(self, n_samples: int = 50) -> np.ndarray:
        """
        Sample points on the hypersurface.
        
        Args:
            n_samples: Number of samples per parameter dimension
        
        Returns:
            Array of sampled points, shape depends on k
        """
        # Create grid of parameter values
        param_grids = []
        for u_min, u_max in self.param_ranges:
            param_grids.append(np.linspace(u_min, u_max, n_samples))
        
        # Create meshgrid
        if self.k == 1:
            # 1D case (curve)
            params = param_grids[0]
            points = np.array([self.evaluate(p) for p in params])
        elif self.k == 2:
            # 2D case (surface)
            U, V = np.meshgrid(param_grids[0], param_grids[1])
            points = np.array([[self.evaluate(u, v) for u, v in zip(u_row, v_row)] 
                              for u_row, v_row in zip(U, V)])
        else:
            # Higher dimensional case - sample along diagonal or use other strategy
            # For simplicity, sample along each parameter axis
            points = []
            for i, param_grid in enumerate(param_grids):
                # Fix other parameters at midpoint
                mid_params = [(u_min + u_max) / 2 for u_min, u_max in self.param_ranges]
                for p in param_grid:
                    params = mid_params.copy()
                    params[i] = p
                    points.append(self.evaluate(*params))
            points = np.array(points)
        
        return points
    
    def __repr__(self) -> str:
        return f"Hypersurface({self.k}→{self.n}D, degree={self.degree})"

