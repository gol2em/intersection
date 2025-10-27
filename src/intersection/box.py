"""
Box Class for LP/PP Subdivision Methods

Tracks domain transformations at multiple levels:
1. Original domain → [0,1]^k (normalization)
2. [0,1]^k → current box (subdivision)

The de Casteljau algorithm always works on [0,1]^k, so we need to track
the mapping from [0,1]^k (Bernstein space) to the actual box ranges.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class Box:
    """
    Represents a k-dimensional box in parameter space with multi-level domain tracking.
    
    Domain Hierarchy:
    ----------------
    1. **Original domain**: The actual parameter ranges of the hypersurface
       Example: u ∈ [-π, π]
       
    2. **Normalized domain**: Always [0,1]^k after normalization
       Example: t ∈ [0, 1] where t = (u + π)/(2π)
       
    3. **Current box**: Sub-box of [0,1]^k after subdivision
       Example: t ∈ [0.25, 0.5] (a sub-box of [0,1])
       
    4. **Bernstein domain**: Always [0,1]^k for de Casteljau
       Example: s ∈ [0, 1] where s = (t - 0.25)/(0.5 - 0.25)
    
    Transformations:
    ---------------
    - original_to_normalized: Uses normalization transform
    - normalized_to_box: Maps [0,1]^k to current box ranges
    - box_to_bernstein: Maps current box to [0,1]^k for de Casteljau
    - bernstein_to_original: Composition of all transformations
    
    Attributes
    ----------
    k : int
        Number of parameters (dimension)
    ranges : List[Tuple[float, float]]
        Current box ranges in normalized [0,1]^k space
        Example: [(0.25, 0.5), (0.0, 0.5)] for 2D box
    normalization_transform : Dict[str, Any] or None
        Transform from original to normalized domain (from normalize_hypersurface)
        Contains 'scales', 'offsets', 'forward', 'inverse'
    parent_box : Box or None
        Parent box (for tracking subdivision hierarchy)
    depth : int
        Subdivision depth (0 for root box)
    """
    
    def __init__(self, 
                 k: int,
                 ranges: Optional[List[Tuple[float, float]]] = None,
                 normalization_transform: Optional[Dict[str, Any]] = None,
                 parent_box: Optional['Box'] = None,
                 depth: int = 0):
        """
        Initialize a box.
        
        Parameters
        ----------
        k : int
            Number of parameters (dimension)
        ranges : List[Tuple[float, float]], optional
            Box ranges in normalized [0,1]^k space
            If None, defaults to [(0,1), (0,1), ..., (0,1)]
        normalization_transform : Dict[str, Any], optional
            Transform from original to normalized domain
            If None, assumes no normalization (original = normalized)
        parent_box : Box, optional
            Parent box in subdivision hierarchy
        depth : int
            Subdivision depth
        """
        self.k = k
        self.ranges = ranges if ranges is not None else [(0.0, 1.0) for _ in range(k)]
        self.normalization_transform = normalization_transform
        self.parent_box = parent_box
        self.depth = depth
        
        # Validate ranges
        if len(self.ranges) != k:
            raise ValueError(f"Expected {k} ranges, got {len(self.ranges)}")
        
        for i, (a, b) in enumerate(self.ranges):
            if a >= b:
                raise ValueError(f"Invalid range [{a}, {b}] for parameter {i}")
    
    def get_center(self) -> np.ndarray:
        """
        Get center point of the box in normalized [0,1]^k space.
        
        Returns
        -------
        np.ndarray
            Center point, shape (k,)
        """
        return np.array([(a + b) / 2 for a, b in self.ranges])
    
    def get_widths(self) -> np.ndarray:
        """
        Get widths of the box in normalized [0,1]^k space.
        
        Returns
        -------
        np.ndarray
            Widths, shape (k,)
        """
        return np.array([b - a for a, b in self.ranges])
    
    def get_volume(self) -> float:
        """
        Get volume (k-dimensional measure) of the box.
        
        Returns
        -------
        float
            Volume in normalized space
        """
        return np.prod(self.get_widths())
    
    def bernstein_to_box(self, *bernstein_params) -> np.ndarray:
        """
        Map from Bernstein domain [0,1]^k to current box in normalized space.
        
        This is the transformation used AFTER de Casteljau subdivision.
        De Casteljau gives coefficients on [0,1]^k, and we need to map
        them to the actual box ranges.
        
        Parameters
        ----------
        *bernstein_params : float
            Parameters in [0,1]^k (Bernstein space)
            
        Returns
        -------
        np.ndarray
            Parameters in current box (normalized space)
            
        Examples
        --------
        >>> box = Box(k=2, ranges=[(0.25, 0.5), (0.0, 0.5)])
        >>> box.bernstein_to_box(0.0, 0.0)  # Returns (0.25, 0.0)
        >>> box.bernstein_to_box(1.0, 1.0)  # Returns (0.5, 0.5)
        >>> box.bernstein_to_box(0.5, 0.5)  # Returns (0.375, 0.25)
        """
        if len(bernstein_params) != self.k:
            raise ValueError(f"Expected {self.k} parameters, got {len(bernstein_params)}")
        
        bernstein = np.array(bernstein_params)
        box_params = np.zeros(self.k)
        
        for i in range(self.k):
            a, b = self.ranges[i]
            box_params[i] = a + bernstein[i] * (b - a)
        
        return box_params
    
    def box_to_bernstein(self, *box_params) -> np.ndarray:
        """
        Map from current box to Bernstein domain [0,1]^k.
        
        This is the inverse of bernstein_to_box.
        
        Parameters
        ----------
        *box_params : float
            Parameters in current box (normalized space)
            
        Returns
        -------
        np.ndarray
            Parameters in [0,1]^k (Bernstein space)
        """
        if len(box_params) != self.k:
            raise ValueError(f"Expected {self.k} parameters, got {len(box_params)}")
        
        box = np.array(box_params)
        bernstein_params = np.zeros(self.k)
        
        for i in range(self.k):
            a, b = self.ranges[i]
            bernstein_params[i] = (box[i] - a) / (b - a)
        
        return bernstein_params
    
    def normalized_to_original(self, *normalized_params) -> np.ndarray:
        """
        Map from normalized [0,1]^k space to original parameter space.
        
        Uses the normalization transform if available.
        
        Parameters
        ----------
        *normalized_params : float
            Parameters in normalized [0,1]^k space
            
        Returns
        -------
        np.ndarray
            Parameters in original space
        """
        if self.normalization_transform is None:
            # No normalization, original = normalized
            return np.array(normalized_params)
        
        # Use forward transform: original = scale * normalized + offset
        normalized = np.array(normalized_params)
        scales = self.normalization_transform['scales']
        offsets = self.normalization_transform['offsets']
        
        return scales * normalized + offsets
    
    def original_to_normalized(self, *original_params) -> np.ndarray:
        """
        Map from original parameter space to normalized [0,1]^k space.
        
        Uses the normalization transform if available.
        
        Parameters
        ----------
        *original_params : float
            Parameters in original space
            
        Returns
        -------
        np.ndarray
            Parameters in normalized [0,1]^k space
        """
        if self.normalization_transform is None:
            # No normalization, original = normalized
            return np.array(original_params)
        
        # Use inverse transform: normalized = (original - offset) / scale
        original = np.array(original_params)
        scales = self.normalization_transform['scales']
        offsets = self.normalization_transform['offsets']
        
        return (original - offsets) / scales
    
    def bernstein_to_original(self, *bernstein_params) -> np.ndarray:
        """
        Map from Bernstein domain [0,1]^k to original parameter space.
        
        This is the composition:
        Bernstein [0,1]^k → Box → Normalized [0,1]^k → Original
        
        Parameters
        ----------
        *bernstein_params : float
            Parameters in [0,1]^k (Bernstein space)
            
        Returns
        -------
        np.ndarray
            Parameters in original space
            
        Examples
        --------
        >>> # Box in normalized space: [0.25, 0.5] x [0.0, 0.5]
        >>> # Original space: [-π, π] x [-1, 1]
        >>> box = Box(k=2, ranges=[(0.25, 0.5), (0.0, 0.5)],
        ...           normalization_transform=transform)
        >>> # Bernstein (0.5, 0.5) → Box (0.375, 0.25) → Original (?, ?)
        >>> box.bernstein_to_original(0.5, 0.5)
        """
        # Step 1: Bernstein → Box (in normalized space)
        box_params = self.bernstein_to_box(*bernstein_params)
        
        # Step 2: Normalized → Original
        original_params = self.normalized_to_original(*box_params)
        
        return original_params
    
    def subdivide(self, axis: int, split_point: float = 0.5) -> Tuple['Box', 'Box']:
        """
        Subdivide the box along a given axis.
        
        Parameters
        ----------
        axis : int
            Axis to subdivide (0 to k-1)
        split_point : float
            Split point in [0, 1] (default: 0.5 for midpoint)
            
        Returns
        -------
        left_box : Box
            Left sub-box
        right_box : Box
            Right sub-box
            
        Examples
        --------
        >>> box = Box(k=2, ranges=[(0.0, 1.0), (0.0, 1.0)])
        >>> left, right = box.subdivide(axis=0, split_point=0.5)
        >>> left.ranges  # [(0.0, 0.5), (0.0, 1.0)]
        >>> right.ranges  # [(0.5, 1.0), (0.0, 1.0)]
        """
        if axis < 0 or axis >= self.k:
            raise ValueError(f"Invalid axis {axis} for {self.k}-dimensional box")
        
        if split_point <= 0 or split_point >= 1:
            raise ValueError(f"Split point must be in (0, 1), got {split_point}")
        
        # Compute split point in box coordinates
        a, b = self.ranges[axis]
        split_value = a + split_point * (b - a)
        
        # Create left and right ranges
        left_ranges = list(self.ranges)
        left_ranges[axis] = (a, split_value)
        
        right_ranges = list(self.ranges)
        right_ranges[axis] = (split_value, b)
        
        # Create sub-boxes
        left_box = Box(
            k=self.k,
            ranges=left_ranges,
            normalization_transform=self.normalization_transform,
            parent_box=self,
            depth=self.depth + 1
        )
        
        right_box = Box(
            k=self.k,
            ranges=right_ranges,
            normalization_transform=self.normalization_transform,
            parent_box=self,
            depth=self.depth + 1
        )
        
        return left_box, right_box
    
    def __repr__(self) -> str:
        """String representation of the box."""
        ranges_str = ", ".join([f"[{a:.4f}, {b:.4f}]" for a, b in self.ranges])
        return f"Box(k={self.k}, ranges=[{ranges_str}], depth={self.depth})"

