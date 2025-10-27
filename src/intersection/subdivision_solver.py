"""
Subdivision-based Polynomial System Solver

Implements PP (Projected Polyhedron), LP (Linear Programming), and Hybrid methods
for solving polynomial systems using Bernstein subdivision.

Based on:
- Sherbrooke & Patrikalakis (1993): "Computation of the solutions of nonlinear 
  polynomial systems"
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from .box import Box
from .de_casteljau import extract_subbox_with_box
from .normalization import normalize_hypersurface


class BoundingMethod(Enum):
    """Method for computing bounds on polynomial range."""
    PP = "pp"  # Projected Polyhedron (min/max of Bernstein coefficients)
    LP = "lp"  # Linear Programming (tighter bounds)
    HYBRID = "hybrid"  # Use PP first, then LP for refinement


@dataclass
class SolverConfig:
    """Configuration for subdivision solver."""
    method: BoundingMethod = BoundingMethod.PP
    tolerance: float = 1e-6  # Size threshold for claiming a root
    crit: float = 0.8  # Critical ratio for subdivision (0.8 = 80% of original)
    max_depth: int = 30  # Maximum subdivision depth
    max_solutions: int = 1000  # Maximum number of solutions to find
    subdivision_tolerance: float = 1e-10  # Tolerance for sub-box extraction
    verbose: bool = False


@dataclass
class SubdivisionBox:
    """Box in subdivision queue with associated data."""
    box: Box  # Domain box
    coeffs: List[np.ndarray]  # Bernstein coefficients for each equation
    depth: int  # Subdivision depth


class SubdivisionSolver:
    """
    Subdivision-based solver for polynomial systems.
    
    Algorithm:
    1. For a normalized box, find a sub-box that contains all roots
    2. If the sub-box is sufficiently small, claim there is a root and return midpoint
    3. If any dimension is not small enough (> CRIT * original size), subdivide in half
    4. Repeat from step 1 for each sub-box
    
    Parameters
    ----------
    config : SolverConfig
        Solver configuration
        
    Attributes
    ----------
    solutions : List[np.ndarray]
        Found solutions (in normalized [0,1]^k space)
    stats : Dict[str, int]
        Statistics (boxes processed, subdivisions, etc.)
    """
    
    def __init__(self, config: SolverConfig = None):
        self.config = config if config is not None else SolverConfig()
        self.solutions: List[np.ndarray] = []
        self.stats = {
            'boxes_processed': 0,
            'boxes_pruned': 0,
            'subdivisions': 0,
            'solutions_found': 0,
        }
    
    def solve(self, 
              equation_coeffs: List[np.ndarray],
              k: int,
              normalization_transform: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        """
        Solve polynomial system using subdivision.
        
        Parameters
        ----------
        equation_coeffs : List[np.ndarray]
            List of Bernstein coefficient arrays, one per equation
            Each array is k-dimensional (tensor product)
        k : int
            Number of parameters (dimension)
        normalization_transform : Dict, optional
            Normalization transform for domain tracking
            
        Returns
        -------
        List[np.ndarray]
            Solutions in normalized [0,1]^k space
            Each solution is a k-dimensional array
            
        Examples
        --------
        >>> # Solve f(t) = 0 where f has Bernstein coefficients [1, 0, -1]
        >>> solver = SubdivisionSolver()
        >>> solutions = solver.solve([np.array([1.0, 0.0, -1.0])], k=1)
        """
        # Reset state
        self.solutions = []
        self.stats = {k: 0 for k in self.stats.keys()}
        
        # Create initial box
        initial_box = Box(
            k=k,
            ranges=[(0.0, 1.0) for _ in range(k)],
            normalization_transform=normalization_transform,
            depth=0
        )
        
        # Initialize queue with initial box
        queue = [SubdivisionBox(
            box=initial_box,
            coeffs=equation_coeffs,
            depth=0
        )]
        
        if self.config.verbose:
            print(f"\n{'='*80}")
            print(f"SUBDIVISION SOLVER ({self.config.method.value.upper()} method)")
            print(f"{'='*80}")
            print(f"Number of equations: {len(equation_coeffs)}")
            print(f"Parameter dimension: {k}")
            print(f"Tolerance: {self.config.tolerance}")
            print(f"Critical ratio: {self.config.crit}")
            print(f"Max depth: {self.config.max_depth}")
            print(f"{'='*80}\n")
        
        # Process queue
        while queue and len(self.solutions) < self.config.max_solutions:
            current = queue.pop(0)
            
            # Process this box
            sub_boxes = self._process_box(current)
            
            # Add sub-boxes to queue
            if sub_boxes is not None:
                queue.extend(sub_boxes)
        
        if self.config.verbose:
            print(f"\n{'='*80}")
            print(f"SOLVER STATISTICS")
            print(f"{'='*80}")
            for key, value in self.stats.items():
                print(f"{key}: {value}")
            print(f"{'='*80}\n")
        
        return self.solutions
    
    def _process_box(self, sub_box: SubdivisionBox) -> Optional[List[SubdivisionBox]]:
        """
        Process a single box.
        
        Returns
        -------
        Optional[List[SubdivisionBox]]
            - None if box is pruned or contains a solution
            - List of sub-boxes if subdivision is needed
        """
        self.stats['boxes_processed'] += 1
        
        if self.config.verbose and self.stats['boxes_processed'] % 100 == 0:
            print(f"Processed {self.stats['boxes_processed']} boxes, "
                  f"found {self.stats['solutions_found']} solutions, "
                  f"pruned {self.stats['boxes_pruned']} boxes")
        
        # Step 1: Find sub-box that contains all roots
        containing_box_ranges = self._find_containing_subbox(
            sub_box.coeffs, sub_box.box
        )
        
        # Check if we can prune this box
        if containing_box_ranges is None:
            self.stats['boxes_pruned'] += 1
            return None
        
        # Step 2: Check if sub-box is sufficiently small
        if self._is_sufficiently_small(containing_box_ranges):
            # Claim there is a root, return midpoint
            midpoint = self._compute_midpoint(containing_box_ranges, sub_box.box)
            self.solutions.append(midpoint)
            self.stats['solutions_found'] += 1
            
            if self.config.verbose:
                print(f"  [Depth {sub_box.depth}] Found solution at {midpoint}")
            
            return None
        
        # Step 3: Check depth limit
        if sub_box.depth >= self.config.max_depth:
            # Max depth reached, claim solution anyway
            midpoint = self._compute_midpoint(containing_box_ranges, sub_box.box)
            self.solutions.append(midpoint)
            self.stats['solutions_found'] += 1
            
            if self.config.verbose:
                print(f"  [Depth {sub_box.depth}] Max depth reached, "
                      f"claiming solution at {midpoint}")
            
            return None
        
        # Step 4: Subdivide along dimensions that are not small enough
        return self._subdivide_box(sub_box, containing_box_ranges)
    
    def _find_containing_subbox(self,
                                 coeffs: List[np.ndarray],
                                 box: Box) -> Optional[List[Tuple[float, float]]]:
        """
        Find sub-box of [0,1]^k that contains all roots.
        
        This is method-dependent:
        - PP: Use min/max of Bernstein coefficients
        - LP: Use linear programming for tighter bounds
        - Hybrid: Use PP first, then LP if needed
        
        Returns
        -------
        Optional[List[Tuple[float, float]]]
            Sub-box ranges in [0,1]^k, or None if no roots exist
        """
        if self.config.method == BoundingMethod.PP:
            return self._find_containing_subbox_pp(coeffs, box)
        elif self.config.method == BoundingMethod.LP:
            return self._find_containing_subbox_lp(coeffs, box)
        else:  # HYBRID
            return self._find_containing_subbox_hybrid(coeffs, box)

    def _find_containing_subbox_pp(self,
                                     coeffs: List[np.ndarray],
                                     box: Box) -> Optional[List[Tuple[float, float]]]:
        """
        PP method: Use convex hull intersection to find tighter bounds.

        For each dimension j:
        1. Extract j-th component from each equation (univariate polynomial)
        2. Compute convex hull intersection with x-axis for each equation
        3. Intersect all ranges to get tightest bound for dimension j
        4. Construct k-dimensional bounding box

        Returns tighter sub-box or None if no roots exist.
        """
        from .convex_hull import find_root_box_pp_nd

        # Use enhanced PP method with convex hull intersection
        result = find_root_box_pp_nd(coeffs, box.k, tolerance=self.config.subdivision_tolerance)

        return result

    def _find_containing_subbox_lp(self,
                                    coeffs: List[np.ndarray],
                                    box: Box) -> Optional[List[Tuple[float, float]]]:
        """
        LP method: Use linear programming for tighter bounds.

        TODO: Implement LP-based bounding
        For now, fall back to PP method.
        """
        # Placeholder: use PP method
        return self._find_containing_subbox_pp(coeffs, box)

    def _find_containing_subbox_hybrid(self,
                                        coeffs: List[np.ndarray],
                                        box: Box) -> Optional[List[Tuple[float, float]]]:
        """
        Hybrid method: Use PP first, then LP for refinement.

        TODO: Implement hybrid approach
        For now, fall back to PP method.
        """
        # Placeholder: use PP method
        return self._find_containing_subbox_pp(coeffs, box)

    def _is_sufficiently_small(self, ranges: List[Tuple[float, float]]) -> bool:
        """
        Check if sub-box is sufficiently small to claim a root.

        A box is sufficiently small if all dimensions are smaller than tolerance.
        """
        for t_min, t_max in ranges:
            if (t_max - t_min) > self.config.tolerance:
                return False
        return True

    def _compute_midpoint(self,
                          ranges: List[Tuple[float, float]],
                          box: Box) -> np.ndarray:
        """
        Compute midpoint of sub-box in normalized [0,1]^k space.

        Parameters
        ----------
        ranges : List[Tuple[float, float]]
            Sub-box ranges in [0,1]^k (relative to current box)
        box : Box
            Current box

        Returns
        -------
        np.ndarray
            Midpoint in normalized [0,1]^k space
        """
        # Compute midpoint in Bernstein space [0,1]^k
        bernstein_midpoint = np.array([(t_min + t_max) / 2 for t_min, t_max in ranges])

        # Map to normalized space
        normalized_midpoint = np.zeros(box.k)
        for i in range(box.k):
            box_min, box_max = box.ranges[i]
            t_mid = bernstein_midpoint[i]
            normalized_midpoint[i] = box_min + t_mid * (box_max - box_min)

        return normalized_midpoint

    def _subdivide_box(self,
                       sub_box: SubdivisionBox,
                       containing_ranges: List[Tuple[float, float]]) -> List[SubdivisionBox]:
        """
        Subdivide box along dimensions that are not small enough.

        Strategy:
        - For each dimension, check if PP-reduced size > CRIT
        - If yes, subdivide in half along that dimension (PP didn't help much)
        - If no (PP reduced by ≥ (1-CRIT)), extract sub-box and apply PP again
        - Create 2^n sub-boxes where n is number of dimensions to subdivide

        Parameters
        ----------
        sub_box : SubdivisionBox
            Current box to subdivide
        containing_ranges : List[Tuple[float, float]]
            Sub-box ranges that contain roots (in [0,1]^k normalized space)

        Returns
        -------
        List[SubdivisionBox]
            List of sub-boxes to process
        """
        k = sub_box.box.k

        # Determine which dimensions need subdivision
        # containing_ranges are in [0,1] space, so original size is 1.0
        dims_to_subdivide = []
        for i, (t_min, t_max) in enumerate(containing_ranges):
            size = t_max - t_min
            # If size > CRIT, PP didn't reduce enough (< (1-CRIT)% reduction)
            # Example: CRIT=0.8 means if size > 0.8, PP reduced < 20%, so subdivide
            if size > self.config.crit:
                dims_to_subdivide.append(i)

        if not dims_to_subdivide:
            # PP successfully reduced all dimensions by ≥ (1-CRIT)
            # Extract the tighter sub-box and apply PP again (don't subdivide!)
            # This is the key to PP method efficiency
            sub_coeffs_list = []
            for eq_coeffs in sub_box.coeffs:
                sub_coeffs, new_box = extract_subbox_with_box(
                    eq_coeffs,
                    sub_box.box,
                    containing_ranges,
                    tolerance=self.config.subdivision_tolerance,
                    verbose=False
                )
                sub_coeffs_list.append(sub_coeffs)

            # Return single sub-box with tighter bounds (same depth, not subdivided)
            return [SubdivisionBox(
                box=new_box,
                coeffs=sub_coeffs_list,
                depth=sub_box.depth  # Same depth - we're tightening, not subdividing
            )]

        # Generate all combinations of subdivisions
        # For each dimension to subdivide, we create 2 sub-boxes (left/right)
        n_subdivisions = len(dims_to_subdivide)
        n_boxes = 2 ** n_subdivisions

        sub_boxes = []

        for box_idx in range(n_boxes):
            # Determine which half to take for each dimension
            # Use binary representation of box_idx
            sub_ranges = list(containing_ranges)  # Copy

            for dim_idx, axis in enumerate(dims_to_subdivide):
                t_min, t_max = containing_ranges[axis]
                t_mid = (t_min + t_max) / 2

                # Check bit dim_idx of box_idx
                if (box_idx >> dim_idx) & 1:
                    # Right half
                    sub_ranges[axis] = (t_mid, t_max)
                else:
                    # Left half
                    sub_ranges[axis] = (t_min, t_mid)

            # Extract coefficients for this sub-box
            sub_coeffs_list = []
            for eq_coeffs in sub_box.coeffs:
                sub_coeffs, new_box = extract_subbox_with_box(
                    eq_coeffs,
                    sub_box.box,
                    sub_ranges,
                    tolerance=self.config.subdivision_tolerance,
                    verbose=False
                )
                sub_coeffs_list.append(sub_coeffs)

            # Create sub-box (use the box from last extraction, they should all be the same)
            sub_boxes.append(SubdivisionBox(
                box=new_box,
                coeffs=sub_coeffs_list,
                depth=sub_box.depth + 1
            ))

        self.stats['subdivisions'] += 1

        if self.config.verbose and sub_box.depth < 5:
            print(f"  [Depth {sub_box.depth}] Subdividing along {len(dims_to_subdivide)} "
                  f"dimension(s): {dims_to_subdivide} -> {len(sub_boxes)} sub-boxes")

        return sub_boxes


def solve_with_subdivision(
    equation_coeffs: List[np.ndarray],
    k: int,
    method: str = 'pp',
    tolerance: float = 1e-6,
    crit: float = 0.8,
    max_depth: int = 30,
    subdivision_tolerance: float = 1e-10,
    normalization_transform: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> List[np.ndarray]:
    """
    Convenience function to solve polynomial system using subdivision.

    Parameters
    ----------
    equation_coeffs : List[np.ndarray]
        Bernstein coefficients for each equation
    k : int
        Number of parameters
    method : str
        Bounding method: 'pp', 'lp', or 'hybrid'
    tolerance : float
        Size threshold for claiming a root in parameter space
    crit : float
        Critical ratio for subdivision (default: 0.8)
    max_depth : int
        Maximum subdivision depth
    subdivision_tolerance : float
        Numerical tolerance for zero detection in function value space (default: 1e-10)
    normalization_transform : Dict, optional
        Normalization transform
    verbose : bool
        Print progress

    Returns
    -------
    List[np.ndarray]
        Solutions in normalized [0,1]^k space

    Examples
    --------
    >>> # Solve f(t) = 0 where f has Bernstein coefficients [1, 0, -1]
    >>> solutions = solve_with_subdivision(
    ...     [np.array([1.0, 0.0, -1.0])],
    ...     k=1,
    ...     method='pp',
    ...     verbose=True
    ... )
    """
    # Create config
    method_enum = BoundingMethod[method.upper()]
    config = SolverConfig(
        method=method_enum,
        tolerance=tolerance,
        crit=crit,
        max_depth=max_depth,
        subdivision_tolerance=subdivision_tolerance,
        verbose=verbose
    )

    # Create solver and solve
    solver = SubdivisionSolver(config)
    return solver.solve(equation_coeffs, k, normalization_transform)


