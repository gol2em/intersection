"""
Subdivision-based Polynomial System Solver (Redesigned)

New architecture matching visualize_2d_step_by_step.py:
- Work with PP bounds directly in [0,1]^k space
- Only extract coefficients when subdividing/tightening
- Never re-compute PP on extracted coefficients
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .box import Box
from .de_casteljau import extract_subbox_with_box
from .convex_hull import find_root_box_pp_nd


class BoundingMethod(Enum):
    """Method for computing bounds on polynomial range."""
    PP = "pp"
    LP = "lp"
    HYBRID = "hybrid"


@dataclass
class SolverConfig:
    """Configuration for subdivision solver."""
    method: BoundingMethod = BoundingMethod.PP
    tolerance: float = 1e-6  # Size threshold for claiming a root
    crit: float = 0.8  # Critical ratio for subdivision
    max_depth: int = 30  # Maximum subdivision depth
    max_solutions: int = 1000  # Maximum number of solutions to find
    subdivision_tolerance: float = 1e-10  # Tolerance for sub-box extraction
    verbose: bool = False


class SubdivisionSolver:
    """
    Subdivision-based solver for polynomial systems.

    Redesigned to match visualize_2d_step_by_step.py logic:
    - Recursive solving with bounding method (PP/LP/Hybrid)
    - Subdivide only first dimension that needs it
    - Preserve tightened bounds across subdivisions

    The solver is designed to be method-agnostic:
    - PP, LP, and Hybrid methods differ only in how they compute bounding boxes
    - All other logic (subdivision, tightening, recursion) is identical
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
        self.normalization_transform = None
    
    def solve(self,
              equation_coeffs: List[np.ndarray],
              k: int,
              normalization_transform: Optional[Any] = None) -> List[np.ndarray]:
        """
        Solve polynomial system using subdivision method.
        
        Parameters
        ----------
        equation_coeffs : List[np.ndarray]
            List of Bernstein coefficient arrays (one per equation)
        k : int
            Number of parameters
        normalization_transform : Optional
            Normalization transform for denormalization
            
        Returns
        -------
        List[np.ndarray]
            List of solution points (each is k-dimensional array)
        """
        # Reset state
        self.solutions = []
        self.stats = {key: 0 for key in self.stats.keys()}
        self.normalization_transform = normalization_transform
        self.k = k
        
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
        
        # Initial box range in [0,1]^k
        initial_range = [(0.0, 1.0) for _ in range(k)]
        
        # Solve recursively
        self._solve_recursive(equation_coeffs, initial_range, depth=0)
        
        if self.config.verbose:
            print(f"\n{'='*80}")
            print(f"SOLVER STATISTICS")
            print(f"{'='*80}")
            print(f"Boxes processed: {self.stats['boxes_processed']}")
            print(f"Boxes pruned: {self.stats['boxes_pruned']}")
            print(f"Subdivisions: {self.stats['subdivisions']}")
            print(f"Solutions found: {self.stats['solutions_found']}")
            print(f"{'='*80}\n")
        
        return self.solutions

    def _find_bounding_box(self,
                          coeffs_list: List[np.ndarray]) -> Optional[List[Tuple[float, float]]]:
        """
        Find bounding box using the configured method (PP/LP/Hybrid).

        This is the ONLY place where the method differs between PP, LP, and Hybrid.
        All other logic (subdivision, tightening, recursion) is identical.

        Parameters
        ----------
        coeffs_list : List[np.ndarray]
            Bernstein coefficients for each equation

        Returns
        -------
        Optional[List[Tuple[float, float]]]
            Bounding box in [0,1]^k space, or None if no roots exist
        """
        if self.config.method == BoundingMethod.PP:
            return self._find_bounding_box_pp(coeffs_list)
        elif self.config.method == BoundingMethod.LP:
            return self._find_bounding_box_lp(coeffs_list)
        elif self.config.method == BoundingMethod.HYBRID:
            return self._find_bounding_box_hybrid(coeffs_list)
        else:
            raise ValueError(f"Unknown bounding method: {self.config.method}")

    def _find_bounding_box_pp(self,
                             coeffs_list: List[np.ndarray]) -> Optional[List[Tuple[float, float]]]:
        """
        PP (Projected Polyhedron) method for finding bounding box.

        Uses convex hull of Bernstein control points projected onto each dimension.

        Parameters
        ----------
        coeffs_list : List[np.ndarray]
            Bernstein coefficients for each equation

        Returns
        -------
        Optional[List[Tuple[float, float]]]
            Bounding box in [0,1]^k space, or None if no roots exist
        """
        return find_root_box_pp_nd(coeffs_list, self.k, tolerance=self.config.subdivision_tolerance)

    def _find_bounding_box_lp(self,
                             coeffs_list: List[np.ndarray]) -> Optional[List[Tuple[float, float]]]:
        """
        LP (Linear Programming) method for finding bounding box.

        Uses linear programming to find tighter bounds than PP method.

        TODO: Implement LP-based bounding method.
        For now, falls back to PP method.

        Parameters
        ----------
        coeffs_list : List[np.ndarray]
            Bernstein coefficients for each equation

        Returns
        -------
        Optional[List[Tuple[float, float]]]
            Bounding box in [0,1]^k space, or None if no roots exist
        """
        # TODO: Implement LP method
        # For now, use PP as fallback
        return self._find_bounding_box_pp(coeffs_list)

    def _find_bounding_box_hybrid(self,
                                 coeffs_list: List[np.ndarray]) -> Optional[List[Tuple[float, float]]]:
        """
        Hybrid method for finding bounding box.

        Uses PP method first for quick bounds, then refines with LP method.

        TODO: Implement Hybrid method.
        For now, falls back to PP method.

        Parameters
        ----------
        coeffs_list : List[np.ndarray]
            Bernstein coefficients for each equation

        Returns
        -------
        Optional[List[Tuple[float, float]]]
            Bounding box in [0,1]^k space, or None if no roots exist
        """
        # TODO: Implement Hybrid method
        # Strategy: Use PP first, then refine with LP
        # For now, use PP as fallback
        return self._find_bounding_box_pp(coeffs_list)

    def _solve_recursive(self,
                        coeffs_list: List[np.ndarray],
                        box_range: List[Tuple[float, float]],
                        depth: int):
        """
        Recursively solve using PP method.
        
        This matches the logic in visualize_2d_step_by_step.py:
        1. Apply PP to get tightened bounds
        2. Check if small enough (solution)
        3. Check depth limit
        4. Check CRIT - tighten or subdivide
        
        Parameters
        ----------
        coeffs_list : List[np.ndarray]
            Bernstein coefficients for each equation
        box_range : List[Tuple[float, float]]
            Current box range in [0,1]^k space
        depth : int
            Current depth
        """
        self.stats['boxes_processed'] += 1
        
        # Print progress
        if self.config.verbose and self.stats['boxes_processed'] % 100 == 0:
            print(f"Processed {self.stats['boxes_processed']} boxes, "
                  f"found {self.stats['solutions_found']} solutions, "
                  f"pruned {self.stats['boxes_pruned']} boxes")
        
        # Step 1: Apply bounding method to get tightened bounds
        # This is the ONLY step that differs between PP, LP, and Hybrid methods
        bounding_result = self._find_bounding_box(coeffs_list)

        if bounding_result is None:
            # No roots - PRUNE
            self.stats['boxes_pruned'] += 1
            if self.config.verbose and depth < 5:
                print(f"{'  ' * depth}[Depth {depth}] PRUNED ({self.config.method.value.upper()} method: no roots)")
            return

        if self.config.verbose and depth < 5:
            print(f"{'  ' * depth}[Depth {depth}] {self.config.method.value.upper()} bounds: {bounding_result}")

        # Calculate widths in box coordinates
        bound_widths = [t_max - t_min for t_min, t_max in bounding_result]
        box_widths = [x_max - x_min for (x_min, x_max) in box_range]
        bound_box_widths = [bound_widths[i] * box_widths[i] for i in range(self.k)]

        # Step 2: Check if small enough
        if all(w < self.config.tolerance for w in bound_box_widths):
            # Found solution - compute midpoint
            solution = np.zeros(self.k)
            for i in range(self.k):
                t_mid = (bounding_result[i][0] + bounding_result[i][1]) / 2
                solution[i] = box_range[i][0] + t_mid * box_widths[i]

            self.solutions.append(solution)
            self.stats['solutions_found'] += 1

            if self.config.verbose and depth < 5:
                print(f"{'  ' * depth}[Depth {depth}] SOLUTION at {solution}")
            return

        # Step 3: Check depth limit
        if depth >= self.config.max_depth:
            # Max depth reached - claim solution at midpoint
            solution = np.zeros(self.k)
            for i in range(self.k):
                t_mid = (bounding_result[i][0] + bounding_result[i][1]) / 2
                solution[i] = box_range[i][0] + t_mid * box_widths[i]

            self.solutions.append(solution)
            self.stats['solutions_found'] += 1

            if self.config.verbose and depth < 5:
                print(f"{'  ' * depth}[Depth {depth}] MAX DEPTH at {solution}")
            return

        # Step 4: Check CRIT - should we subdivide or tighten?
        dims_to_subdivide = []
        for i, bound_width in enumerate(bound_widths):
            if bound_width > self.config.crit:
                dims_to_subdivide.append(i)
        
        if not dims_to_subdivide:
            # TIGHTEN all dimensions
            if self.config.verbose and depth < 5:
                print(f"{'  ' * depth}[Depth {depth}] TIGHTEN (all dims reduced ≥ {(1-self.config.crit)*100:.0f}%)")

            # Extract sub-box with tightened bounds
            tight_coeffs_list = []
            for eq_coeffs in coeffs_list:
                tight_coeffs, _ = extract_subbox_with_box(
                    eq_coeffs,
                    Box(k=self.k, ranges=box_range),
                    bounding_result,
                    tolerance=self.config.subdivision_tolerance,
                    verbose=False
                )
                tight_coeffs_list.append(tight_coeffs)

            # Map bounding box to box coordinates
            tight_range = [
                (box_range[i][0] + bounding_result[i][0] * box_widths[i],
                 box_range[i][0] + bounding_result[i][1] * box_widths[i])
                for i in range(self.k)
            ]

            # Recursively solve (same depth - we're tightening, not subdividing)
            self._solve_recursive(tight_coeffs_list, tight_range, depth)
        
        else:
            # SUBDIVIDE along first dimension that needs it
            axis = dims_to_subdivide[0]
            t_mid = (bounding_result[axis][0] + bounding_result[axis][1]) / 2

            if self.config.verbose and depth < 5:
                print(f"{'  ' * depth}[Depth {depth}] SUBDIVIDE dim {axis} at t={t_mid:.6f}")

            self.stats['subdivisions'] += 1

            # Create left and right sub-ranges from bounding box
            left_ranges = list(bounding_result)
            left_ranges[axis] = (bounding_result[axis][0], t_mid)

            right_ranges = list(bounding_result)
            right_ranges[axis] = (t_mid, bounding_result[axis][1])
            
            # Extract coefficients for left sub-box
            left_coeffs_list = []
            for eq_coeffs in coeffs_list:
                left_coeffs, _ = extract_subbox_with_box(
                    eq_coeffs,
                    Box(k=self.k, ranges=box_range),
                    left_ranges,
                    tolerance=self.config.subdivision_tolerance,
                    verbose=False
                )
                left_coeffs_list.append(left_coeffs)
            
            # Extract coefficients for right sub-box
            right_coeffs_list = []
            for eq_coeffs in coeffs_list:
                right_coeffs, _ = extract_subbox_with_box(
                    eq_coeffs,
                    Box(k=self.k, ranges=box_range),
                    right_ranges,
                    tolerance=self.config.subdivision_tolerance,
                    verbose=False
                )
                right_coeffs_list.append(right_coeffs)
            
            # Map to box coordinates
            left_box_range = [
                (box_range[i][0] + left_ranges[i][0] * box_widths[i],
                 box_range[i][0] + left_ranges[i][1] * box_widths[i])
                for i in range(self.k)
            ]
            
            right_box_range = [
                (box_range[i][0] + right_ranges[i][0] * box_widths[i],
                 box_range[i][0] + right_ranges[i][1] * box_widths[i])
                for i in range(self.k)
            ]
            
            # Recursively solve both sub-boxes
            self._solve_recursive(left_coeffs_list, left_box_range, depth + 1)
            self._solve_recursive(right_coeffs_list, right_box_range, depth + 1)


def solve_with_subdivision(
    equation_coeffs: List[np.ndarray],
    k: int,
    method: str = 'pp',
    tolerance: float = 1e-6,
    crit: float = 0.8,
    max_depth: int = 30,
    subdivision_tolerance: float = 1e-10,
    normalization_transform: Optional[Any] = None,
    verbose: bool = False
) -> List[np.ndarray]:
    """
    Solve polynomial system using subdivision method.
    
    Convenience function that creates solver and runs it.
    """
    config = SolverConfig(
        method=BoundingMethod(method),
        tolerance=tolerance,
        crit=crit,
        max_depth=max_depth,
        subdivision_tolerance=subdivision_tolerance,
        verbose=verbose
    )
    
    solver = SubdivisionSolver(config)
    return solver.solve(equation_coeffs, k, normalization_transform)

