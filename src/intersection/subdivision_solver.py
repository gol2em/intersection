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
            'max_depth_used': 0,
        }
        self.normalization_transform = None
    
    def solve(self,
              equation_coeffs: List[np.ndarray],
              k: int,
              normalization_transform: Optional[Any] = None) -> tuple:
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
        tuple
            (solutions, stats) where:
            - solutions: List[np.ndarray] - List of solution points (each is k-dimensional array)
            - stats: dict - Statistics dictionary with keys:
                - 'boxes_processed': int
                - 'boxes_pruned': int
                - 'subdivisions': int
                - 'solutions_found': int
                - 'max_depth_used': int
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
            print(f"Max depth used: {self.stats['max_depth_used']}")
            print(f"{'='*80}\n")

        return self.solutions, self.stats

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

        For a system of equations, a root must lie in the INTERSECTION of
        the convex hulls of control points for each equation.

        For each dimension j:
        1. Minimize/maximize x_j
        2. Subject to: for EACH equation i, (x_0, ..., x_{k-1}, 0) is in
           the convex hull of that equation's control points
        3. This requires separate λ^i variables for each equation

        Parameters
        ----------
        coeffs_list : List[np.ndarray]
            Bernstein coefficients for each equation

        Returns
        -------
        Optional[List[Tuple[float, float]]]
            Bounding box in [0,1]^k space, or None if no roots exist
        """
        try:
            from scipy.optimize import linprog
        except ImportError:
            # scipy not available - cannot use LP method
            if self.config.verbose:
                print("Warning: scipy not available, LP method requires scipy")
            return None

        if not coeffs_list:
            return None

        # Quick check: if any equation has all coefficients with same sign, no root
        for coeffs in coeffs_list:
            c_min = np.min(coeffs)
            c_max = np.max(coeffs)
            if c_min > self.config.subdivision_tolerance:
                return None
            if c_max < -self.config.subdivision_tolerance:
                return None

        # Initialize bounding box as full [0,1]^k
        bounding_box = [(0.0, 1.0) for _ in range(self.k)]

        # For each dimension, find bounds using LP over ALL equations simultaneously
        for dim in range(self.k):
            bounds = self._lp_bounds_for_dimension_all_equations(coeffs_list, dim)

            if bounds is None:
                # No solution possible
                return None

            bounding_box[dim] = bounds

        return bounding_box

    def _lp_bounds_for_dimension_all_equations(self,
                                               coeffs_list: List[np.ndarray],
                                               dim: int) -> Optional[Tuple[float, float]]:
        """
        Compute LP-based bounds for a single dimension across ALL equations.

        The correct LP formulation for a system of equations:
        - For each equation i, we have control points (t_0, ..., t_{k-1}, f_i)
        - A root must lie in the INTERSECTION of all convex hulls
        - For each equation i, we need separate λ^i variables

        To find min/max of x_j:
        - Minimize/Maximize x_j
        - Subject to: for EACH equation i:
          - (x_0, ..., x_{k-1}, 0) = Σ_m λ^i_m · control_point^i_m
          - Σ_m λ^i_m = 1
          - λ^i_m ≥ 0

        Parameters
        ----------
        coeffs_list : List[np.ndarray]
            Bernstein coefficients for all equations
        dim : int
            Dimension index (0 to k-1)

        Returns
        -------
        Optional[Tuple[float, float]]
            (t_min, t_max) bounds for this dimension, or None if no roots possible
        """
        from scipy.optimize import linprog

        # Build control points for each equation
        all_control_points = []
        num_points_per_eq = []

        for coeffs in coeffs_list:
            shape = coeffs.shape
            control_points = []

            for multi_idx in np.ndindex(shape):
                # Compute parameter values for this control point
                t_values = []
                for j in range(self.k):
                    i_j = multi_idx[j]
                    n_j = shape[j]
                    if n_j == 1:
                        t_j = 0.5
                    else:
                        t_j = i_j / (n_j - 1)
                    t_values.append(t_j)

                # Function value
                f_value = coeffs[multi_idx]

                # Control point is (t_0, ..., t_{k-1}, f)
                control_point = t_values + [f_value]
                control_points.append(control_point)

            all_control_points.append(np.array(control_points))
            num_points_per_eq.append(len(control_points))

        # Variables: [λ^0_0, ..., λ^0_{M0}, λ^1_0, ..., λ^1_{M1}, ..., x_0, ..., x_{k-1}]
        # Total: sum(num_points_per_eq) + k
        total_lambda_vars = sum(num_points_per_eq)
        total_vars = total_lambda_vars + self.k

        # Objective: minimize x_dim
        c = np.zeros(total_vars)
        c[total_lambda_vars + dim] = 1.0

        # Build constraints
        A_eq = []
        b_eq = []

        # For each equation i:
        lambda_offset = 0
        for eq_idx, control_points in enumerate(all_control_points):
            num_points = num_points_per_eq[eq_idx]

            # Constraint: Σ λ^i_m = 1
            row = np.zeros(total_vars)
            row[lambda_offset:lambda_offset + num_points] = 1.0
            A_eq.append(row)
            b_eq.append(1.0)

            # Constraint: For each dimension j, x_j = Σ λ^i_m · t_j^m
            for j in range(self.k):
                row = np.zeros(total_vars)
                # Coefficients for λ^i variables
                row[lambda_offset:lambda_offset + num_points] = control_points[:, j]
                # Coefficient for x_j variable
                row[total_lambda_vars + j] = -1.0
                A_eq.append(row)
                b_eq.append(0.0)

            # Constraint: 0 = Σ λ^i_m · f^i_m
            row = np.zeros(total_vars)
            row[lambda_offset:lambda_offset + num_points] = control_points[:, self.k]
            A_eq.append(row)
            b_eq.append(0.0)

            lambda_offset += num_points

        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)

        # Bounds: λ^i_m ≥ 0, x_j ∈ [0, 1]
        bounds = [(0, None)] * total_lambda_vars + [(0, 1)] * self.k

        # Solve LP for minimum
        result_min = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if not result_min.success:
            # No feasible solution
            return None

        t_min = result_min.x[total_lambda_vars + dim]

        # Solve LP for maximum (negate objective)
        c_max = -c
        result_max = linprog(c_max, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if not result_max.success:
            # No feasible solution
            return None

        t_max = result_max.x[total_lambda_vars + dim]

        # Ensure t_min <= t_max
        if t_min > t_max:
            t_min, t_max = t_max, t_min

        return (t_min, t_max)

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

        # Track max depth used
        if depth > self.stats['max_depth_used']:
            self.stats['max_depth_used'] = depth

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
) -> tuple:
    """
    Solve polynomial system using subdivision method.

    Convenience function that creates solver and runs it.

    Returns
    -------
    tuple
        (solutions, stats) where:
        - solutions: List[np.ndarray] - List of solution points
        - stats: dict - Statistics dictionary
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

