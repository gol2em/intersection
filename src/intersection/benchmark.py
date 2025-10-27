"""
Scientific Benchmarking Framework for Polynomial System Solver

This module provides tools for rigorous performance testing and analysis
of the polynomial system solver with different methods and parameters.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from pathlib import Path

from .polynomial_solver import PolynomialSystem, solve_polynomial_system
from .subdivision_solver import SubdivisionSolver, SolverConfig, BoundingMethod


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    
    # Test identification
    test_name: str
    test_description: str
    timestamp: str
    
    # System properties
    dimension: int  # k (number of parameters)
    num_equations: int
    degree: int
    param_ranges: List[Tuple[float, float]]
    
    # Solver configuration
    method: str  # 'pp', 'lp', or 'hybrid'
    tolerance: float
    crit: float
    max_depth: int
    subdivision_tolerance: float
    refine: bool
    
    # Performance metrics
    cpu_time: float  # seconds
    boxes_processed: int
    boxes_pruned: int
    subdivisions: int
    max_depth_reached: int
    
    # Solution quality
    num_solutions: int
    solutions: List[Dict[str, float]]
    expected_solutions: Optional[List[Dict[str, float]]] = None
    max_residual: Optional[float] = None
    max_error: Optional[float] = None  # If expected solutions are known
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'BenchmarkResult':
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"{'='*80}",
            f"BENCHMARK RESULT: {self.test_name}",
            f"{'='*80}",
            f"Description: {self.test_description}",
            f"Timestamp: {self.timestamp}",
            f"",
            f"SYSTEM PROPERTIES:",
            f"  Dimension (k): {self.dimension}",
            f"  Equations: {self.num_equations}",
            f"  Degree: {self.degree}",
            f"  Parameter ranges: {self.param_ranges}",
            f"",
            f"SOLVER CONFIGURATION:",
            f"  Method: {self.method.upper()}",
            f"  Tolerance: {self.tolerance}",
            f"  CRIT: {self.crit}",
            f"  Max depth: {self.max_depth}",
            f"  Subdivision tolerance: {self.subdivision_tolerance}",
            f"  Refinement: {self.refine}",
            f"",
            f"PERFORMANCE METRICS:",
            f"  CPU time: {self.cpu_time:.6f} seconds",
            f"  Boxes processed: {self.boxes_processed}",
            f"  Boxes pruned: {self.boxes_pruned}",
            f"  Subdivisions: {self.subdivisions}",
            f"  Max depth reached: {self.max_depth_reached}",
            f"  Efficiency: {self.boxes_pruned/max(1, self.boxes_processed)*100:.1f}% pruned",
            f"",
            f"SOLUTION QUALITY:",
            f"  Solutions found: {self.num_solutions}",
        ]
        
        if self.max_residual is not None:
            lines.append(f"  Max residual: {self.max_residual:.2e}")
        
        if self.max_error is not None:
            lines.append(f"  Max error: {self.max_error:.2e}")
        
        if self.expected_solutions:
            lines.append(f"  Expected solutions: {len(self.expected_solutions)}")
        
        lines.append(f"{'='*80}")
        
        return '\n'.join(lines)


class ScientificBenchmark:
    """
    Scientific benchmarking framework for polynomial system solver.
    
    Features:
    - Automated performance testing with multiple configurations
    - Statistical analysis of solver behavior
    - Comparison of different methods (PP, LP, Hybrid)
    - Integration with visualization tools
    - Result persistence and reporting
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize benchmark framework.

        Parameters
        ----------
        output_dir : str
            Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
    
    def run_benchmark(
        self,
        system: PolynomialSystem,
        test_name: str,
        test_description: str = "",
        method: str = 'pp',
        tolerance: float = 1e-6,
        crit: float = 0.8,
        max_depth: int = 30,
        subdivision_tolerance: float = 1e-10,
        refine: bool = True,
        expected_solutions: Optional[List[Dict[str, float]]] = None,
        visualize: bool = False,
        visualizer: Optional[Callable] = None,
        verbose: bool = False
    ) -> BenchmarkResult:
        """
        Run a single benchmark test.
        
        Parameters
        ----------
        system : PolynomialSystem
            The polynomial system to solve
        test_name : str
            Unique name for this test
        test_description : str
            Human-readable description
        method : str
            Solving method: 'pp', 'lp', or 'hybrid'
        tolerance : float
            Size threshold for claiming a root
        crit : float
            Critical ratio for subdivision
        max_depth : int
            Maximum subdivision depth
        subdivision_tolerance : float
            Numerical tolerance for zero detection
        refine : bool
            Whether to refine solutions using Newton iteration
        expected_solutions : List[Dict[str, float]], optional
            Known solutions for error computation
        visualize : bool
            Whether to generate debug visualizations
        visualizer : Callable, optional
            Custom visualization function
        verbose : bool
            Print progress
            
        Returns
        -------
        BenchmarkResult
            Complete benchmark results
        """
        from datetime import datetime
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"RUNNING BENCHMARK: {test_name}")
            print(f"{'='*80}")
            print(f"Description: {test_description}")
            print(f"Method: {method.upper()}")
            print(f"Tolerance: {tolerance}")
            print(f"CRIT: {crit}")
            print(f"{'='*80}\n")
        
        # Create solver with instrumentation
        method_enum = BoundingMethod[method.upper()]
        config = SolverConfig(
            method=method_enum,
            tolerance=tolerance,
            crit=crit,
            max_depth=max_depth,
            subdivision_tolerance=subdivision_tolerance,
            verbose=verbose
        )
        
        solver = SubdivisionSolver(config)
        
        # Run solver with timing
        start_time = time.perf_counter()
        
        solutions_normalized = solver.solve(
            system.equation_coeffs,
            system.k,
            normalization_transform={
                'original_ranges': system.param_ranges,
                'normalized_ranges': [(0.0, 1.0) for _ in range(system.k)]
            }
        )
        
        cpu_time = time.perf_counter() - start_time
        
        # Convert solutions to original domain
        solutions = []
        for sol_norm in solutions_normalized:
            sol_dict = {}
            for i, param_name in enumerate(system.param_names[:system.k]):
                t_norm = sol_norm[i]
                t_min, t_max = system.param_ranges[i]
                t_orig = t_min + t_norm * (t_max - t_min)
                sol_dict[param_name] = float(t_orig)
            solutions.append(sol_dict)
        
        # Optionally refine solutions
        if refine and solutions:
            from .polynomial_solver import _refine_solution_newton_standalone
            refined_solutions = []
            for sol_norm in solutions_normalized:
                try:
                    refined_norm = _refine_solution_newton_standalone(
                        system.equation_coeffs,
                        sol_norm,
                        system.k,
                        max_iter=10,
                        tol=1e-10
                    )
                    # Convert refined solution to original domain
                    sol_dict = {}
                    for i, param_name in enumerate(system.param_names[:system.k]):
                        t_norm = refined_norm[i]
                        t_min, t_max = system.param_ranges[i]
                        t_orig = t_min + t_norm * (t_max - t_min)
                        sol_dict[param_name] = float(t_orig)
                    refined_solutions.append(sol_dict)
                except:
                    # If refinement fails, use unrefined solution
                    pass

            # Only use refined solutions if we got the same number
            if len(refined_solutions) == len(solutions):
                solutions = refined_solutions
        
        # Compute residuals
        max_residual = self._compute_max_residual(system, solutions)
        
        # Compute errors if expected solutions are provided
        max_error = None
        if expected_solutions:
            max_error = self._compute_max_error(solutions, expected_solutions)
        
        # Find max depth reached
        max_depth_reached = self._estimate_max_depth(solver.stats['boxes_processed'])
        
        # Create result
        result = BenchmarkResult(
            test_name=test_name,
            test_description=test_description,
            timestamp=datetime.now().isoformat(),
            dimension=system.k,
            num_equations=len(system.equation_coeffs),
            degree=system.degree,
            param_ranges=system.param_ranges,
            method=method,
            tolerance=tolerance,
            crit=crit,
            max_depth=max_depth,
            subdivision_tolerance=subdivision_tolerance,
            refine=refine,
            cpu_time=cpu_time,
            boxes_processed=solver.stats['boxes_processed'],
            boxes_pruned=solver.stats['boxes_pruned'],
            subdivisions=solver.stats['subdivisions'],
            max_depth_reached=max_depth_reached,
            num_solutions=len(solutions),
            solutions=solutions,
            expected_solutions=expected_solutions,
            max_residual=max_residual,
            max_error=max_error
        )
        
        # Store result
        self.results.append(result)
        
        # Save to file
        result_file = self.output_dir / f"{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result.to_json(str(result_file))
        
        if verbose:
            print(result.summary())

        return result

    def _compute_max_residual(
        self,
        system: PolynomialSystem,
        solutions: List[Dict[str, float]]
    ) -> Optional[float]:
        """Compute maximum residual across all equations and solutions."""
        if not solutions:
            return None

        max_res = 0.0
        for sol in solutions:
            # Evaluate each equation at solution
            for eq_coeffs in system.equation_coeffs:
                # Convert solution to parameter array
                params = np.array([sol[name] for name in system.param_names[:system.k]])

                # Evaluate polynomial (simple power form evaluation)
                # Note: This is approximate - for exact evaluation we'd need to
                # convert back from Bernstein or store original power coefficients
                residual = abs(self._evaluate_bernstein(eq_coeffs, params, system.param_ranges))
                max_res = max(max_res, residual)

        return max_res

    def _evaluate_bernstein(
        self,
        coeffs: np.ndarray,
        params: np.ndarray,
        param_ranges: List[Tuple[float, float]]
    ) -> float:
        """Evaluate Bernstein polynomial at given parameters."""
        from .bernstein import evaluate_bernstein_1d, evaluate_bernstein_2d, evaluate_bernstein_kd

        # Normalize parameters to [0, 1]
        params_norm = np.zeros_like(params)
        for i in range(len(params)):
            t_min, t_max = param_ranges[i]
            params_norm[i] = (params[i] - t_min) / (t_max - t_min)

        # Call appropriate evaluation function based on dimension
        k = len(params_norm)
        if k == 1:
            return evaluate_bernstein_1d(coeffs, params_norm[0])
        elif k == 2:
            return evaluate_bernstein_2d(coeffs, params_norm[0], params_norm[1])
        else:
            return evaluate_bernstein_kd(coeffs, *params_norm)

    def _compute_max_error(
        self,
        solutions: List[Dict[str, float]],
        expected_solutions: List[Dict[str, float]]
    ) -> float:
        """Compute maximum error between found and expected solutions."""
        if not solutions or not expected_solutions:
            return float('inf')

        max_error = 0.0

        # For each expected solution, find closest found solution
        for expected in expected_solutions:
            min_dist = float('inf')
            for found in solutions:
                # Compute Euclidean distance
                dist = 0.0
                for key in expected.keys():
                    if key in found:
                        dist += (expected[key] - found[key]) ** 2
                dist = np.sqrt(dist)
                min_dist = min(min_dist, dist)
            max_error = max(max_error, min_dist)

        return max_error

    def _estimate_max_depth(self, boxes_processed: int) -> int:
        """Estimate maximum depth reached based on boxes processed."""
        if boxes_processed <= 1:
            return 0
        # Rough estimate: depth ≈ log2(boxes_processed)
        return int(np.log2(boxes_processed))

    def compare_methods(
        self,
        system: PolynomialSystem,
        test_name: str,
        methods: List[str] = ['pp', 'lp', 'hybrid'],
        tolerance: float = 1e-6,
        crit: float = 0.8,
        max_depth: int = 30,
        subdivision_tolerance: float = 1e-10,
        expected_solutions: Optional[List[Dict[str, float]]] = None,
        verbose: bool = True
    ) -> List[BenchmarkResult]:
        """
        Compare different solving methods on the same system.

        Parameters
        ----------
        system : PolynomialSystem
            The polynomial system to solve
        test_name : str
            Base name for tests
        methods : List[str]
            Methods to compare
        tolerance : float
            Size threshold
        crit : float
            Critical ratio
        max_depth : int
            Maximum depth
        subdivision_tolerance : float
            Numerical tolerance
        expected_solutions : List[Dict[str, float]], optional
            Known solutions
        verbose : bool
            Print progress

        Returns
        -------
        List[BenchmarkResult]
            Results for each method
        """
        results = []

        for method in methods:
            result = self.run_benchmark(
                system=system,
                test_name=f"{test_name}_{method}",
                test_description=f"Method comparison: {method.upper()}",
                method=method,
                tolerance=tolerance,
                crit=crit,
                max_depth=max_depth,
                subdivision_tolerance=subdivision_tolerance,
                expected_solutions=expected_solutions,
                verbose=verbose
            )
            results.append(result)

        if verbose:
            self._print_comparison(results)

        return results

    def _print_comparison(self, results: List[BenchmarkResult]):
        """Print comparison table of results."""
        print(f"\n{'='*80}")
        print("METHOD COMPARISON")
        print(f"{'='*80}")
        print(f"{'Method':<10} {'Time (s)':<12} {'Boxes':<10} {'Pruned':<10} {'Solutions':<10} {'Max Error':<12}")
        print(f"{'-'*80}")

        for result in results:
            error_str = f"{result.max_error:.2e}" if result.max_error is not None else "N/A"
            print(f"{result.method.upper():<10} "
                  f"{result.cpu_time:<12.6f} "
                  f"{result.boxes_processed:<10} "
                  f"{result.boxes_pruned:<10} "
                  f"{result.num_solutions:<10} "
                  f"{error_str:<12}")

        print(f"{'='*80}\n")

    def parameter_sweep(
        self,
        system: PolynomialSystem,
        test_name: str,
        parameter_name: str,
        parameter_values: List[Any],
        method: str = 'pp',
        base_config: Optional[Dict[str, Any]] = None,
        expected_solutions: Optional[List[Dict[str, float]]] = None,
        verbose: bool = True
    ) -> List[BenchmarkResult]:
        """
        Sweep a parameter and measure performance.

        Parameters
        ----------
        system : PolynomialSystem
            The polynomial system to solve
        test_name : str
            Base name for tests
        parameter_name : str
            Name of parameter to sweep ('tolerance', 'crit', 'max_depth', etc.)
        parameter_values : List[Any]
            Values to test
        method : str
            Solving method
        base_config : Dict[str, Any], optional
            Base configuration (other parameters)
        expected_solutions : List[Dict[str, float]], optional
            Known solutions
        verbose : bool
            Print progress

        Returns
        -------
        List[BenchmarkResult]
            Results for each parameter value
        """
        if base_config is None:
            base_config = {
                'tolerance': 1e-6,
                'crit': 0.8,
                'max_depth': 30,
                'subdivision_tolerance': 1e-10,
                'refine': True
            }

        results = []

        for value in parameter_values:
            config = base_config.copy()
            config[parameter_name] = value

            result = self.run_benchmark(
                system=system,
                test_name=f"{test_name}_{parameter_name}_{value}",
                test_description=f"Parameter sweep: {parameter_name}={value}",
                method=method,
                expected_solutions=expected_solutions,
                verbose=verbose,
                **config
            )
            results.append(result)

        if verbose:
            self._print_parameter_sweep(results, parameter_name)

        return results

    def _print_parameter_sweep(self, results: List[BenchmarkResult], param_name: str):
        """Print parameter sweep results."""
        print(f"\n{'='*80}")
        print(f"PARAMETER SWEEP: {param_name}")
        print(f"{'='*80}")
        print(f"{param_name:<15} {'Time (s)':<12} {'Boxes':<10} {'Solutions':<10} {'Max Error':<12}")
        print(f"{'-'*80}")

        for result in results:
            param_value = getattr(result, param_name)
            error_str = f"{result.max_error:.2e}" if result.max_error is not None else "N/A"
            print(f"{param_value:<15} "
                  f"{result.cpu_time:<12.6f} "
                  f"{result.boxes_processed:<10} "
                  f"{result.num_solutions:<10} "
                  f"{error_str:<12}")

        print(f"{'='*80}\n")

    def generate_report(self, output_file: str = "benchmark_report.txt"):
        """Generate comprehensive report of all benchmarks."""
        report_path = self.output_dir / output_file

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BENCHMARK REPORT\n")
            f.write("="*80 + "\n\n")

            for result in self.results:
                f.write(result.summary() + "\n\n")

        print(f"Report saved to: {report_path}")
        return str(report_path)

