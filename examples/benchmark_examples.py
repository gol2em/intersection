"""
Benchmark Suite for Polynomial System Solver

This script benchmarks the solver on various test cases.
Easy to add new examples for benchmarking.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from intersection.polynomial_solver import (
    create_polynomial_system,
    solve_polynomial_system
)
from intersection.bernstein import (
    polynomial_nd_to_bernstein,
    transform_polynomial_domain_1d
)
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Callable


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    method: str  # 'pp' or 'lp'
    dimension: int
    degree: int
    expected_solutions: int
    found_solutions: int
    boxes_processed: int
    subdivisions: int
    boxes_pruned: int
    setup_time: float
    solve_time: float
    total_time: float
    tolerance: float
    max_depth_used: int  # Actual maximum depth reached during solving
    max_error: float
    avg_error: float
    success: bool
    notes: str = ""


class BenchmarkExample:
    """Base class for benchmark examples."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def setup(self) -> Dict[str, Any]:
        """
        Setup the benchmark problem.
        
        Returns
        -------
        dict with keys:
            - system: PolynomialSystem
            - expected_roots: list of expected solutions
            - tolerance: float
            - other solver parameters
        """
        raise NotImplementedError
    
    def verify(self, solutions: List[Dict], expected_roots: List) -> tuple:
        """
        Verify solutions against expected roots.
        
        Returns
        -------
        (max_error, avg_error, success)
        """
        raise NotImplementedError


class Example1DWilkinson(BenchmarkExample):
    """Wilkinson polynomial: (x-1)(x-2)...(x-20) = 0"""

    def __init__(self):
        super().__init__(
            name="1D: Wilkinson",
            description="Wilkinson polynomial (x-1)(x-2)...(x-20) on domain [0, 25]"
        )

    def setup(self) -> Dict[str, Any]:
        # Define roots
        roots = list(range(1, 21))

        # Expand polynomial
        poly = np.array([1.0])
        for root in roots:
            factor = np.array([-root, 1.0])
            poly = np.convolve(poly, factor)

        # Transform to [0, 1] domain
        power_coeffs_normalized = transform_polynomial_domain_1d(
            poly,
            from_range=(0.0, 25.0),
            to_range=(0.0, 1.0),
            verbose=False
        )

        # Convert to Bernstein basis
        bern_coeffs = polynomial_nd_to_bernstein(power_coeffs_normalized, k=1)

        # Create system
        system = create_polynomial_system(
            equation_coeffs=[bern_coeffs],
            param_ranges=[(0.0, 25.0)],
            param_names=['x']
        )

        return {
            'system': system,
            'expected_roots': roots,
            'tolerance': 1e-8,
            'max_depth': None,  # Auto-calculate based on degree and dimension
            'crit': 0.8,
            'refine': False  # No Newton refinement for benchmarking
        }
    
    def verify(self, solutions: List[Dict], expected_roots: List) -> tuple:
        if len(solutions) != len(expected_roots):
            return (float('inf'), float('inf'), False)
        
        # Sort solutions
        solutions_sorted = sorted(solutions, key=lambda s: s['x'])
        
        # Compute errors
        errors = []
        for sol, expected in zip(solutions_sorted, expected_roots):
            error = abs(sol['x'] - expected)
            errors.append(error)
        
        max_error = max(errors)
        avg_error = np.mean(errors)
        success = len(solutions) == len(expected_roots)
        
        return (max_error, avg_error, success)


class Example2DCircleEllipse(BenchmarkExample):
    """2D Circle-Ellipse intersection"""
    
    def __init__(self):
        super().__init__(
            name="2D: Circle-Ellipse",
            description="Circle x^2+y^2=1 and Ellipse x^2/4+4y^2=1 on [0,1]×[0,1]"
        )
    
    def setup(self) -> Dict[str, Any]:
        # Circle: x^2 + y^2 - 1 = 0
        circle_power = np.array([
            [-1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])

        # Ellipse: x^2/4 + 4*y^2 - 1 = 0
        ellipse_power = np.array([
            [-1.0, 0.0, 4.0],
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0]
        ])

        # Convert to Bernstein basis
        circle_bern = polynomial_nd_to_bernstein(circle_power, k=2)
        ellipse_bern = polynomial_nd_to_bernstein(ellipse_power, k=2)

        # Create system
        system = create_polynomial_system(
            equation_coeffs=[circle_bern, ellipse_bern],
            param_ranges=[(0.0, 1.0), (0.0, 1.0)],
            param_names=['x', 'y']
        )

        # Expected solution: x = 2/sqrt(5), y = 1/sqrt(5)
        expected_x = 2.0 / np.sqrt(5.0)  # ≈ 0.894427
        expected_y = 1.0 / np.sqrt(5.0)  # ≈ 0.447214

        return {
            'system': system,
            'expected_roots': [(expected_x, expected_y)],
            'tolerance': 1e-6,
            'max_depth': None,  # Auto-calculate based on degree and dimension
            'crit': 0.8,
            'refine': False  # No Newton refinement for benchmarking
        }
    
    def verify(self, solutions: List[Dict], expected_roots: List) -> tuple:
        if len(solutions) != len(expected_roots):
            return (float('inf'), float('inf'), False)
        
        # Compute error
        sol = solutions[0]
        expected_x, expected_y = expected_roots[0]
        
        error_x = abs(sol['x'] - expected_x)
        error_y = abs(sol['y'] - expected_y)
        max_error = max(error_x, error_y)
        avg_error = (error_x + error_y) / 2
        
        success = len(solutions) == 1
        
        return (max_error, avg_error, success)


def run_benchmark(example: BenchmarkExample, method: str = 'pp', verbose: bool = False) -> BenchmarkResult:
    """Run a single benchmark example with specified method."""

    # Setup
    setup_start = time.time()
    config = example.setup()
    setup_time = time.time() - setup_start

    system = config['system']
    expected_roots = config['expected_roots']
    tolerance = config['tolerance']

    # Solve
    solve_start = time.time()
    solutions, solver_stats = solve_polynomial_system(
        system,
        method=method,
        tolerance=config['tolerance'],
        max_depth=config.get('max_depth', 30),
        crit=config.get('crit', 0.8),
        refine=config.get('refine', True),
        verbose=verbose,
        return_stats=True
    )
    solve_time = time.time() - solve_start

    # Get solver statistics
    boxes_processed = solver_stats.get('boxes_processed', 0)
    subdivisions = solver_stats.get('subdivisions', 0)
    boxes_pruned = solver_stats.get('boxes_pruned', 0)
    max_depth_used = solver_stats.get('max_depth_used', 0)

    # Verify
    max_error, avg_error, success = example.verify(solutions, expected_roots)

    # Create result
    result = BenchmarkResult(
        name=example.name,
        method=method.upper(),
        dimension=system.k,
        degree=system.degree,
        expected_solutions=len(expected_roots),
        found_solutions=len(solutions),
        boxes_processed=boxes_processed,
        subdivisions=subdivisions,
        boxes_pruned=boxes_pruned,
        setup_time=setup_time,
        solve_time=solve_time,
        total_time=setup_time + solve_time,
        tolerance=tolerance,
        max_depth_used=max_depth_used,
        max_error=max_error,
        avg_error=avg_error,
        success=success
    )

    return result


def print_summary_table(results: List[BenchmarkResult], method: str):
    """Print a summary table for a specific method."""

    print("\n" + "=" * 120)
    print(f"BENCHMARK SUMMARY - {method.upper()} METHOD")
    print("=" * 120)

    # Header
    print(f"{'Example':<25} {'Dim':<5} {'Deg':<5} {'Tol':<10} {'Depth':<7} {'Boxes':<8} {'Runtime(s)':<12} {'Max Err':<12} {'Status':<8}")
    print("-" * 120)

    # Filter results for this method
    method_results = [r for r in results if r.method == method.upper()]

    # Rows
    for r in method_results:
        status = "✅ PASS" if r.success else "❌ FAIL"
        print(f"{r.name:<25} {r.dimension:<5} {r.degree:<5} {r.tolerance:<10.2e} "
              f"{r.max_depth_used:<7} {r.boxes_processed:<8} {r.solve_time:<12.6f} "
              f"{r.max_error:<12.2e} {status:<8}")

    print("=" * 120)

    # Overall statistics
    total_time = sum(r.solve_time for r in method_results)
    passed = sum(1 for r in method_results if r.success)
    total = len(method_results)

    print(f"\nTotal benchmarks: {total}")
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"Total solve time: {total_time:.6f}s")
    print("=" * 120)


def main():
    """Run all benchmarks."""

    print("=" * 80)
    print("POLYNOMIAL SYSTEM SOLVER - BENCHMARK SUITE")
    print("=" * 80)

    # Define all benchmark examples
    examples = [
        Example2DCircleEllipse(),  # 2D Circle-Ellipse intersection
        Example1DWilkinson(),      # Wilkinson polynomial
    ]

    # Define methods to test
    methods = ['pp', 'lp']

    print(f"\nRunning {len(examples)} examples with {len(methods)} methods...")
    print(f"Examples: {', '.join(e.name for e in examples)}")
    print(f"Methods: {', '.join(m.upper() for m in methods)}")

    # Run benchmarks
    results = []
    total_runs = len(examples) * len(methods)
    run_count = 0

    for example in examples:
        print("\n" + "=" * 80)
        print(f"BENCHMARK: {example.name}")
        print("=" * 80)
        print(f"Description: {example.description}")

        # Setup once
        config = example.setup()
        system = config['system']
        print(f"\n  Dimension: {system.k}D")
        print(f"  Degree: {system.degree}")
        print(f"  Expected solutions: {len(config['expected_roots'])}")
        print(f"  Tolerance: {config['tolerance']:.2e}")

        for method in methods:
            run_count += 1
            print(f"\n  [{run_count}/{total_runs}] Testing {method.upper()} method...")
            result = run_benchmark(example, method=method, verbose=False)
            results.append(result)

            status = "✅" if result.success else "❌"
            print(f"      Solutions: {result.found_solutions}, "
                  f"Boxes: {result.boxes_processed}, "
                  f"Depth: {result.max_depth_used}, "
                  f"Time: {result.solve_time:.4f}s {status}")

    # Print summary tables for each method
    for method in methods:
        print_summary_table(results, method)
    
    # Save results to file
    output_file = Path(__file__).parent / "benchmark_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("POLYNOMIAL SYSTEM SOLVER - BENCHMARK RESULTS\n")
        f.write("=" * 120 + "\n\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total benchmarks: {len(results)}\n")
        f.write(f"Examples: {len(examples)}\n")
        f.write(f"Methods: {', '.join(m.upper() for m in methods)}\n")
        f.write(f"Passed: {sum(1 for r in results if r.success)}/{len(results)}\n")
        f.write(f"Total time: {sum(r.total_time for r in results):.3f}s\n")
        f.write(f"Newton refinement: DISABLED (for benchmarking)\n")
        f.write("\n")

        # Summary tables for each method
        for method in methods:
            method_results = [r for r in results if r.method == method.upper()]

            f.write("=" * 120 + "\n")
            f.write(f"SUMMARY TABLE - {method.upper()} METHOD\n")
            f.write("=" * 120 + "\n")
            f.write(f"{'Example':<25} {'Dim':<5} {'Deg':<5} {'Tol':<10} {'Depth':<7} {'Boxes':<8} {'Runtime(s)':<12} {'Max Err':<12} {'Status':<8}\n")
            f.write("-" * 120 + "\n")

            for r in method_results:
                status = "PASS" if r.success else "FAIL"
                f.write(f"{r.name:<25} {r.dimension:<5} {r.degree:<5} {r.tolerance:<10.2e} "
                        f"{r.max_depth_used:<7} {r.boxes_processed:<8} {r.solve_time:<12.6f} "
                        f"{r.max_error:<12.2e} {status:<8}\n")

            f.write("=" * 120 + "\n")
            f.write(f"Total solve time ({method.upper()}): {sum(r.solve_time for r in method_results):.6f}s\n")
            f.write("\n")

        # Detailed results for each example
        f.write("\n")
        f.write("=" * 120 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("=" * 120 + "\n\n")

        for example in examples:
            # Get results for this example
            example_results = [r for r in results if r.name == example.name]

            f.write(f"{example.name}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Description: {example.description}\n")
            f.write(f"\n")

            # Problem parameters (same for all methods)
            r0 = example_results[0]
            f.write(f"  Problem Parameters:\n")
            f.write(f"    Dimension: {r0.dimension}D\n")
            f.write(f"    Degree: {r0.degree}\n")
            f.write(f"    Tolerance: {r0.tolerance:.2e}\n")
            f.write(f"    Expected solutions: {r0.expected_solutions}\n")
            f.write(f"\n")

            # Results for each method
            for r in example_results:
                f.write(f"  Method: {r.method}\n")
                f.write(f"    Found solutions: {r.found_solutions}\n")
                f.write(f"    Success: {'PASS' if r.success else 'FAIL'}\n")
                f.write(f"    Solve time: {r.solve_time:.6f}s\n")
                f.write(f"    Boxes processed: {r.boxes_processed}\n")
                f.write(f"    Boxes pruned: {r.boxes_pruned}\n")
                f.write(f"    Subdivisions: {r.subdivisions}\n")
                f.write(f"    Max depth used: {r.max_depth_used}\n")
                f.write(f"    Max error: {r.max_error:.6e}\n")
                f.write(f"    Avg error: {r.avg_error:.6e}\n")
                if r.notes:
                    f.write(f"    Notes: {r.notes}\n")
                f.write(f"\n")

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

