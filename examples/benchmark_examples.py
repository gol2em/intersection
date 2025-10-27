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


def run_benchmark(example: BenchmarkExample, verbose: bool = False) -> BenchmarkResult:
    """Run a single benchmark example."""
    
    print("\n" + "=" * 80)
    print(f"BENCHMARK: {example.name}")
    print("=" * 80)
    print(f"Description: {example.description}")
    
    # Setup
    print("\nSetting up problem...")
    setup_start = time.time()
    config = example.setup()
    setup_time = time.time() - setup_start
    
    system = config['system']
    expected_roots = config['expected_roots']
    tolerance = config['tolerance']
    
    print(f"  Dimension: {system.k}D")
    print(f"  Degree: {system.degree}")
    print(f"  Expected solutions: {len(expected_roots)}")
    print(f"  Tolerance: {tolerance:.2e}")
    print(f"  Setup time: {setup_time:.3f}s")
    
    # Solve
    print("\nSolving...")
    solve_start = time.time()
    solutions, solver_stats = solve_polynomial_system(
        system,
        method='pp',
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
    
    print(f"  Solve time: {solve_time:.3f}s")
    print(f"  Solutions found: {len(solutions)}")
    
    # Verify
    print("\nVerifying...")
    max_error, avg_error, success = example.verify(solutions, expected_roots)
    
    print(f"  Max error: {max_error:.2e}")
    print(f"  Avg error: {avg_error:.2e}")
    print(f"  Success: {'✅' if success else '❌'}")
    
    # Create result
    result = BenchmarkResult(
        name=example.name,
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


def print_summary_table(results: List[BenchmarkResult]):
    """Print a summary table of all benchmark results."""

    print("\n" + "=" * 110)
    print("BENCHMARK SUMMARY")
    print("=" * 110)

    # Header
    print(f"{'Example':<25} {'Dim':<5} {'Deg':<5} {'Tol':<10} {'Depth':<7} {'Steps':<7} {'Runtime(s)':<12} {'Max Err':<12} {'Status':<8}")
    print("-" * 110)

    # Rows
    for r in results:
        status = "✅ PASS" if r.success else "❌ FAIL"
        print(f"{r.name:<25} {r.dimension:<5} {r.degree:<5} {r.tolerance:<10.2e} "
              f"{r.max_depth_used:<7} {r.boxes_processed:<7} {r.solve_time:<12.6f} "
              f"{r.max_error:<12.2e} {status:<8}")

    print("=" * 110)

    # Overall statistics
    total_time = sum(r.total_time for r in results)
    passed = sum(1 for r in results if r.success)
    total = len(results)

    print(f"\nTotal benchmarks: {total}")
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"Total time: {total_time:.6f}s")
    print("=" * 110)


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
    
    print(f"\nRunning {len(examples)} benchmark examples...")
    
    # Run benchmarks
    results = []
    for i, example in enumerate(examples, 1):
        print(f"\n[{i}/{len(examples)}]")
        result = run_benchmark(example, verbose=False)
        results.append(result)
    
    # Print summary
    print_summary_table(results)
    
    # Save results to file
    output_file = Path(__file__).parent / "benchmark_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("POLYNOMIAL SYSTEM SOLVER - BENCHMARK RESULTS\n")
        f.write("=" * 120 + "\n\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total benchmarks: {len(results)}\n")
        f.write(f"Passed: {sum(1 for r in results if r.success)}/{len(results)}\n")
        f.write(f"Total time: {sum(r.total_time for r in results):.3f}s\n")
        f.write(f"Newton refinement: DISABLED (for benchmarking)\n")
        f.write("\n")

        # Summary table
        f.write("=" * 110 + "\n")
        f.write("SUMMARY TABLE\n")
        f.write("=" * 110 + "\n")
        f.write(f"{'Example':<25} {'Dim':<5} {'Deg':<5} {'Tol':<10} {'Depth':<7} {'Steps':<7} {'Runtime(s)':<12} {'Max Err':<12} {'Status':<8}\n")
        f.write("-" * 110 + "\n")

        for r in results:
            status = "PASS" if r.success else "FAIL"
            f.write(f"{r.name:<25} {r.dimension:<5} {r.degree:<5} {r.tolerance:<10.2e} "
                    f"{r.max_depth_used:<7} {r.boxes_processed:<7} {r.solve_time:<12.6f} "
                    f"{r.max_error:<12.2e} {status:<8}\n")

        f.write("=" * 110 + "\n\n")

        # Detailed results for each example
        f.write("=" * 120 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("=" * 120 + "\n\n")

        for i, r in enumerate(results, 1):
            f.write(f"[{i}] {r.name}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Description: {examples[i-1].description}\n")
            f.write(f"\n")
            f.write(f"  Problem Parameters:\n")
            f.write(f"    Dimension: {r.dimension}D\n")
            f.write(f"    Degree: {r.degree}\n")
            f.write(f"    Tolerance: {r.tolerance:.2e}\n")
            f.write(f"\n")
            f.write(f"  Results:\n")
            f.write(f"    Expected solutions: {r.expected_solutions}\n")
            f.write(f"    Found solutions: {r.found_solutions}\n")
            f.write(f"    Success: {'PASS' if r.success else 'FAIL'}\n")
            f.write(f"\n")
            f.write(f"  Performance:\n")
            f.write(f"    Setup time: {r.setup_time:.6f}s\n")
            f.write(f"    Solve time (runtime): {r.solve_time:.6f}s\n")
            f.write(f"    Total time: {r.total_time:.6f}s\n")
            f.write(f"\n")
            f.write(f"  Solver Statistics:\n")
            f.write(f"    Steps (boxes processed): {r.boxes_processed}\n")
            f.write(f"    Boxes pruned: {r.boxes_pruned}\n")
            f.write(f"    Subdivisions: {r.subdivisions}\n")
            f.write(f"    Max depth used: {r.max_depth_used}\n")
            f.write(f"\n")
            f.write(f"  Accuracy:\n")
            f.write(f"    Max error: {r.max_error:.6e}\n")
            f.write(f"    Avg error: {r.avg_error:.6e}\n")
            f.write(f"\n")
            if r.notes:
                f.write(f"  Notes: {r.notes}\n")
                f.write(f"\n")

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

