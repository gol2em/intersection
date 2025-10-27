"""
Scientific Benchmark Example for Polynomial System Solver

This example demonstrates how to use the benchmarking framework to:
1. Test solver performance on different systems
2. Compare different methods (PP, LP, Hybrid)
3. Perform parameter sweeps
4. Generate comprehensive reports
"""

import sys
sys.path.insert(0, 'D:/Python/Intersection')

import numpy as np
from intersection.polynomial_solver import create_polynomial_system
from intersection.bernstein import polynomial_nd_to_bernstein
from intersection.benchmark import ScientificBenchmark


def benchmark_1d_quadratic():
    """Benchmark 1D quadratic equation: (t - 0.3)(t - 0.7) = 0"""
    print("\n" + "="*80)
    print("BENCHMARK 1: 1D Quadratic Equation")
    print("="*80)
    
    # Create system: (t - 0.3)(t - 0.7) = t^2 - t + 0.21
    power_coeffs = np.array([0.21, -1.0, 1.0])
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    system = create_polynomial_system(
        equation_coeffs=[bern_coeffs],
        param_ranges=[(0.0, 1.0)],
        param_names=['t']
    )
    
    expected_solutions = [{'t': 0.3}, {'t': 0.7}]
    
    # Create benchmark
    benchmark = ScientificBenchmark(output_dir="benchmark_results/1d_quadratic")
    
    # Run single test
    result = benchmark.run_benchmark(
        system=system,
        test_name="1d_quadratic_pp",
        test_description="1D quadratic with 2 roots using PP method",
        method='pp',
        tolerance=1e-6,
        crit=0.8,
        max_depth=30,
        subdivision_tolerance=1e-10,
        refine=True,
        expected_solutions=expected_solutions,
        verbose=True
    )
    
    print(f"\nResult: Found {result.num_solutions} solutions in {result.cpu_time:.6f} seconds")
    print(f"Boxes processed: {result.boxes_processed}")
    print(f"Max error: {result.max_error:.2e}" if result.max_error else "Max error: N/A")
    
    return benchmark


def benchmark_2d_circle_ellipse():
    """Benchmark 2D circle-ellipse intersection"""
    print("\n" + "="*80)
    print("BENCHMARK 2: 2D Circle-Ellipse Intersection")
    print("="*80)
    
    # Equation 1: x^2 + y^2 - 1 = 0 (circle)
    eq1_power = np.array([
        [-1.0, 0.0, 1.0],  # -1 + y^2
        [0.0, 0.0, 0.0],   # 0
        [1.0, 0.0, 0.0]    # x^2
    ])
    
    # Equation 2: x^2/4 + 4*y^2 - 1 = 0 (ellipse)
    eq2_power = np.array([
        [-1.0, 0.0, 4.0],  # -1 + 4*y^2
        [0.0, 0.0, 0.0],   # 0
        [0.25, 0.0, 0.0]   # x^2/4
    ])
    
    eq1_bern = polynomial_nd_to_bernstein(eq1_power, k=2)
    eq2_bern = polynomial_nd_to_bernstein(eq2_power, k=2)
    
    system = create_polynomial_system(
        equation_coeffs=[eq1_bern, eq2_bern],
        param_ranges=[(0.0, 1.0), (0.0, 1.0)],
        param_names=['x', 'y']
    )
    
    # Expected solution: x ≈ 0.894427, y ≈ 0.447214
    expected_solutions = [{'x': 0.894427, 'y': 0.447214}]
    
    # Create benchmark
    benchmark = ScientificBenchmark(output_dir="benchmark_results/2d_circle_ellipse")
    
    # Run single test
    result = benchmark.run_benchmark(
        system=system,
        test_name="2d_circle_ellipse_pp",
        test_description="2D circle-ellipse intersection using PP method",
        method='pp',
        tolerance=1e-6,
        crit=0.8,
        max_depth=50,
        subdivision_tolerance=1e-10,
        refine=False,
        expected_solutions=expected_solutions,
        verbose=True
    )
    
    print(f"\nResult: Found {result.num_solutions} solutions in {result.cpu_time:.6f} seconds")
    print(f"Boxes processed: {result.boxes_processed}")
    print(f"Max error: {result.max_error:.2e}" if result.max_error else "Max error: N/A")
    
    return benchmark


def benchmark_method_comparison():
    """Compare PP, LP, and Hybrid methods on the same system"""
    print("\n" + "="*80)
    print("BENCHMARK 3: Method Comparison")
    print("="*80)
    
    # Simple 1D cubic: (t - 0.2)(t - 0.5)(t - 0.8) = 0
    # Expanded: t^3 - 1.5*t^2 + 0.74*t - 0.08
    power_coeffs = np.array([-0.08, 0.74, -1.5, 1.0])
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    system = create_polynomial_system(
        equation_coeffs=[bern_coeffs],
        param_ranges=[(0.0, 1.0)],
        param_names=['t']
    )
    
    expected_solutions = [{'t': 0.2}, {'t': 0.5}, {'t': 0.8}]
    
    # Create benchmark
    benchmark = ScientificBenchmark(output_dir="benchmark_results/method_comparison")
    
    # Compare methods
    results = benchmark.compare_methods(
        system=system,
        test_name="1d_cubic_comparison",
        methods=['pp'],  # Only PP is fully implemented
        tolerance=1e-6,
        crit=0.8,
        max_depth=30,
        subdivision_tolerance=1e-10,
        expected_solutions=expected_solutions,
        verbose=True
    )
    
    return benchmark


def benchmark_parameter_sweep():
    """Sweep CRIT parameter to see its effect on performance"""
    print("\n" + "="*80)
    print("BENCHMARK 4: CRIT Parameter Sweep")
    print("="*80)
    
    # 1D cubic with 3 roots
    power_coeffs = np.array([-0.08, 0.74, -1.5, 1.0])
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    system = create_polynomial_system(
        equation_coeffs=[bern_coeffs],
        param_ranges=[(0.0, 1.0)],
        param_names=['t']
    )
    
    expected_solutions = [{'t': 0.2}, {'t': 0.5}, {'t': 0.8}]
    
    # Create benchmark
    benchmark = ScientificBenchmark(output_dir="benchmark_results/crit_sweep")
    
    # Sweep CRIT values
    crit_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = benchmark.parameter_sweep(
        system=system,
        test_name="1d_cubic_crit_sweep",
        parameter_name='crit',
        parameter_values=crit_values,
        method='pp',
        base_config={
            'tolerance': 1e-6,
            'max_depth': 30,
            'subdivision_tolerance': 1e-10,
            'refine': True
        },
        expected_solutions=expected_solutions,
        verbose=True
    )
    
    return benchmark


def benchmark_tolerance_sweep():
    """Sweep tolerance parameter to see its effect on performance"""
    print("\n" + "="*80)
    print("BENCHMARK 5: Tolerance Parameter Sweep")
    print("="*80)
    
    # 1D cubic with 3 roots
    power_coeffs = np.array([-0.08, 0.74, -1.5, 1.0])
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    system = create_polynomial_system(
        equation_coeffs=[bern_coeffs],
        param_ranges=[(0.0, 1.0)],
        param_names=['t']
    )
    
    expected_solutions = [{'t': 0.2}, {'t': 0.5}, {'t': 0.8}]
    
    # Create benchmark
    benchmark = ScientificBenchmark(output_dir="benchmark_results/tolerance_sweep")
    
    # Sweep tolerance values
    tolerance_values = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    
    results = benchmark.parameter_sweep(
        system=system,
        test_name="1d_cubic_tolerance_sweep",
        parameter_name='tolerance',
        parameter_values=tolerance_values,
        method='pp',
        base_config={
            'crit': 0.8,
            'max_depth': 30,
            'subdivision_tolerance': 1e-10,
            'refine': True
        },
        expected_solutions=expected_solutions,
        verbose=True
    )
    
    return benchmark


def main():
    """Run all benchmarks"""
    print("\n" + "="*80)
    print("SCIENTIFIC BENCHMARKING FRAMEWORK")
    print("Polynomial System Solver Performance Testing")
    print("="*80)
    
    # Run individual benchmarks
    benchmark1 = benchmark_1d_quadratic()
    benchmark2 = benchmark_2d_circle_ellipse()
    benchmark3 = benchmark_method_comparison()
    benchmark4 = benchmark_parameter_sweep()
    benchmark5 = benchmark_tolerance_sweep()
    
    # Generate comprehensive reports
    print("\n" + "="*80)
    print("GENERATING REPORTS")
    print("="*80)
    
    benchmark1.generate_report("1d_quadratic_report.txt")
    benchmark2.generate_report("2d_circle_ellipse_report.txt")
    benchmark3.generate_report("method_comparison_report.txt")
    benchmark4.generate_report("crit_sweep_report.txt")
    benchmark5.generate_report("tolerance_sweep_report.txt")
    
    print("\n" + "="*80)
    print("ALL BENCHMARKS COMPLETE")
    print("="*80)
    print("\nResults saved to benchmark_results/ directory")
    print("Check the generated reports for detailed analysis")


if __name__ == "__main__":
    main()

