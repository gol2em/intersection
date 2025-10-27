"""
Quick Benchmark Script

Run a single benchmark test quickly to verify solver performance.
"""

import sys
sys.path.insert(0, 'D:/Python/Intersection')

import numpy as np
from intersection.polynomial_solver import create_polynomial_system
from intersection.bernstein import polynomial_nd_to_bernstein
from intersection.benchmark import ScientificBenchmark


def quick_test_1d():
    """Quick test: 1D quadratic"""
    print("\n" + "="*80)
    print("QUICK BENCHMARK: 1D Quadratic")
    print("="*80)
    
    # (t - 0.3)(t - 0.7) = t^2 - t + 0.21
    power_coeffs = np.array([0.21, -1.0, 1.0])
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    system = create_polynomial_system(
        equation_coeffs=[bern_coeffs],
        param_ranges=[(0.0, 1.0)],
        param_names=['t']
    )
    
    expected_solutions = [{'t': 0.3}, {'t': 0.7}]
    
    benchmark = ScientificBenchmark(output_dir="benchmark_results/quick")
    
    result = benchmark.run_benchmark(
        system=system,
        test_name="quick_1d_quadratic",
        test_description="Quick test: 1D quadratic with 2 roots",
        method='pp',
        tolerance=1e-6,
        crit=0.8,
        max_depth=30,
        subdivision_tolerance=1e-10,
        refine=True,
        expected_solutions=expected_solutions,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("QUICK SUMMARY")
    print("="*80)
    print(f"CPU Time:        {result.cpu_time:.6f} seconds")
    print(f"Boxes Processed: {result.boxes_processed}")
    print(f"Boxes Pruned:    {result.boxes_pruned}")
    print(f"Solutions Found: {result.num_solutions}")
    print(f"Expected:        {len(expected_solutions)}")
    print(f"Max Error:       {result.max_error:.2e}" if result.max_error else "Max Error:       N/A")
    print(f"Max Residual:    {result.max_residual:.2e}" if result.max_residual else "Max Residual:    N/A")
    print("="*80)
    
    return result


def quick_test_2d():
    """Quick test: 2D circle-ellipse"""
    print("\n" + "="*80)
    print("QUICK BENCHMARK: 2D Circle-Ellipse Intersection")
    print("="*80)
    
    # Equation 1: x^2 + y^2 - 1 = 0
    eq1_power = np.array([
        [-1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    
    # Equation 2: x^2/4 + 4*y^2 - 1 = 0
    eq2_power = np.array([
        [-1.0, 0.0, 4.0],
        [0.0, 0.0, 0.0],
        [0.25, 0.0, 0.0]
    ])
    
    eq1_bern = polynomial_nd_to_bernstein(eq1_power, k=2)
    eq2_bern = polynomial_nd_to_bernstein(eq2_power, k=2)
    
    system = create_polynomial_system(
        equation_coeffs=[eq1_bern, eq2_bern],
        param_ranges=[(0.0, 1.0), (0.0, 1.0)],
        param_names=['x', 'y']
    )
    
    expected_solutions = [{'x': 0.894427, 'y': 0.447214}]
    
    benchmark = ScientificBenchmark(output_dir="benchmark_results/quick")
    
    result = benchmark.run_benchmark(
        system=system,
        test_name="quick_2d_circle_ellipse",
        test_description="Quick test: 2D circle-ellipse intersection",
        method='pp',
        tolerance=1e-6,
        crit=0.8,
        max_depth=50,
        subdivision_tolerance=1e-10,
        refine=False,
        expected_solutions=expected_solutions,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("QUICK SUMMARY")
    print("="*80)
    print(f"CPU Time:        {result.cpu_time:.6f} seconds")
    print(f"Boxes Processed: {result.boxes_processed}")
    print(f"Boxes Pruned:    {result.boxes_pruned}")
    print(f"Solutions Found: {result.num_solutions}")
    print(f"Expected:        {len(expected_solutions)}")
    print(f"Max Error:       {result.max_error:.2e}" if result.max_error else "Max Error:       N/A")
    print(f"Max Residual:    {result.max_residual:.2e}" if result.max_residual else "Max Residual:    N/A")
    print("="*80)
    
    return result


def quick_test_3_roots():
    """Quick test: 1D cubic with 3 roots"""
    print("\n" + "="*80)
    print("QUICK BENCHMARK: 1D Cubic with 3 Roots")
    print("="*80)
    
    # (t - 0.2)(t - 0.5)(t - 0.8) = t^3 - 1.5*t^2 + 0.74*t - 0.08
    power_coeffs = np.array([-0.08, 0.74, -1.5, 1.0])
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    system = create_polynomial_system(
        equation_coeffs=[bern_coeffs],
        param_ranges=[(0.0, 1.0)],
        param_names=['t']
    )
    
    expected_solutions = [{'t': 0.2}, {'t': 0.5}, {'t': 0.8}]
    
    benchmark = ScientificBenchmark(output_dir="benchmark_results/quick")
    
    result = benchmark.run_benchmark(
        system=system,
        test_name="quick_1d_cubic_3roots",
        test_description="Quick test: 1D cubic with 3 roots",
        method='pp',
        tolerance=1e-6,
        crit=0.8,
        max_depth=30,
        subdivision_tolerance=1e-10,
        refine=True,
        expected_solutions=expected_solutions,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("QUICK SUMMARY")
    print("="*80)
    print(f"CPU Time:        {result.cpu_time:.6f} seconds")
    print(f"Boxes Processed: {result.boxes_processed}")
    print(f"Boxes Pruned:    {result.boxes_pruned}")
    print(f"Solutions Found: {result.num_solutions}")
    print(f"Expected:        {len(expected_solutions)}")
    print(f"Max Error:       {result.max_error:.2e}" if result.max_error else "Max Error:       N/A")
    print(f"Max Residual:    {result.max_residual:.2e}" if result.max_residual else "Max Residual:    N/A")
    print("="*80)
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick benchmark tests')
    parser.add_argument('test', nargs='?', default='all',
                       choices=['1d', '2d', '3roots', 'all'],
                       help='Which test to run (default: all)')
    
    args = parser.parse_args()
    
    if args.test == '1d' or args.test == 'all':
        quick_test_1d()
    
    if args.test == '2d' or args.test == 'all':
        quick_test_2d()
    
    if args.test == '3roots' or args.test == 'all':
        quick_test_3_roots()
    
    print("\n" + "="*80)
    print("QUICK BENCHMARK COMPLETE")
    print("="*80)
    print("\nResults saved to benchmark_results/quick/")

