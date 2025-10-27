"""
Test the limits of root multiplicities that the algorithm can handle.

This test examines polynomials of the form (x - 0.5)^m = 0 where m is the multiplicity.
For a root of multiplicity m, the polynomial and its first (m-1) derivatives all vanish at the root,
making it increasingly difficult to detect and isolate.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.intersection.polynomial_solver import create_polynomial_system, solve_polynomial_system
from src.intersection.bernstein import polynomial_nd_to_bernstein
import numpy as np
import time
import argparse
from math import comb


def test_multiplicity(multiplicity, tolerance=1e-6, max_depth=None, timeout_seconds=10, verbose=False):
    """
    Test the solver on a polynomial with a root of given multiplicity.

    Returns:
        dict with test results
    """
    if not verbose:
        print(f"Testing multiplicity {multiplicity}...", end=' ', flush=True)

    try:
        # Create polynomial (x - 0.5)^m in power basis
        # Expand using binomial theorem
        coeffs = np.zeros(multiplicity + 1)
        for k in range(multiplicity + 1):
            coeffs[k] = comb(multiplicity, k) * ((-0.5) ** (multiplicity - k))

        # Convert to Bernstein basis
        bern_coeffs = polynomial_nd_to_bernstein(coeffs, k=1)

        # Create system
        system = create_polynomial_system(
            equation_coeffs=[bern_coeffs],
            param_ranges=[(0.0, 1.0)],
            param_names=['x']
        )

        # Expected: one root at x = 0.5
        expected_roots = [{'x': 0.5}]

        # Solve
        start = time.time()
        solutions, stats = solve_polynomial_system(
            system,
            tolerance=tolerance,
            max_depth=max_depth,
            crit=0.8,
            refine=False,
            verbose=verbose,
            return_stats=True,
            timeout_seconds=timeout_seconds
        )
        elapsed = time.time() - start

        # Check if timed out
        if stats.get('timed_out', False):
            if not verbose:
                print("⏱️ TIMEOUT")
            return {
                'multiplicity': multiplicity,
                'status': 'TIMEOUT',
                'time': elapsed,
                'boxes': stats.get('boxes_processed', 0),
                'depth': stats.get('max_depth_used', 0),
                'solutions': len(solutions)
            }

        # Check solutions
        # For high multiplicity, we may find duplicates or no solutions
        found_root = False
        num_near_root = 0
        for sol in solutions:
            if abs(sol['x'] - 0.5) < tolerance * 10:  # Within 10x tolerance
                num_near_root += 1
                if abs(sol['x'] - 0.5) < tolerance:
                    found_root = True

        # Determine status
        if found_root and num_near_root == 1:
            status = 'PASS'
        elif found_root and num_near_root > 1:
            status = 'DUPLICATES'
        elif num_near_root > 0:
            status = 'NEAR'
        elif len(solutions) == 0:
            status = 'NO_SOLUTION'
        else:
            status = 'FAIL'

        if not verbose:
            status_symbol = {
                'PASS': '✅',
                'DUPLICATES': '⚠️',
                'NEAR': '⚠️',
                'NO_SOLUTION': '❌',
                'FAIL': '❌'
            }.get(status, '?')
            print(f"{status_symbol} {status}")

        return {
            'multiplicity': multiplicity,
            'status': status,
            'time': elapsed,
            'boxes': stats.get('boxes_processed', 0),
            'depth': stats.get('max_depth_used', 0),
            'solutions': len(solutions),
            'near_root': num_near_root
        }

    except Exception as e:
        if not verbose:
            print(f"💥 ERROR: {str(e)}")
        return {
            'multiplicity': multiplicity,
            'status': 'ERROR',
            'error': str(e),
            'time': 0,
            'boxes': 0,
            'depth': 0,
            'solutions': 0
        }


def main():
    parser = argparse.ArgumentParser(description='Test multiplicity limits of the polynomial solver')
    parser.add_argument('--max-mult', type=int, default=10, help='Maximum multiplicity to test')
    parser.add_argument('--timeout', type=int, default=10, help='Timeout per test in seconds')
    parser.add_argument('--tolerance', type=float, default=1e-6, help='Solver tolerance')
    parser.add_argument('--max-depth', type=int, default=None, help='Max depth (None = auto-calculate)')
    parser.add_argument('--verbose', action='store_true', help='Verbose solver output')
    args = parser.parse_args()
    
    print("=" * 100)
    print("TESTING ROOT MULTIPLICITY LIMITS")
    print("=" * 100)
    print()
    print("Testing polynomials of the form (x - 0.5)^m = 0")
    print("where m is the multiplicity of the root at x = 0.5")
    print()
    print(f"Tolerance: {args.tolerance}")
    print(f"Max depth: {args.max_depth if args.max_depth else 'Auto (ceil(d * log2(m)) + 5)'}")
    print(f"Timeout per test: {args.timeout} seconds")
    print(f"Testing multiplicities: 1 to {args.max_mult}")
    print()
    
    results = []
    
    print("=" * 100)
    print(f"{'Mult':<6} {'Status':<15} {'Solutions':<10} {'Near Root':<10} {'Depth':<7} {'Boxes':<8} {'Time(s)':<10}")
    print("-" * 100)

    for m in range(1, args.max_mult + 1):
        result = test_multiplicity(
            m,
            tolerance=args.tolerance,
            max_depth=args.max_depth,
            timeout_seconds=args.timeout,
            verbose=args.verbose
        )
        results.append(result)

        # Print result
        status_symbol = {
            'PASS': '✅ PASS',
            'FAIL': '❌ FAIL',
            'TIMEOUT': '⏱️ TIMEOUT',
            'DUPLICATES': '⚠️ DUPES',
            'NO_SOLUTION': '❌ NO_SOL',
            'NEAR': '⚠️ NEAR',
            'ERROR': '💥 ERROR'
        }.get(result['status'], result['status'])

        print(f"{result['multiplicity']:<6} {status_symbol:<15} "
              f"{result.get('solutions', 0):<10} "
              f"{result.get('near_root', 0):<10} "
              f"{result.get('depth', 0):<7} "
              f"{result.get('boxes', 0):<8} "
              f"{result.get('time', 0):<10.6f}")

    print("=" * 100)
    print()
    
    # Summary
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = sum(1 for r in results if r['status'] in ['FAIL', 'NO_SOLUTION'])
    timeout = sum(1 for r in results if r['status'] == 'TIMEOUT')
    duplicates = sum(1 for r in results if r['status'] in ['DUPLICATES', 'NEAR'])
    errors = sum(1 for r in results if r['status'] == 'ERROR')
    
    print(f"Total tests: {len(results)}")
    print(f"Passed: {passed}/{len(results)} ({100*passed/len(results):.1f}%)")
    print(f"Failed: {failed}/{len(results)} ({100*failed/len(results):.1f}%)")
    print(f"Timeouts: {timeout}/{len(results)} ({100*timeout/len(results):.1f}%)")
    print(f"Duplicates: {duplicates}/{len(results)} ({100*duplicates/len(results):.1f}%)")
    print(f"Errors: {errors}/{len(results)} ({100*errors/len(results):.1f}%)")
    print()
    
    # Find the maximum multiplicity that passed
    max_passed = max([r['multiplicity'] for r in results if r['status'] == 'PASS'], default=0)
    print(f"Maximum multiplicity handled successfully: {max_passed}")
    
    # Find first failure
    first_fail = next((r for r in results if r['status'] in ['FAIL', 'NO_SOLUTION', 'TIMEOUT', 'ERROR']), None)
    if first_fail:
        print(f"First failure at multiplicity: {first_fail['multiplicity']} ({first_fail['status']})")

    # Find first duplicates
    first_dup = next((r for r in results if r['status'] in ['DUPLICATES', 'NEAR']), None)
    if first_dup:
        print(f"First duplicates at multiplicity: {first_dup['multiplicity']} ({first_dup.get('near_root', 0)} solutions near root)")

    print("=" * 100)


if __name__ == '__main__':
    main()

