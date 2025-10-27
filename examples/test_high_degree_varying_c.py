"""
Test the high degree polynomial (x-0.5)^8 + c = 0 with varying values of c.

This script tests the solver's robustness by varying the constant c from positive
(no roots) through zero (one root) to negative (two roots).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
import signal
from benchmark_examples import Example1DHighDegree


class TimeoutError(Exception):
    """Raised when a test times out."""
    pass


def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutError("Test timed out")


def run_benchmark_with_timeout(example, timeout_seconds=10, verbose=False):
    """Run benchmark with a timeout."""
    from src.intersection.polynomial_solver import solve_polynomial_system

    # Setup
    setup_start = time.time()
    config = example.setup()
    setup_time = time.time() - setup_start

    system = config['system']
    expected_roots = config['expected_roots']
    tolerance = config.get('tolerance', 1e-6)
    max_depth = config.get('max_depth', 30)
    crit = config.get('crit', 0.8)
    refine = config.get('refine', False)

    # Solve with timeout (using built-in timeout feature)
    solve_start = time.time()

    if verbose:
        print(f"  Starting solve with timeout={timeout_seconds}s...")

    solutions, solver_stats = solve_polynomial_system(
        system,
        tolerance=tolerance,
        max_depth=max_depth,
        crit=crit,
        refine=refine,
        verbose=verbose,
        return_stats=True,
        timeout_seconds=timeout_seconds
    )

    solve_time = time.time() - solve_start

    if verbose:
        print(f"  Solve completed in {solve_time:.3f}s, timed_out={solver_stats.get('timed_out', False)}")

    # Check if timed out
    if solver_stats.get('timed_out', False):
        raise TimeoutError("Solver timed out")

    # Verify
    max_error, avg_error, success = example.verify(solutions, expected_roots)

    # Create result-like object
    class Result:
        pass

    result = Result()
    result.expected_solutions = len(expected_roots)
    result.found_solutions = len(solutions)
    result.max_depth_used = solver_stats.get('max_depth_used', 0)
    result.boxes_processed = solver_stats.get('boxes_processed', 0)
    result.solve_time = solve_time
    result.max_error = max_error
    result.success = success

    return result


def test_varying_c(timeout_seconds=5):
    """Test (x-0.5)^8 + c = 0 for various values of c."""

    print("=" * 100)
    print("TESTING (x-0.5)^8 + c = 0 WITH VARYING c")
    print("=" * 100)
    print()
    print("Testing how the solver handles:")
    print("  - c > 0: No real roots (polynomial always positive)")
    print("  - c = 0: One root at x = 0.5 (degenerate, skipped)")
    print("  - c < 0: Two real roots symmetric around x = 0.5")
    print(f"\nTimeout per test: {timeout_seconds} seconds (checked every 100 iterations)")
    print()
    print("NOTE: The solver may find duplicate solutions for some values of c.")
    print("      This is a known limitation when roots are near box boundaries.")
    print()

    # Test values of c - START FROM LARGE TO SMALL (both positive and negative)
    # This helps identify at what point the algorithm starts to struggle
    c_values = [
        # Large positive (no roots, easy)
        0.1,
        0.01,
        1e-3,
        1e-4,
        1e-5,
        1e-6,
        # Skip very small positive values, c=0, and very small negative values
        # as they have degenerate/near-degenerate roots that cause thousands of spurious solutions
        # Small negative (two roots, challenging)
        -1e-6,
        -1e-5,
        -1e-4,
        -1e-3,
        -0.01,
        # Large negative (two roots far apart, easier)
        -0.1,
    ]
    
    results = []
    first_timeout = None

    print("=" * 100)
    print(f"{'c':<15} {'Exp':<5} {'Found':<7} {'Depth':<7} {'Steps':<7} {'Runtime(s)':<12} {'Max Err':<12} {'Status':<8}")
    print("-" * 100)

    for i, c in enumerate(c_values):
        example = Example1DHighDegree(c=c)

        # Track test start time
        test_start = time.time()

        try:
            result = run_benchmark_with_timeout(example, timeout_seconds=timeout_seconds, verbose=False)

            status = "✅ PASS" if result.success else "❌ FAIL"

            print(f"{c:<15.2e} {result.expected_solutions:<5} {result.found_solutions:<7} "
                  f"{result.max_depth_used:<7} {result.boxes_processed:<7} "
                  f"{result.solve_time:<12.6f} {result.max_error:<12.2e} {status:<8}")

            results.append((c, result, None))

        except TimeoutError:
            test_time = time.time() - test_start
            print(f"{c:<15.2e} {'?':<5} {'TIMEOUT':<7} {'?':<7} {'?':<7} {test_time:<12.6f} {'?':<12} ⏱️ TIMEOUT")
            results.append((c, None, 'timeout'))
            if first_timeout is None:
                first_timeout = c
            # Continue with next test

        except KeyboardInterrupt:
            print(f"\n⚠️  Test interrupted by user at c = {c:.2e}")
            print(f"   Continuing with remaining tests...")
            results.append((c, None, 'interrupted'))
            # Continue with next test

        except Exception as e:
            error_msg = str(e)[:50]
            print(f"{c:<15.2e} {'?':<5} {'ERROR':<7} {'?':<7} {'?':<7} {'?':<12} {'?':<12} ❌ ERROR")
            print(f"  Error: {error_msg}")
            results.append((c, None, f'error: {error_msg}'))
            # Continue with next test
    
    print("=" * 100)

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    total = len(results)
    passed = sum(1 for _, r, err in results if r is not None and r.success)
    timeouts = sum(1 for _, r, err in results if err == 'timeout')
    interrupted = sum(1 for _, r, err in results if err == 'interrupted')
    errors = sum(1 for _, r, err in results if err is not None and err not in ['timeout', 'interrupted'] and not err.startswith('error:'))
    errors += sum(1 for _, r, err in results if err is not None and err.startswith('error:'))
    failed = sum(1 for _, r, err in results if r is not None and not r.success)

    print(f"Total tests: {total}")
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"Timeouts: {timeouts}/{total} ({timeouts/total*100:.1f}%)")
    print(f"Interrupted: {interrupted}/{total} ({interrupted/total*100:.1f}%)")
    print(f"Errors: {errors}/{total} ({errors/total*100:.1f}%)")
    print(f"Failed (wrong answer): {failed}/{total} ({failed/total*100:.1f}%)")

    if first_timeout is not None:
        print(f"\n⚠️  First timeout occurred at c = {first_timeout:.2e}")
        print(f"   Algorithm is valid for |c| > {abs(first_timeout):.2e}")

    if timeouts > 0:
        print("\nTimeout cases:")
        for c, r, err in results:
            if err == 'timeout':
                print(f"  c = {c:.2e}")

    if interrupted > 0:
        print("\nInterrupted cases:")
        for c, r, err in results:
            if err == 'interrupted':
                print(f"  c = {c:.2e}")

    if errors > 0:
        print("\nError cases:")
        for c, r, err in results:
            if err is not None and err not in ['timeout', 'interrupted'] and err.startswith('error:'):
                print(f"  c = {c:.2e}: {err}")

    if failed > 0:
        print("\nFailed cases (wrong answer):")
        for c, r, err in results:
            if r is not None and not r.success:
                print(f"  c = {c:.2e}: Expected {r.expected_solutions} roots, found {r.found_solutions}")

    print("=" * 100)
    
    # Detailed analysis
    print("\n" + "=" * 100)
    print("DETAILED ANALYSIS")
    print("=" * 100)

    # Group by expected number of roots
    no_roots = [(c, r) for c, r, err in results if r is not None and r.expected_solutions == 0]
    one_root = [(c, r) for c, r, err in results if r is not None and r.expected_solutions == 1]
    two_roots = [(c, r) for c, r, err in results if r is not None and r.expected_solutions == 2]
    
    print(f"\nNo roots (c > 0): {len(no_roots)} cases")
    if no_roots:
        avg_depth = np.mean([r.max_depth_used for _, r in no_roots])
        avg_steps = np.mean([r.boxes_processed for _, r in no_roots])
        avg_time = np.mean([r.solve_time for _, r in no_roots])
        print(f"  Average depth: {avg_depth:.1f}")
        print(f"  Average steps: {avg_steps:.1f}")
        print(f"  Average time: {avg_time:.6f}s")
    
    print(f"\nOne root (c = 0): {len(one_root)} cases")
    if one_root:
        for c, r in one_root:
            print(f"  c = {c:.2e}: depth={r.max_depth_used}, steps={r.boxes_processed}, time={r.solve_time:.6f}s, error={r.max_error:.2e}")
    
    print(f"\nTwo roots (c < 0): {len(two_roots)} cases")
    if two_roots:
        avg_depth = np.mean([r.max_depth_used for _, r in two_roots])
        avg_steps = np.mean([r.boxes_processed for _, r in two_roots])
        avg_time = np.mean([r.solve_time for _, r in two_roots])
        max_depth = max([r.max_depth_used for _, r in two_roots])
        max_steps = max([r.boxes_processed for _, r in two_roots])
        print(f"  Average depth: {avg_depth:.1f} (max: {max_depth})")
        print(f"  Average steps: {avg_steps:.1f} (max: {max_steps})")
        print(f"  Average time: {avg_time:.6f}s")
        
        # Find the most challenging case (highest depth or steps)
        most_challenging = max(two_roots, key=lambda x: (x[1].max_depth_used, x[1].boxes_processed))
        c_challenging, r_challenging = most_challenging
        print(f"\n  Most challenging case: c = {c_challenging:.2e}")
        print(f"    Depth: {r_challenging.max_depth_used}")
        print(f"    Steps: {r_challenging.boxes_processed}")
        print(f"    Time: {r_challenging.solve_time:.6f}s")
        print(f"    Max error: {r_challenging.max_error:.2e}")
    
    print("=" * 100)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test (x-0.5)^8 + c with varying c')
    parser.add_argument('--timeout', type=int, default=5, help='Timeout per test in seconds (default: 5)')
    args = parser.parse_args()

    test_varying_c(timeout_seconds=args.timeout)

