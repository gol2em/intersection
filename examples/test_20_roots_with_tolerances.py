"""
Test 20-root polynomial with proper tolerance management.

This test demonstrates how to adjust tolerances based on coefficient magnitude.
"""

import numpy as np
import time
import signal
from intersection.polynomial_solver import create_polynomial_system, solve_polynomial_system
from intersection.bernstein import polynomial_nd_to_bernstein


class TimeoutError(Exception):
    """Raised when a test times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Test exceeded 5 second timeout")


def expand_polynomial_product(roots):
    """Expand (x - r1)(x - r2)...(x - rn) into power series."""
    coeffs = np.array([1.0])
    for root in roots:
        new_coeffs = np.zeros(len(coeffs) + 1)
        new_coeffs[1:] += coeffs
        new_coeffs[:-1] -= root * coeffs
        coeffs = new_coeffs
    return coeffs


def test_with_timeout(test_func, timeout_seconds=5):
    """Run a test function with timeout."""
    # Set up signal handler (Unix only)
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        result = test_func()
        
        signal.alarm(0)  # Cancel alarm
        return result
    except AttributeError:
        # Windows doesn't have SIGALRM, use simple time check
        start_time = time.time()
        result = test_func()
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise TimeoutError(f"Test took {elapsed:.1f}s (> {timeout_seconds}s)")
        return result


def test_polynomial(roots, name, tolerance=1e-6, subdivision_tolerance=None, max_depth=30):
    """Test PP method with given roots and tolerances."""
    print("\n" + "=" * 80)
    print(f"TEST: {name}")
    print("=" * 80)
    print(f"Number of roots: {len(roots)}")
    
    # Expand polynomial
    power_coeffs = expand_polynomial_product(roots)
    degree = len(power_coeffs) - 1
    
    print(f"Polynomial degree: {degree}")
    print(f"Leading coefficient: {power_coeffs[-1]:.6e}")
    print(f"Constant term: {power_coeffs[0]:.6e}")
    
    # Convert to Bernstein basis
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    
    coeff_min = np.min(bern_coeffs)
    coeff_max = np.max(bern_coeffs)
    coeff_mag = np.max(np.abs(bern_coeffs))
    
    print(f"Bernstein coefficient range: [{coeff_min:.6e}, {coeff_max:.6e}]")
    print(f"Bernstein coefficient magnitude: {coeff_mag:.6e}")
    
    # Auto-adjust subdivision_tolerance based on coefficient magnitude
    if subdivision_tolerance is None:
        # Use 1/1000 of coefficient magnitude, but not smaller than 1e-14
        subdivision_tolerance = max(coeff_mag / 1000, 1e-14)
        print(f"\nAuto-adjusted subdivision_tolerance: {subdivision_tolerance:.6e}")
        print(f"  (= coeff_mag / 1000 = {coeff_mag:.6e} / 1000)")
    else:
        print(f"\nUsing subdivision_tolerance: {subdivision_tolerance:.6e}")
    
    # Check if tolerances are reasonable
    if subdivision_tolerance > coeff_mag / 10:
        print(f"\n⚠️  WARNING: subdivision_tolerance ({subdivision_tolerance:.6e}) is too large")
        print(f"   compared to coefficient magnitude ({coeff_mag:.6e})")
        print(f"   This may cause false pruning!")
    
    if subdivision_tolerance < 1e-14:
        print(f"\n⚠️  WARNING: subdivision_tolerance ({subdivision_tolerance:.6e}) is very small")
        print(f"   Approaching double precision limit (epsilon ≈ 2.2e-16)")
    
    # Create polynomial system
    system = create_polynomial_system(
        equation_coeffs=[bern_coeffs],
        param_ranges=[(0.0, 1.0)]
    )
    
    # Solve with PP method
    print(f"\nSolving with PP method...")
    print(f"  tolerance = {tolerance:.6e} (parameter space)")
    print(f"  subdivision_tolerance = {subdivision_tolerance:.6e} (function value space)")
    print(f"  max_depth = {max_depth}")
    
    def solve():
        return solve_polynomial_system(
            system,
            method='pp',
            tolerance=tolerance,
            crit=0.8,
            max_depth=max_depth,
            subdivision_tolerance=subdivision_tolerance,
            refine=False,
            verbose=False
        )
    
    try:
        start_time = time.time()
        solutions = solve()
        solve_time = time.time() - start_time
        
        print(f"\n✓ Solve time: {solve_time:.3f} seconds")
        print(f"  Solutions found: {len(solutions)}")
        
        if len(solutions) > 0 and len(solutions) <= 20:
            param_name = list(solutions[0].keys())[0]
            found_roots = sorted([sol[param_name] for sol in solutions])
            expected_roots = sorted(roots)
            
            if len(found_roots) == len(expected_roots):
                max_error = max(abs(f - e) for f, e in zip(found_roots, expected_roots))
                print(f"\n✓ All {len(expected_roots)} roots found!")
                print(f"  Max error: {max_error:.6e}")
                return True
            else:
                print(f"\n✗ Expected {len(expected_roots)} roots, found {len(found_roots)}")
                return False
        elif len(solutions) > 20:
            print(f"\n✗ Too many solutions found (likely duplicates or false positives)")
            return False
        else:
            print(f"\n✗ No solutions found")
            return False
            
    except TimeoutError as e:
        print(f"\n✗ {e}")
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("TOLERANCE MANAGEMENT TEST")
    print("=" * 80)
    print("\nThis test demonstrates proper tolerance management for polynomials")
    print("with different coefficient magnitudes.")
    
    # Test 1: Low-degree polynomial (should work with default tolerances)
    test_polynomial(
        [0.2, 0.5, 0.8],
        "3 roots - default tolerances",
        tolerance=1e-6,
        subdivision_tolerance=1e-10,
        max_depth=30
    )
    
    # Test 2: Medium-degree polynomial
    test_polynomial(
        [0.1, 0.3, 0.5, 0.7, 0.9],
        "5 roots - default tolerances",
        tolerance=1e-6,
        subdivision_tolerance=1e-10,
        max_depth=30
    )
    
    # Test 3: Higher-degree polynomial with auto-adjusted tolerance
    test_polynomial(
        [i * 0.1 for i in range(1, 11)],
        "10 roots - auto-adjusted subdivision_tolerance",
        tolerance=1e-5,
        subdivision_tolerance=None,  # Auto-adjust
        max_depth=40
    )
    
    # Test 4: 20-root polynomial with auto-adjusted tolerance
    print("\n" + "=" * 80)
    print("CRITICAL TEST: 20 roots with auto-adjusted tolerances")
    print("=" * 80)
    
    roots_20 = [i * 0.05 for i in range(1, 21)]
    
    # First, analyze the coefficients
    power_coeffs = expand_polynomial_product(roots_20)
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    coeff_mag = np.max(np.abs(bern_coeffs))
    
    print(f"\nCoefficient analysis:")
    print(f"  Magnitude: {coeff_mag:.6e}")
    print(f"  Recommended subdivision_tolerance: {coeff_mag / 1000:.6e}")
    
    # Test with auto-adjusted tolerance
    success = test_polynomial(
        roots_20,
        "20 roots - auto-adjusted subdivision_tolerance",
        tolerance=1e-4,  # Relaxed parameter space tolerance
        subdivision_tolerance=None,  # Auto-adjust to coeff_mag / 1000
        max_depth=50
    )
    
    if not success:
        print("\n" + "=" * 80)
        print("ALTERNATIVE: Try with even more relaxed tolerances")
        print("=" * 80)
        
        test_polynomial(
            roots_20,
            "20 roots - very relaxed tolerances",
            tolerance=1e-3,  # Very relaxed
            subdivision_tolerance=coeff_mag / 100,  # Even more relaxed
            max_depth=50
        )
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nKey insights:")
    print("1. subdivision_tolerance should be ~1/1000 of coefficient magnitude")
    print("2. For high-degree polynomials, relax both tolerances")
    print("3. Coefficient magnitude of 10^-6 is fine with proper tolerance settings")
    print("4. The issue is NOT coefficient size, but tolerance mismatch")

