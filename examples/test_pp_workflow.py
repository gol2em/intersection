"""
Test to verify the correct PP workflow:
1. Find convex hull intersection → get bounds [t_min, t_max]
2. If bounds are tight (width <= CRIT): use them directly
3. If bounds are loose (width > CRIT): bisect them

This should show that PP method uses tighter bounds instead of always bisecting.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.intersection.polynomial_solver import create_polynomial_system, solve_polynomial_system
from src.intersection.bernstein import polynomial_nd_to_bernstein

def test_pp_workflow():
    """Test with a simple polynomial to see PP workflow."""
    print("\n" + "=" * 80)
    print("TEST: PP Workflow Verification")
    print("=" * 80)
    
    # Simple polynomial: (x - 0.3)(x - 0.7) = x^2 - x + 0.21
    # Roots at x = 0.3, 0.7
    power_coeffs = np.array([0.21, -1.0, 1.0])
    
    print("\nPolynomial: (x - 0.3)(x - 0.7)")
    print("Expected roots: x = 0.3, 0.7")
    print(f"Power coefficients: {power_coeffs}")
    
    # Convert to Bernstein
    bern_coeffs = polynomial_nd_to_bernstein(power_coeffs, k=1)
    print(f"\nBernstein coefficients: {bern_coeffs}")
    
    # Create system
    system = create_polynomial_system(
        equation_coeffs=[bern_coeffs],
        param_ranges=[(0.0, 1.0)]
    )
    
    # Solve with PP method with VERBOSE output
    print("\n" + "=" * 80)
    print("SOLVING WITH PP METHOD (verbose=True)")
    print("=" * 80)
    
    solutions = solve_polynomial_system(
        system,
        method='pp',
        tolerance=1e-6,
        crit=0.5,  # Use 0.5 as critical ratio
        max_depth=20,
        refine=False,
        verbose=True  # This should show the subdivision process
    )
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Solutions found: {len(solutions)}")
    for i, sol in enumerate(solutions, 1):
        x = sol['x']
        residual = 0.21 - x + x**2
        print(f"  {i}. x = {x:.8f} (residual: {residual:.6e})")
    
    # Check accuracy
    expected = [0.3, 0.7]
    print("\nExpected roots: [0.3, 0.7]")
    
    found_x = sorted([sol['x'] for sol in solutions])
    errors = [abs(found_x[i] - expected[i]) for i in range(min(len(found_x), len(expected)))]
    
    if len(errors) == 2:
        print(f"Errors: [{errors[0]:.6e}, {errors[1]:.6e}]")
        print(f"Max error: {max(errors):.6e}")
        
        if max(errors) < 1e-5:
            print("\n✅ SUCCESS! Both roots found accurately")
        else:
            print("\n⚠️  Roots found but with larger errors")
    else:
        print(f"\n❌ FAILED! Found {len(found_x)} roots instead of 2")

if __name__ == "__main__":
    test_pp_workflow()

