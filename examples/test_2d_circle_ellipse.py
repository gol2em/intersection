"""
Test 2D polynomial system: Circle and Ellipse intersection.

System:
  x² + y² - 1 = 0        (circle of radius 1)
  x²/4 + 4y² - 1 = 0     (ellipse)
  
Domain: [0, 1] × [0, 1]

Expected: Exactly 1 solution in the first quadrant.
"""

import numpy as np
import matplotlib.pyplot as plt
from intersection.polynomial_solver import solve_polynomial_system, PolynomialSystem
from intersection.bernstein import polynomial_nd_to_bernstein
import time


def solve_analytical():
    """
    Solve the system analytically.
    
    From eq1: y² = 1 - x²
    Substitute into eq2: x²/4 + 4(1 - x²) - 1 = 0
                        x²/4 + 4 - 4x² - 1 = 0
                        x²/4 - 4x² + 3 = 0
                        x²(1/4 - 4) + 3 = 0
                        x²(-15/4) + 3 = 0
                        x² = 3 / (15/4) = 12/15 = 4/5
                        x = ±√(4/5) = ±2/√5
    
    In [0,1]: x = 2/√5 ≈ 0.894427
    
    From eq1: y² = 1 - 4/5 = 1/5
              y = ±1/√5
    
    In [0,1]: y = 1/√5 ≈ 0.447214
    """
    x = 2 / np.sqrt(5)
    y = 1 / np.sqrt(5)
    
    # Verify
    eq1 = x**2 + y**2 - 1
    eq2 = x**2/4 + 4*y**2 - 1
    
    print("Analytical Solution:")
    print(f"  x = 2/√5 = {x:.6f}")
    print(f"  y = 1/√5 = {y:.6f}")
    print(f"\nVerification:")
    print(f"  eq1 (x² + y² - 1) = {eq1:.6e}")
    print(f"  eq2 (x²/4 + 4y² - 1) = {eq2:.6e}")
    
    return x, y


def create_visualization(solution=None):
    """Visualize the curves and solution."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create grid
    x = np.linspace(0, 1, 500)
    y = np.linspace(0, 1, 500)
    X, Y = np.meshgrid(x, y)
    
    # Equation 1: x² + y² - 1 = 0
    Z1 = X**2 + Y**2 - 1
    
    # Equation 2: x²/4 + 4y² - 1 = 0
    Z2 = X**2/4 + 4*Y**2 - 1
    
    # Plot contours
    ax.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2, label='x² + y² = 1')
    ax.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2, label='x²/4 + 4y² = 1')
    
    # Plot analytical solution
    x_exact, y_exact = solve_analytical()
    ax.plot(x_exact, y_exact, 'go', markersize=12, label='Analytical', zorder=5)
    
    # Plot numerical solution if provided
    if solution is not None:
        ax.plot(solution['x'], solution['y'], 'r*', markersize=15, 
               label='Numerical', zorder=6)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Circle-Ellipse Intersection', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('test_2d_circle_ellipse.png', dpi=150, bbox_inches='tight')
    print("\nSaved: test_2d_circle_ellipse.png")
    plt.close()


if __name__ == "__main__":
    print("=" * 80)
    print("2D POLYNOMIAL SYSTEM TEST: Circle-Ellipse Intersection")
    print("=" * 80)
    
    # Analytical solution
    print("\n" + "=" * 80)
    x_exact, y_exact = solve_analytical()
    print("=" * 80)
    
    # Define polynomials in power form
    # Equation 1: x² + y² - 1 = 0
    # In power form: coeffs[i,j] is coefficient of x^i * y^j
    # We need degree 2 in both variables
    eq1_power = np.zeros((3, 3))  # degree 2 → size 3×3
    eq1_power[0, 0] = -1  # constant term
    eq1_power[2, 0] = 1   # x² term
    eq1_power[0, 2] = 1   # y² term
    
    # Equation 2: x²/4 + 4y² - 1 = 0
    eq2_power = np.zeros((3, 3))
    eq2_power[0, 0] = -1    # constant term
    eq2_power[2, 0] = 0.25  # x²/4 term
    eq2_power[0, 2] = 4     # 4y² term
    
    print("\n" + "=" * 80)
    print("POWER FORM COEFFICIENTS")
    print("=" * 80)
    print("\nEquation 1 (x² + y² - 1):")
    print(eq1_power)
    print("\nEquation 2 (x²/4 + 4y² - 1):")
    print(eq2_power)
    
    # Convert to Bernstein form
    print("\n" + "=" * 80)
    print("CONVERTING TO BERNSTEIN FORM")
    print("=" * 80)
    
    eq1_bern = polynomial_nd_to_bernstein(eq1_power, k=2)
    eq2_bern = polynomial_nd_to_bernstein(eq2_power, k=2)
    
    print("\nEquation 1 Bernstein coefficients:")
    print(eq1_bern)
    print(f"Shape: {eq1_bern.shape}")
    print(f"Magnitude: {np.max(np.abs(eq1_bern)):.6e}")
    
    print("\nEquation 2 Bernstein coefficients:")
    print(eq2_bern)
    print(f"Shape: {eq2_bern.shape}")
    print(f"Magnitude: {np.max(np.abs(eq2_bern)):.6e}")
    
    # Create polynomial system
    system = PolynomialSystem(
        equation_coeffs=[eq1_bern, eq2_bern],
        param_ranges=[(0.0, 1.0), (0.0, 1.0)],
        k=2,
        degree=2,
        param_names=['x', 'y']
    )
    
    # Solve with PP method
    print("\n" + "=" * 80)
    print("SOLVING WITH PP METHOD")
    print("=" * 80)
    
    start_time = time.time()
    
    solutions = solve_polynomial_system(
        system,
        method='pp',
        tolerance=1e-6,
        crit=0.5,  # Lower CRIT to allow more tightening
        max_depth=50,  # Increase max depth
        subdivision_tolerance=1e-10,
        refine=False,
        verbose=True
    )
    
    solve_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Solve time: {solve_time:.3f} seconds")
    print(f"Solutions found: {len(solutions)}")
    
    if len(solutions) == 0:
        print("\n✗ FAILURE: No solutions found!")
        create_visualization()
    else:
        print("\nNumerical solutions:")
        for i, sol in enumerate(solutions):
            print(f"\n  Solution {i+1}:")
            print(f"    x = {sol['x']:.6f}")
            print(f"    y = {sol['y']:.6f}")
            
            # Verify
            x, y = sol['x'], sol['y']
            eq1_val = x**2 + y**2 - 1
            eq2_val = x**2/4 + 4*y**2 - 1
            print(f"    eq1 residual: {eq1_val:.6e}")
            print(f"    eq2 residual: {eq2_val:.6e}")
        
        # Compare with analytical
        print("\n" + "=" * 80)
        print("COMPARISON WITH ANALYTICAL SOLUTION")
        print("=" * 80)
        
        if len(solutions) == 1:
            sol = solutions[0]
            error_x = abs(sol['x'] - x_exact)
            error_y = abs(sol['y'] - y_exact)
            error_norm = np.sqrt(error_x**2 + error_y**2)
            
            print(f"Error in x: {error_x:.6e}")
            print(f"Error in y: {error_y:.6e}")
            print(f"Euclidean error: {error_norm:.6e}")
            
            if error_norm < 1e-5:
                print("\n✓ SUCCESS: Found correct solution!")
            else:
                print("\n✗ FAILURE: Solution error too large!")
            
            create_visualization(sol)
        else:
            print(f"\n✗ FAILURE: Expected 1 solution, found {len(solutions)}")
            create_visualization(solutions[0] if solutions else None)

