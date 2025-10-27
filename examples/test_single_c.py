"""Quick test for a single c value to debug timeout issues."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmark_examples import Example1DHighDegree
from src.intersection.polynomial_solver import solve_polynomial_system
import time

# Test c = 0 which seems to timeout
c = 0.0
print(f"Testing c = {c}")

example = Example1DHighDegree(c=c)
config = example.setup()

system = config['system']
expected_roots = config['expected_roots']
tolerance = config.get('tolerance', 1e-6)
max_depth = config.get('max_depth', 30)
crit = config.get('crit', 0.8)
refine = config.get('refine', False)

print(f"Expected roots: {len(expected_roots)}")
print(f"Tolerance: {tolerance}")
print(f"Max depth: {max_depth}")

start = time.time()
print("\nStarting solve with 5 second timeout...")

solutions, solver_stats = solve_polynomial_system(
    system,
    tolerance=tolerance,
    max_depth=max_depth,
    crit=crit,
    refine=refine,
    verbose=True,
    return_stats=True,
    timeout_seconds=5.0
)

elapsed = time.time() - start

print(f"\nCompleted in {elapsed:.3f}s")
print(f"Timed out: {solver_stats.get('timed_out', False)}")
print(f"Boxes processed: {solver_stats.get('boxes_processed', 0)}")
print(f"Solutions found: {len(solutions)}")

