# Intersection Computation

Compute all intersections of a straight line with parametric curves (2D) or surfaces (3D).

## Algorithm Pipeline

1. **Interpolation**: Interpolate the parametric curve/surface as polynomials
2. **Bernstein Basis**: Convert polynomials to Bernstein basis representation
3. **Polynomial System**: Combine the straight line with polynomials to form a system
4. **Solver**: Solve the polynomial system using custom algorithm
5. **Point Conversion**: Convert solutions to intersection points

## Project Structure

```
intersection/
├── src/
│   └── intersection/
│       ├── __init__.py
│       ├── interpolation.py      # Step 1: Polynomial interpolation
│       ├── bernstein.py          # Step 2: Bernstein basis conversion
│       ├── polynomial_system.py  # Step 3: System formation
│       ├── solver.py             # Step 4: System solver
│       ├── geometry.py           # Geometric primitives (Line, Curve, Surface)
│       └── utils.py              # Utilities and visualization
├── examples/
│   ├── example_2d.py             # 2D curve intersection example
│   └── example_3d.py             # 3D surface intersection example
├── tests/
│   └── test_*.py                 # Unit tests
└── pyproject.toml
```

## Installation

```bash
# Install dependencies using uv
uv sync

# Or install in development mode
uv sync --all-extras
```

## Usage

```python
from intersection import Line2D, ParametricCurve, compute_intersections_2d

# Define a parametric curve
def curve_x(t): return t
def curve_y(t): return t**2

curve = ParametricCurve(curve_x, curve_y, t_range=(0, 2))

# Define a line
line = Line2D(point=(0, 0.5), direction=(1, 0))

# Compute intersections
intersections = compute_intersections_2d(line, curve, verbose=True)
```

## Running with uv

```bash
# Run examples
uv run python examples/example_2d.py
uv run python examples/example_3d.py

# Run tests
uv run pytest
```

## PyCharm Configuration

See instructions in the project documentation for configuring PyCharm to use `uv run` by default.

