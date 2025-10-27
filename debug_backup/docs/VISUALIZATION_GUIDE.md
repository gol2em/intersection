# Visualization Guide - 2D Polynomial System Solver

## Quick Start

To see the complete visualization suite for the 2D circle-ellipse intersection:

```bash
# Generate step-by-step 3D visualizations with projections
uv run python examples/visualize_2d_step_by_step.py

# Generate projection demonstrations
uv run python examples/demo_projection_walls.py

# Generate top-down 2D views
uv run python examples/visualize_2d_topdown.py
```

## Generated Visualizations

### 1. Step-by-Step 3D Views (8 files)

**Files**: `debug_2d_step_0001_depth0_subdivide.png` through `debug_2d_step_0008_depth1_solution.png`

**What they show**:
- Three panels per step:
  - Left: Equation 1 (circle) surface
  - Middle: Equation 2 (ellipse) surface
  - Right: Both surfaces combined with bounds

**Key features**:
- ✅ 3D polynomial surfaces
- ✅ Curves where surfaces cross z=0 (thick colored lines)
- ✅ Control points in 3D (black dots)
- ✅ **Projections on background walls**:
  - Back wall (y=1): (x, f) projection with convex hull
  - Side wall (x=0): (y, f) projection with convex hull
  - Thick lines at f=0: PP bounds for each dimension
- ✅ Current box (black) and PP bounds (green) on z=0 plane
- ✅ Expected solution (gold star)

**Step progression**:
1. **Step 1**: Initial box [0,1]² → Subdivide x
2. **Step 2**: Left half → PRUNED (no solution)
3. **Step 3**: Right half → TIGHTEN (both dims reduced)
4. **Steps 4-7**: Progressive tightening
5. **Step 8**: SOLUTION found!

### 2. Projection Demonstrations (2 files)

**Files**: 
- `projection_demo_circle.png` - Circle equation
- `projection_demo_ellipse.png` - Ellipse equation

**What they show**:
- Single equation at a time for clarity
- Large, clear view of projections on walls
- Perfect for understanding the PP method

**Features**:
- Control points in 3D (black dots with white edges)
- Projections on walls (colored dots)
- Convex hulls (dashed lines)
- PP bounds (thick solid lines at f=0)
- Semi-transparent background walls

**Use case**: Educational - shows exactly how PP method projects control points

### 3. Top-Down 2D Views (4 files)

**Files**:
- `topdown_step1_initial.png` - Initial box [0,1]²
- `topdown_step2_right_half.png` - After first subdivision
- `topdown_step3_tightened.png` - After first tightening
- `topdown_step4_near_solution.png` - Near solution

**What they show**:
- Bird's-eye view of the z=0 plane
- Curves (circle and ellipse) as contour lines
- Bounding boxes (black = current, green = PP bounds)
- Expected solution (gold star)

**Features**:
- Clear 2D view without 3D complexity
- Shows progressive tightening of bounds
- Easy to see how boxes shrink around solution

## Understanding the Visualizations

### Color Coding

| Color | Meaning |
|-------|---------|
| 🔵 Blue | Equation 1 (circle: x² + y² - 1 = 0) |
| 🔴 Red | Equation 2 (ellipse: x²/4 + 4y² - 1 = 0) |
| 🟢 Green | z=0 plane and PP bounds |
| ⚫ Black | Control points and current box |
| ⭐ Gold | Expected solution |

### What Each Element Shows

**3D Surfaces**:
- The actual polynomial f(x, y)
- Semi-transparent to see through

**Curves on z=0**:
- Thick colored lines
- Where f(x, y) = 0 (the solution curve)
- Intersection of these curves = solution point

**Control Points**:
- Black dots in 3D space
- Bernstein polynomial control points
- Their convex hull bounds the surface

**Projections on Walls**:

**Back Wall (y = y_max)**:
- Shows (x, f) projection
- Colored dots: projected control points
- Dashed lines: convex hull
- Thick line at f=0: PP bounds for x

**Side Wall (x = x_min)**:
- Shows (y, f) projection
- Colored dots: projected control points
- Dashed lines: convex hull
- Thick line at f=0: PP bounds for y

**Bounding Boxes on z=0**:
- Black rectangle: current search box
- Green rectangle: PP-tightened bounds
- Shows how PP method narrows the search

## How to Read the Visualizations

### Step-by-Step Process

1. **Look at the curves**: Where do they intersect? That's the solution!

2. **Check the current box**: Black rectangle on z=0 plane

3. **See PP bounds**: Green rectangle - tighter than current box

4. **Examine projections on walls**:
   - Back wall: How tight are x-bounds?
   - Side wall: How tight are y-bounds?

5. **Watch the progression**: Each step shows tightening or subdivision

### Reading the Projections

**On the back wall (y=1)**:
1. Find the colored dots (projected control points)
2. Follow the dashed lines (convex hull)
3. Look at where hull crosses f=0 (thick line)
4. This is the PP bound for x-dimension!

**On the side wall (x=0)**:
1. Find the colored dots (projected control points)
2. Follow the dashed lines (convex hull)
3. Look at where hull crosses f=0 (thick line)
4. This is the PP bound for y-dimension!

## Example: Step 1 Analysis

**File**: `debug_2d_step_0001_depth0_subdivide.png`

**What's happening**:
- Initial box: [0, 1] × [0, 1]
- PP bounds: [0, 1] × [0.1875, 0.625]

**On back wall (x-projection)**:
- Circle: Convex hull crosses f=0 at x ∈ [0, 1]
- Ellipse: Convex hull crosses f=0 at x ∈ [0, 1]
- Combined: x ∈ [0, 1] (no tightening)

**On side wall (y-projection)**:
- Circle: Convex hull crosses f=0 at y ∈ [0, 1]
- Ellipse: Convex hull crosses f=0 at y ∈ [0.1875, 0.625]
- Combined: y ∈ [0.1875, 0.625] (good tightening!)

**Decision**: Subdivide x-dimension (didn't tighten)

## Example: Step 8 Analysis

**File**: `debug_2d_step_0008_depth1_solution.png`

**What's happening**:
- Box is very small: ~0.001 × 0.0006
- Contains the solution
- Marked as SOLUTION

**On the walls**:
- Both projections show very tight bounds
- Convex hulls are small
- PP bounds closely match the box

**Result**: Solution found at (0.894425, 0.447221) ✓

## Comparing Visualizations

### 3D Step-by-Step vs Projection Demo

**3D Step-by-Step**:
- Shows the solving process
- Multiple steps
- Both equations together
- Smaller projections (to fit everything)

**Projection Demo**:
- Shows the method clearly
- Single snapshot
- One equation at a time
- Larger, clearer projections

### 3D Views vs Top-Down Views

**3D Views**:
- Shows surfaces and projections
- More information
- Better for understanding PP method
- Can be complex

**Top-Down Views**:
- Simpler, clearer
- Just curves and boxes
- Better for seeing convergence
- Easy to understand

## Tips for Best Understanding

1. **Start with projection demos**: Understand how PP method works
2. **Then watch step-by-step**: See it in action
3. **Check top-down views**: Verify convergence
4. **Compare steps**: See how bounds tighten

## Technical Details

### Viewing Angles

**3D views**: 
- Elevation: 20°
- Azimuth: 45°
- Chosen to show both walls clearly

**Top-down views**:
- Pure 2D (looking down at z=0 plane)
- Equal aspect ratio

### Resolution

- All images: 150 DPI
- 3D step-by-step: 16×6 inches (three panels)
- Projection demos: 14×10 inches (single large view)
- Top-down: 10×10 inches (square)

## Troubleshooting

**Can't see projections on walls?**
- Look for colored dots on the back and side walls
- They might be small - zoom in on the image

**Walls look empty?**
- Check if control points are all on one side of f=0
- If so, no intersection → no PP bounds

**Convex hull looks wrong?**
- It's correct! The hull might be surprising
- Remember: it's the convex hull of PROJECTED points

## Summary

The visualization suite provides:
- ✅ Complete view of the solving process
- ✅ Clear explanation of PP method
- ✅ Visual verification of correctness
- ✅ Educational value for understanding the algorithm
- ✅ Debugging capability for development

Enjoy exploring the visualizations!

