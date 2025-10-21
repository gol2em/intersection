"""
Utility functions for visualization and debugging.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_2d(line, curve, intersections=None, title="2D Line-Curve Intersection"):
    """
    Visualize 2D line-curve intersection.
    
    Parameters
    ----------
    line : Line2D
        The straight line
    curve : ParametricCurve
        The parametric curve
    intersections : list of dict, optional
        Intersection points to highlight
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot curve
    t_values = np.linspace(curve.t_range[0], curve.t_range[1], 200)
    curve_points = curve.evaluate(t_values)
    ax.plot(curve_points[:, 0], curve_points[:, 1], 'b-', linewidth=2, label='Curve')
    
    # Plot line
    # Find reasonable bounds for line visualization
    x_min, x_max = curve_points[:, 0].min(), curve_points[:, 0].max()
    y_min, y_max = curve_points[:, 1].min(), curve_points[:, 1].max()
    
    # Extend line beyond curve bounds
    margin = 0.2
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Find s values that cover the plot area
    s_values = np.linspace(-max(x_range, y_range), max(x_range, y_range), 100)
    line_points = np.array([line.evaluate(s) for s in s_values])
    
    ax.plot(line_points[:, 0], line_points[:, 1], 'r--', linewidth=2, label='Line')
    
    # Plot intersections
    if intersections:
        for i, intersection in enumerate(intersections):
            point = intersection['point']
            ax.plot(point[0], point[1], 'go', markersize=12, 
                   label=f'Intersection {i+1}' if i == 0 else '')
            ax.annotate(f't={intersection["parameter"]:.3f}', 
                       xy=point, xytext=(10, 10), 
                       textcoords='offset points',
                       fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    return fig, ax


def visualize_3d(line, surface, intersections=None, title="3D Line-Surface Intersection"):
    """
    Visualize 3D line-surface intersection.
    
    Parameters
    ----------
    line : Line3D
        The straight line
    surface : ParametricSurface
        The parametric surface
    intersections : list of dict, optional
        Intersection points to highlight
    title : str
        Plot title
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    X, Y, Z = surface.sample(n_u=30, n_v=30)
    ax.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis', edgecolor='none')
    
    # Plot line
    # Find reasonable bounds
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    z_min, z_max = Z.min(), Z.max()
    
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    
    s_values = np.linspace(-max_range, max_range, 100)
    line_points = np.array([line.evaluate(s) for s in s_values])
    
    ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 
           'r-', linewidth=3, label='Line')
    
    # Plot intersections
    if intersections:
        for i, intersection in enumerate(intersections):
            point = intersection['point']
            ax.scatter(point[0], point[1], point[2], 
                      c='green', s=200, marker='o', edgecolors='black', linewidths=2,
                      label=f'Intersection {i+1}' if i == 0 else '')
            
            # Add text annotation
            u, v = intersection['parameters']
            ax.text(point[0], point[1], point[2], 
                   f'  ({u:.2f},{v:.2f})', fontsize=9)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    
    # Set equal aspect ratio
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    mid_x = (x_max + x_min) * 0.5
    mid_y = (y_max + y_min) * 0.5
    mid_z = (z_max + z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    return fig, ax


def print_intersection_summary(intersections, dimension='2D'):
    """
    Print a summary of intersection results.
    
    Parameters
    ----------
    intersections : list of dict
        Intersection results
    dimension : str
        '2D' or '3D'
    """
    print("\n" + "=" * 70)
    print(f"INTERSECTION SUMMARY ({dimension})")
    print("=" * 70)
    
    if not intersections:
        print("No intersections found.")
        return
    
    print(f"Total intersections found: {len(intersections)}\n")
    
    for i, intersection in enumerate(intersections, 1):
        print(f"Intersection {i}:")
        
        if dimension == '2D':
            t = intersection['parameter']
            point = intersection['point']
            dist = intersection['distance_to_line']
            
            print(f"  Parameter t: {t:.8f}")
            print(f"  Point: ({point[0]:.8f}, {point[1]:.8f})")
            print(f"  Distance to line: {dist:.2e}")
        
        elif dimension == '3D':
            u, v = intersection['parameters']
            point = intersection['point']
            dist = intersection['distance_to_line']
            
            print(f"  Parameters (u, v): ({u:.8f}, {v:.8f})")
            print(f"  Point: ({point[0]:.8f}, {point[1]:.8f}, {point[2]:.8f})")
            print(f"  Distance to line: {dist:.2e}")
        
        print()
    
    print("=" * 70)


def save_results(intersections, filename, dimension='2D'):
    """
    Save intersection results to a file.
    
    Parameters
    ----------
    intersections : list of dict
        Intersection results
    filename : str
        Output filename
    dimension : str
        '2D' or '3D'
    """
    with open(filename, 'w') as f:
        f.write(f"Intersection Results ({dimension})\n")
        f.write("=" * 70 + "\n\n")
        
        if not intersections:
            f.write("No intersections found.\n")
            return
        
        f.write(f"Total intersections: {len(intersections)}\n\n")
        
        for i, intersection in enumerate(intersections, 1):
            f.write(f"Intersection {i}:\n")
            
            if dimension == '2D':
                t = intersection['parameter']
                point = intersection['point']
                dist = intersection['distance_to_line']
                
                f.write(f"  Parameter t: {t:.12f}\n")
                f.write(f"  Point: ({point[0]:.12f}, {point[1]:.12f})\n")
                f.write(f"  Distance to line: {dist:.6e}\n")
            
            elif dimension == '3D':
                u, v = intersection['parameters']
                point = intersection['point']
                dist = intersection['distance_to_line']
                
                f.write(f"  Parameters (u, v): ({u:.12f}, {v:.12f})\n")
                f.write(f"  Point: ({point[0]:.12f}, {point[1]:.12f}, {point[2]:.12f})\n")
                f.write(f"  Distance to line: {dist:.6e}\n")
            
            f.write("\n")
    
    print(f"Results saved to {filename}")

