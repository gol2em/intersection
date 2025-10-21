"""
Pytest tests for geometry classes: Line2D, Line3D, Curve, Surface
"""

import pytest
import numpy as np
from src.intersection.geometry import Line2D, Line3D, Curve, Surface


class TestLine2D:
    """Tests for Line2D class (implicit form: ax + by + c = 0)"""
    
    def test_creation_direct(self):
        """Test direct creation with coefficients"""
        line = Line2D(a=1, b=-1, c=0)
        assert line.a == 1
        assert line.b == -1
        assert line.c == 0
    
    def test_creation_invalid(self):
        """Test that creation fails when a=b=0"""
        with pytest.raises(ValueError, match="cannot both be zero"):
            Line2D(a=0, b=0, c=1)
    
    def test_from_two_points(self):
        """Test creation from two points"""
        line = Line2D.from_two_points((0, 0), (1, 1))
        # Should create a line through origin with slope 1
        # Line equation: y - x = 0, or -x + y = 0
        assert line.distance_to_point((1, 1)) < 1e-10
        assert line.distance_to_point((2, 2)) < 1e-10
    
    def test_from_point_and_direction(self):
        """Test creation from point and direction"""
        line = Line2D.from_point_and_direction((0, 0), (1, 1))
        # Should create same line as from_two_points
        assert line.distance_to_point((1, 1)) < 1e-10
        assert line.distance_to_point((2, 2)) < 1e-10
    
    def test_evaluate_y(self):
        """Test evaluating y for given x"""
        line = Line2D(a=1, b=-1, c=0)  # x - y = 0, so y = x
        assert np.isclose(line.evaluate_y(2.0), 2.0)
        assert np.isclose(line.evaluate_y(5.0), 5.0)
    
    def test_evaluate_y_vertical_line(self):
        """Test that evaluate_y fails for vertical line"""
        line = Line2D(a=1, b=0, c=-2)  # x = 2 (vertical)
        with pytest.raises(ValueError, match="vertical line"):
            line.evaluate_y(1.0)
    
    def test_evaluate_x(self):
        """Test evaluating x for given y"""
        line = Line2D(a=1, b=-1, c=0)  # x - y = 0, so x = y
        assert np.isclose(line.evaluate_x(3.0), 3.0)
        assert np.isclose(line.evaluate_x(7.0), 7.0)
    
    def test_evaluate_x_horizontal_line(self):
        """Test that evaluate_x fails for horizontal line"""
        line = Line2D(a=0, b=1, c=-3)  # y = 3 (horizontal)
        with pytest.raises(ValueError, match="horizontal line"):
            line.evaluate_x(1.0)
    
    def test_distance_to_point(self):
        """Test distance calculation"""
        line = Line2D(a=1, b=-1, c=0)  # x - y = 0
        # Point (1, 0) should be at distance 1/sqrt(2)
        dist = line.distance_to_point((1, 0))
        assert np.isclose(dist, 1.0 / np.sqrt(2))
    
    def test_repr(self):
        """Test string representation"""
        line = Line2D(a=1, b=-1, c=0)
        repr_str = repr(line)
        assert "Line2D" in repr_str
        assert "1.000x" in repr_str


class TestLine3D:
    """Tests for Line3D class (two planes)"""
    
    def test_creation_direct(self):
        """Test direct creation with plane coefficients"""
        line = Line3D(
            a1=0, b1=1, c1=0, d1=0,
            a2=0, b2=0, c2=1, d2=0
        )
        assert line.a1 == 0
        assert line.b1 == 1
    
    def test_creation_parallel_planes(self):
        """Test that creation fails for parallel planes"""
        with pytest.raises(ValueError, match="not be parallel"):
            Line3D(
                a1=1, b1=0, c1=0, d1=0,
                a2=1, b2=0, c2=0, d2=1  # Parallel to first plane
            )
    
    def test_from_point_and_direction(self):
        """Test creation from point and direction"""
        line = Line3D.from_point_and_direction((0, 0, 0), (1, 0, 0))
        direction = line.get_direction()
        # Direction should be along x-axis
        assert np.allclose(np.abs(direction), [1, 0, 0])
    
    def test_get_direction(self):
        """Test getting direction vector"""
        line = Line3D.from_point_and_direction((0, 0, 0), (1, 2, 3))
        direction = line.get_direction()
        # Should be normalized
        assert np.isclose(np.linalg.norm(direction), 1.0)
        # Should be parallel to (1, 2, 3)
        expected = np.array([1, 2, 3]) / np.linalg.norm([1, 2, 3])
        assert np.allclose(np.abs(direction), np.abs(expected))
    
    def test_get_point(self):
        """Test getting a point on the line"""
        line = Line3D.from_point_and_direction((1, 2, 3), (1, 0, 0))
        point = line.get_point()
        # Point should satisfy both plane equations
        assert np.isclose(
            line.a1 * point[0] + line.b1 * point[1] + line.c1 * point[2] + line.d1,
            0.0
        )
        assert np.isclose(
            line.a2 * point[0] + line.b2 * point[1] + line.c2 * point[2] + line.d2,
            0.0
        )
    
    def test_distance_to_point(self):
        """Test distance calculation"""
        # Line along x-axis through origin
        line = Line3D.from_point_and_direction((0, 0, 0), (1, 0, 0))
        # Point (0, 1, 0) should be at distance 1
        dist = line.distance_to_point((0, 1, 0))
        assert np.isclose(dist, 1.0)
    
    def test_repr(self):
        """Test string representation"""
        line = Line3D.from_point_and_direction((0, 0, 0), (1, 0, 0))
        repr_str = repr(line)
        assert "Line3D" in repr_str
        assert "Plane1" in repr_str
        assert "Plane2" in repr_str


class TestCurve:
    """Tests for Curve class (automatic interpolation)"""
    
    def test_creation_simple(self):
        """Test creation with simple functions"""
        curve = Curve(
            x_func=lambda u: u,
            y_func=lambda u: u**2,
            u_range=(0, 1),
            degree=3,
            verbose=False
        )
        assert curve.degree == 3
        assert curve.u_range == (0, 1)
    
    def test_automatic_interpolation(self):
        """Test that interpolation happens automatically"""
        curve = Curve(
            x_func=lambda u: u,
            y_func=lambda u: u**2,
            u_range=(0, 1),
            degree=3,
            verbose=False
        )
        # Check that polynomials were created
        assert curve.poly_x is not None
        assert curve.poly_y is not None
        assert hasattr(curve.poly_x, 'coef')
    
    def test_automatic_bernstein_conversion(self):
        """Test that Bernstein conversion happens automatically"""
        curve = Curve(
            x_func=lambda u: u,
            y_func=lambda u: u**2,
            u_range=(0, 1),
            degree=3,
            verbose=False
        )
        # Check that Bernstein coefficients were created
        assert curve.bernstein_x is not None
        assert curve.bernstein_y is not None
        assert isinstance(curve.bernstein_x, np.ndarray)
        assert len(curve.bernstein_x) == 4  # degree + 1
    
    def test_evaluate(self):
        """Test evaluation at parameter value"""
        curve = Curve(
            x_func=lambda u: u,
            y_func=lambda u: 2*u,
            u_range=(0, 1),
            degree=3,
            verbose=False
        )
        point = curve.evaluate(0.5)
        assert np.allclose(point, [0.5, 1.0])
    
    def test_sample(self):
        """Test sampling multiple points"""
        curve = Curve(
            x_func=lambda u: u,
            y_func=lambda u: u**2,
            u_range=(0, 1),
            degree=3,
            verbose=False
        )
        points = curve.sample(n_points=10)
        assert points.shape == (10, 2)
        # First point should be at u=0
        assert np.allclose(points[0], [0, 0])
        # Last point should be at u=1
        assert np.allclose(points[-1], [1, 1])
    
    def test_verbose_mode(self, capsys):
        """Test that verbose mode prints output"""
        curve = Curve(
            x_func=lambda u: u,
            y_func=lambda u: u**2,
            u_range=(0, 1),
            degree=3,
            verbose=True
        )
        captured = capsys.readouterr()
        assert "Curve Initialization" in captured.out
        assert "Interpolation" in captured.out
        assert "Bernstein" in captured.out
    
    def test_repr(self):
        """Test string representation"""
        curve = Curve(
            x_func=lambda u: u,
            y_func=lambda u: u**2,
            u_range=(0, 1),
            degree=5,
            verbose=False
        )
        repr_str = repr(curve)
        assert "Curve" in repr_str
        assert "u_range" in repr_str
        assert "degree=5" in repr_str


class TestSurface:
    """Tests for Surface class (automatic interpolation)"""
    
    def test_creation_simple(self):
        """Test creation with simple functions"""
        surface = Surface(
            x_func=lambda u, v: u,
            y_func=lambda u, v: v,
            z_func=lambda u, v: u + v,
            u_range=(0, 1),
            v_range=(0, 1),
            degree=3,
            verbose=False
        )
        assert surface.degree == 3
        assert surface.u_range == (0, 1)
        assert surface.v_range == (0, 1)
    
    def test_automatic_interpolation(self):
        """Test that interpolation happens automatically"""
        surface = Surface(
            x_func=lambda u, v: u,
            y_func=lambda u, v: v,
            z_func=lambda u, v: u**2 + v**2,
            u_range=(0, 1),
            v_range=(0, 1),
            degree=3,
            verbose=False
        )
        # Check that polynomial matrices were created
        assert surface.poly_x is not None
        assert surface.poly_y is not None
        assert surface.poly_z is not None
        assert isinstance(surface.poly_x, np.ndarray)
    
    def test_automatic_bernstein_conversion(self):
        """Test that Bernstein conversion happens automatically"""
        surface = Surface(
            x_func=lambda u, v: u,
            y_func=lambda u, v: v,
            z_func=lambda u, v: u + v,
            u_range=(0, 1),
            v_range=(0, 1),
            degree=3,
            verbose=False
        )
        # Check that Bernstein coefficient matrices were created
        assert surface.bernstein_x is not None
        assert surface.bernstein_y is not None
        assert surface.bernstein_z is not None
        assert isinstance(surface.bernstein_x, np.ndarray)
        assert surface.bernstein_x.shape == (4, 4)  # (degree+1, degree+1)
    
    def test_evaluate(self):
        """Test evaluation at parameter values"""
        surface = Surface(
            x_func=lambda u, v: u,
            y_func=lambda u, v: v,
            z_func=lambda u, v: u + v,
            u_range=(0, 1),
            v_range=(0, 1),
            degree=3,
            verbose=False
        )
        point = surface.evaluate(0.5, 0.5)
        assert np.allclose(point, [0.5, 0.5, 1.0])
    
    def test_sample(self):
        """Test sampling grid of points"""
        surface = Surface(
            x_func=lambda u, v: u,
            y_func=lambda u, v: v,
            z_func=lambda u, v: 0.0 * u * v,  # Constant zero function
            u_range=(0, 1),
            v_range=(0, 1),
            degree=3,
            verbose=False
        )
        X, Y, Z = surface.sample(n_u=5, n_v=5)
        assert X.shape == (5, 5)
        assert Y.shape == (5, 5)
        assert Z.shape == (5, 5)
    
    def test_verbose_mode(self, capsys):
        """Test that verbose mode prints output"""
        surface = Surface(
            x_func=lambda u, v: u,
            y_func=lambda u, v: v,
            z_func=lambda u, v: u + v,
            u_range=(0, 1),
            v_range=(0, 1),
            degree=3,
            verbose=True
        )
        captured = capsys.readouterr()
        assert "Surface Initialization" in captured.out
        assert "Interpolation" in captured.out
        assert "Bernstein" in captured.out
    
    def test_repr(self):
        """Test string representation"""
        surface = Surface(
            x_func=lambda u, v: u,
            y_func=lambda u, v: v,
            z_func=lambda u, v: u + v,
            u_range=(0, 1),
            v_range=(0, 1),
            degree=5,
            verbose=False
        )
        repr_str = repr(surface)
        assert "Surface" in repr_str
        assert "u_range" in repr_str
        assert "v_range" in repr_str
        assert "degree=5" in repr_str

