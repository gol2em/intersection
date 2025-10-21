"""
Pytest tests for interpolation module
"""

import pytest
import numpy as np
from numpy.polynomial import Polynomial
from src.intersection.geometry import Curve, Surface
from src.intersection.interpolation import interpolate_curve, interpolate_surface


class TestInterpolateCurve:
    """Tests for curve interpolation"""
    
    def test_linear_function(self):
        """Test interpolation of linear function (should be exact)"""
        curve = Curve(
            x_func=lambda u: u,
            y_func=lambda u: 2*u + 1,
            u_range=(0, 1),
            degree=1,
            verbose=False
        )
        
        # For linear function, interpolation should be nearly exact
        poly_x, poly_y = curve.poly_x, curve.poly_y
        
        # Test at several points
        for u in [0, 0.25, 0.5, 0.75, 1.0]:
            assert np.isclose(poly_x(u), u, atol=1e-10)
            assert np.isclose(poly_y(u), 2*u + 1, atol=1e-10)
    
    def test_quadratic_function(self):
        """Test interpolation of quadratic function"""
        curve = Curve(
            x_func=lambda u: u,
            y_func=lambda u: u**2,
            u_range=(0, 1),
            degree=2,
            verbose=False
        )
        
        poly_x, poly_y = curve.poly_x, curve.poly_y
        
        # Test at several points
        for u in [0, 0.5, 1.0]:
            assert np.isclose(poly_x(u), u, atol=1e-10)
            assert np.isclose(poly_y(u), u**2, atol=1e-10)
    
    def test_polynomial_degree(self):
        """Test that polynomial has correct degree"""
        curve = Curve(
            x_func=lambda u: u,
            y_func=lambda u: u**2,
            u_range=(0, 1),
            degree=5,
            verbose=False
        )
        
        # Polynomial coefficients should have length degree+1
        assert len(curve.poly_x.coef) == 6
        assert len(curve.poly_y.coef) == 6
    
    def test_different_ranges(self):
        """Test interpolation with different parameter ranges"""
        curve = Curve(
            x_func=lambda u: u,
            y_func=lambda u: u**2,
            u_range=(-1, 1),
            degree=3,
            verbose=False
        )
        
        # Test at range boundaries
        assert np.isclose(curve.poly_x(-1), -1, atol=1e-10)
        assert np.isclose(curve.poly_x(1), 1, atol=1e-10)
        assert np.isclose(curve.poly_y(-1), 1, atol=1e-10)
        assert np.isclose(curve.poly_y(1), 1, atol=1e-10)
    
    def test_verbose_output(self, capsys):
        """Test that verbose mode produces output"""
        curve = Curve(
            x_func=lambda u: u,
            y_func=lambda u: u**2,
            u_range=(0, 1),
            degree=3,
            verbose=True
        )
        
        captured = capsys.readouterr()
        assert "Chebyshev nodes" in captured.out
        assert "Polynomial degree" in captured.out
        assert "error" in captured.out.lower()


class TestInterpolateSurface:
    """Tests for surface interpolation"""
    
    def test_planar_surface(self):
        """Test interpolation of planar surface"""
        surface = Surface(
            x_func=lambda u, v: u,
            y_func=lambda u, v: v,
            z_func=lambda u, v: u + v,
            u_range=(0, 1),
            v_range=(0, 1),
            degree=1,
            verbose=False
        )
        
        # For planar surface, interpolation should be good
        # Test at corners
        test_points = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for u, v in test_points:
            point = surface.evaluate(u, v)
            assert np.isclose(point[0], u, atol=0.1)
            assert np.isclose(point[1], v, atol=0.1)
            assert np.isclose(point[2], u + v, atol=0.1)
    
    def test_polynomial_shape(self):
        """Test that polynomial matrices have correct shape"""
        surface = Surface(
            x_func=lambda u, v: u,
            y_func=lambda u, v: v,
            z_func=lambda u, v: u**2 + v**2,
            u_range=(0, 1),
            v_range=(0, 1),
            degree=3,
            verbose=False
        )
        
        # Polynomial coefficient matrices should be (degree+1) x (degree+1)
        assert surface.poly_x.shape == (4, 4)
        assert surface.poly_y.shape == (4, 4)
        assert surface.poly_z.shape == (4, 4)
    
    def test_different_ranges(self):
        """Test interpolation with different parameter ranges"""
        surface = Surface(
            x_func=lambda u, v: u,
            y_func=lambda u, v: v,
            z_func=lambda u, v: u * v,
            u_range=(-1, 1),
            v_range=(-1, 1),
            degree=3,
            verbose=False
        )
        
        # Test at center
        point = surface.evaluate(0, 0)
        assert np.isclose(point[0], 0, atol=0.1)
        assert np.isclose(point[1], 0, atol=0.1)
        assert np.isclose(point[2], 0, atol=0.1)
    
    def test_verbose_output(self, capsys):
        """Test that verbose mode produces output"""
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
        assert "Chebyshev grid" in captured.out
        assert "Polynomial degree" in captured.out
        assert "error" in captured.out.lower()


class TestInterpolationAccuracy:
    """Tests for interpolation accuracy"""
    
    def test_curve_accuracy_increases_with_degree(self):
        """Test that higher degree gives better accuracy"""
        def test_func(u):
            return np.sin(2 * np.pi * u)
        
        # Low degree
        curve_low = Curve(
            x_func=lambda u: u,
            y_func=test_func,
            u_range=(0, 1),
            degree=3,
            verbose=False
        )
        
        # High degree
        curve_high = Curve(
            x_func=lambda u: u,
            y_func=test_func,
            u_range=(0, 1),
            degree=8,
            verbose=False
        )
        
        # Test at midpoint
        u_test = 0.5
        error_low = abs(curve_low.poly_y(u_test) - test_func(u_test))
        error_high = abs(curve_high.poly_y(u_test) - test_func(u_test))
        
        # Higher degree should have lower error (usually)
        # Note: This might not always be true due to Runge's phenomenon,
        # but with Chebyshev nodes it should be
        assert error_high < error_low or error_high < 0.01
    
    def test_curve_chebyshev_nodes_used(self):
        """Test that Chebyshev nodes are being used"""
        # This is implicit in the implementation, but we can verify
        # that the interpolation is stable even for high degrees
        curve = Curve(
            x_func=lambda u: u,
            y_func=lambda u: 1 / (1 + 25 * (u - 0.5)**2),  # Runge's function
            u_range=(0, 1),
            degree=10,
            verbose=False
        )
        
        # With Chebyshev nodes, this should not oscillate wildly
        # Test at several points
        u_test = np.linspace(0, 1, 20)
        for u in u_test:
            y_true = 1 / (1 + 25 * (u - 0.5)**2)
            y_interp = curve.poly_y(u)
            # Error should be reasonable (not huge oscillations)
            assert abs(y_interp - y_true) < 1.0  # Generous bound

