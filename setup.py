"""Setup configuration for the intersection package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="intersection",
    version="0.1.0",
    author="gol2em",
    description="Polynomial system solver using LP method with Bernstein basis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gol2em/intersection",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
        "optional": [
            "sympy>=1.12",
            "mpmath>=1.3.0",
            "matplotlib>=3.7.0",
        ],
    },
)

