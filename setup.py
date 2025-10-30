from setuptools import setup, find_packages

setup(
    name="echo_i",
    version="0.1.0",
    description="Echo(I) - 6.S890 Course Project",
    author="Your Name",
    author_email="your.email@mit.edu",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
    },
)


