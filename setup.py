from setuptools import setup, find_packages

setup(
    name="unit_disk_mapping",
    version="0.1.0",
    description="Map optimization problems to unit disk graphs for quantum computing",
    author="Python Port",
    author_email="example@example.com",
    url="https://github.com/yourusername/unit_disk_mapping",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # Include example scripts
    package_data={
        "": ["examples/*.py", "README.md"]
    },
    # Include test files in development mode
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
        ],
    },
    install_requires=[
        "networkx>=2.5",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.7",
)