from setuptools import setup, find_packages
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="pytradepath",
    version="1.0.0",
    author="Sparky",
    author_email="sparkyofficialmail@proton.me",
    description="A comprehensive framework for backtesting and algorithmic trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pytradepath",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*", "data", "data.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.6",
    install_requires=[],
    extras_require={
        "dev": [
            "pytest>=6.0",
        ],
        "optional": [
            "psutil>=5.8.0",  # For memory monitoring
            "markdown>=3.3.0",  # For HTML documentation generation
        ],
    },
    entry_points={
        "console_scripts": [
            "pytradepath=pytradepath.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)