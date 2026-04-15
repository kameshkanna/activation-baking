"""
Setup configuration for the activation-baking package.

Targets: Python >= 3.10, Lambda Labs A100 / local development.
"""

from pathlib import Path

from setuptools import find_packages, setup

_HERE = Path(__file__).parent
_LONG_DESC = (_HERE / "README.md").read_text(encoding="utf-8") if (_HERE / "README.md").exists() else ""

setup(
    name="activation-baking",
    version="0.1.0",
    description=(
        "PCA-directed activation adapters via weight-space symmetry calibration. "
        "Research framework for ICML 2026 Workshop on Weight-Space Symmetries."
    ),
    long_description=_LONG_DESC,
    long_description_content_type="text/markdown",
    author="Activation Baking Research Team",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "experiments*", "analysis*", "paper*"]),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
        ],
        "quant": [
            "bitsandbytes>=0.43.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "activation steering",
        "mechanistic interpretability",
        "weight-space symmetries",
        "PCA",
        "language models",
        "ICML 2026",
    ],
)
