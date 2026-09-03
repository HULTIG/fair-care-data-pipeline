from setuptools import setup, find_packages

setup(
    name="pace",
    version="1.0.0",
    description="PACE Lakehouse: Ethical AI Data Governance Pipeline",
    author="Anonymous for Review",
    license="Apache-2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pyspark>=3.5.0",
        "delta-spark>=3.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "aif360>=0.6.0",
        "diffprivlib>=0.6.0",
        "dowhy>=0.11.0",
        "sentence-transformers>=2.2.0",
        "huggingface-hub>=0.25.0",
        "pyyaml>=6.0.0",
        "requests>=2.31.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
    ],
    extras_require={
        "dev": ["black", "flake8", "mypy", "jupyter"],
    },
    entry_points={
        "console_scripts": [
            "pace-pipeline=pace.orchestration.pipeline:main",
        ],
    },
)
